import os
import sys
import json
import logging
from datetime import date
from pathlib import Path
from dotenv import load_dotenv

import boto3
import PyPDF2
from botocore.exceptions import ClientError

from agents.agent_0a_profiler import build_candidate_profile
from agents.agent_0b_scraper import scrape_jobs, JobListing
from agents.agent_0c_ranker import rank_and_filter_jobs
from agents.agent_1 import run_tailor, run_tailor_gemini
from agents.agent_2 import publish_to_s3

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("main")

RESUME_PDF_PATH = Path("data/master_resume.pdf")
SEEN_JOBS_S3_KEY = "state/seen_jobs.json"
STATE_VERSION = 2


# ── Seen-jobs deduplication ───────────────────────────────────────────────────
# State shape (v2):
#   {"version": 2, "jobs": {job_id: {status, first_seen, last_attempt, company, title}}}
# status is "published" (dedup on next run) or "failed" (retry on next run).

def _load_seen_state(s3_client, bucket: str) -> dict:
    """Load processed-jobs state from S3. Returns {job_id: entry}."""
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=SEEN_JOBS_S3_KEY)
        data = json.loads(obj["Body"].read())
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            logger.info("No seen-jobs state found in S3 — starting fresh")
            return {}
        logger.warning(f"Could not load seen-jobs state: {e}. Proceeding without dedup.")
        return {}

    if isinstance(data, dict) and data.get("version") == STATE_VERSION:
        jobs = data.get("jobs", {}) or {}
        published = sum(1 for e in jobs.values() if e.get("status") == "published")
        failed = sum(1 for e in jobs.values() if e.get("status") == "failed")
        logger.info(f"Loaded state: {published} published, {failed} failed (will retry)")
        return jobs

    # v1 shape: {"job_ids": [...]}. The hash scheme changed alongside v2, so
    # old IDs cannot match new hashes — drop them and reset dedup for this run.
    if isinstance(data, dict) and "job_ids" in data:
        logger.warning(
            f"Migrating from v1 state ({len(data['job_ids'])} entries). "
            f"Hash scheme changed — dropping old entries; dedup resets this run."
        )
        return {}

    logger.warning(f"Unknown state shape in {SEEN_JOBS_S3_KEY}; starting fresh")
    return {}


def _save_seen_state(s3_client, bucket: str, jobs: dict) -> None:
    try:
        body = {"version": STATE_VERSION, "jobs": jobs}
        s3_client.put_object(
            Bucket=bucket,
            Key=SEEN_JOBS_S3_KEY,
            Body=json.dumps(body, indent=2, sort_keys=True),
            ContentType="application/json",
        )
        logger.info(f"Saved state: {len(jobs)} total entries")
    except Exception as e:
        logger.warning(f"Could not save seen-jobs state: {e}")


def _filter_seen(jobs: list[JobListing], state: dict) -> list[JobListing]:
    """Drop jobs already successfully published. Failed jobs fall through for retry."""
    fresh = [j for j in jobs if state.get(j.job_id, {}).get("status") != "published"]
    skipped = len(jobs) - len(fresh)
    if skipped:
        logger.info(f"Dedup: skipped {skipped} already-published jobs, {len(fresh)} to process")
    return fresh


def _record_result(state: dict, job: JobListing, status: str) -> None:
    today = date.today().isoformat()
    existing = state.get(job.job_id, {})
    state[job.job_id] = {
        "status": status,
        "first_seen": existing.get("first_seen") or today,
        "last_attempt": today,
        "company": job.company,
        "title": job.title,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_resume_text(pdf_path: Path) -> str:
    text_parts = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
    return "\n".join(text_parts)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("AI Job Application Pipeline — Starting")
    logger.info("=" * 60)

    if not RESUME_PDF_PATH.exists():
        logger.error(f"Master resume not found at {RESUME_PDF_PATH}.")
        sys.exit(1)

    bucket_name = os.getenv("AWS_S3_BUCKET")
    if not bucket_name:
        logger.error("AWS_S3_BUCKET not set.")
        sys.exit(1)

    s3_client = boto3.client("s3")

    # ── Agent 0A: Build candidate profile ────────────────────────────
    logger.info("\n[AGENT 0A] Building candidate profile from resume PDF...")
    profile = build_candidate_profile(str(RESUME_PDF_PATH))

    master_resume_text = extract_resume_text(RESUME_PDF_PATH)
    logger.info(f"Resume text extracted: {len(master_resume_text)} characters")

    # ── Agent 0B: Scrape jobs ─────────────────────────────────────────
    logger.info("\n[AGENT 0B] Scraping jobs via JobSpy (LinkedIn + Indeed)...")
    raw_jobs = scrape_jobs(profile, target_raw=80)

    if not raw_jobs:
        logger.warning("No jobs found. Pipeline complete (nothing to process).")
        sys.exit(0)

    # ── Dedup: drop seen jobs BEFORE the ranker so we don't waste tokens
    #     scoring listings we've already processed in earlier runs. ────
    seen_state = _load_seen_state(s3_client, bucket_name)
    raw_jobs = _filter_seen(raw_jobs, seen_state)

    if not raw_jobs:
        logger.info("All scraped jobs were already processed in earlier runs. Nothing new.")
        sys.exit(0)

    logger.info(f"{len(raw_jobs)} fresh jobs to rank...")

    # ── Agent 0C: Rank and filter ─────────────────────────────────────
    logger.info("\n[AGENT 0C] Ranking and filtering jobs...")
    ranked_jobs = rank_and_filter_jobs(raw_jobs, profile, target_output=55, min_score=45)

    if not ranked_jobs:
        logger.warning("No jobs passed the ranking threshold.")
        sys.exit(0)

    logger.info(f"{len(ranked_jobs)} ranked jobs to tailor...")

    # ── Process each job: Agent 1 → Agent 2 ──────────────────────────
    # Tiered tailoring: top N jobs use Claude Sonnet (highest quality),
    # the remaining long tail uses Gemini Flash (free tier, ~1500 RPD).
    results_summary = []
    top_tier_count = int(os.getenv("TOP_TIER_CLAUDE_COUNT", "6"))

    for i, (job, score) in enumerate(ranked_jobs):
        use_claude = i < top_tier_count
        tier = "CLAUDE" if use_claude else "GEMINI"

        logger.info(f"\n{'─' * 50}")
        logger.info(f"[{tier}] Processing: {job.title} @ {job.company} (score: {score}/100)")

        try:
            tailor_fn = run_tailor if use_claude else run_tailor_gemini
            tailored = tailor_fn(job, profile, master_resume_text)
            uploads = publish_to_s3(job, tailored, profile, score)

            # Record success immediately — even if later jobs fail this one is done
            _record_result(seen_state, job, "published")

            results_summary.append({
                "job": f"{job.title} @ {job.company}",
                "status": "PUBLISHED",
                "files": uploads,
            })

        except Exception as e:
            logger.error(f"Pipeline failed for '{job.title}' @ {job.company}: {e}")
            _record_result(seen_state, job, "failed")
            results_summary.append({
                "job": f"{job.title} @ {job.company}",
                "status": "ERROR",
                "reason": str(e),
            })
            continue

    # ── Persist state so next run dedups published and retries failed ─
    _save_seen_state(s3_client, bucket_name, seen_state)

    # ── Final summary ─────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE — Summary")
    logger.info("=" * 60)

    published = [r for r in results_summary if r["status"] == "PUBLISHED"]
    errors = [r for r in results_summary if r["status"] == "ERROR"]

    logger.info(f"✅ Published: {len(published)}")
    logger.info(f"❌ Errors: {len(errors)}")

    for r in published:
        logger.info(f"  → {r['job']}")


if __name__ == "__main__":
    main()
