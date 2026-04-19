import os
import sys
import json
import logging
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


# ── Seen-jobs deduplication ───────────────────────────────────────────────────

def _load_seen_ids(s3_client, bucket: str) -> set[str]:
    """
    Loads the set of already-processed job IDs from S3.
    Returns an empty set if the file doesn't exist yet (first run).
    """
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=SEEN_JOBS_S3_KEY)
        data = json.loads(obj["Body"].read())
        seen = set(data.get("job_ids", []))
        logger.info(f"Loaded {len(seen)} previously processed job IDs from S3")
        return seen
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            logger.info("No seen-jobs state found in S3 — starting fresh")
            return set()
        logger.warning(f"Could not load seen-jobs state: {e}. Proceeding without dedup.")
        return set()


def _save_seen_ids(s3_client, bucket: str, seen_ids: set[str]) -> None:
    """Persists the seen job IDs back to S3 after the run."""
    try:
        s3_client.put_object(
            Bucket=bucket,
            Key=SEEN_JOBS_S3_KEY,
            Body=json.dumps({"job_ids": sorted(seen_ids)}, indent=2),
            ContentType="application/json",
        )
        logger.info(f"Saved {len(seen_ids)} seen job IDs to S3")
    except Exception as e:
        logger.warning(f"Could not save seen-jobs state: {e}")


def _filter_seen(jobs: list[JobListing], seen_ids: set[str]) -> list[JobListing]:
    """Removes jobs that have already been processed in a previous run."""
    fresh = [job for job in jobs if job.job_id not in seen_ids]
    skipped = len(jobs) - len(fresh)
    if skipped:
        logger.info(f"Dedup: skipped {skipped} already-processed jobs, {len(fresh)} new")
    return fresh


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
    seen_ids = _load_seen_ids(s3_client, bucket_name)
    raw_jobs = _filter_seen(raw_jobs, seen_ids)

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

            # Mark as processed immediately — even if later jobs fail this one is done
            seen_ids.add(job.job_id)

            results_summary.append({
                "job": f"{job.title} @ {job.company}",
                "status": "PUBLISHED",
                "files": uploads,
            })

        except Exception as e:
            logger.error(f"Pipeline failed for '{job.title}' @ {job.company}: {e}")
            results_summary.append({
                "job": f"{job.title} @ {job.company}",
                "status": "ERROR",
                "reason": str(e),
            })
            continue

    # ── Persist seen IDs so next run skips these jobs ─────────────────
    _save_seen_ids(s3_client, bucket_name, seen_ids)

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
