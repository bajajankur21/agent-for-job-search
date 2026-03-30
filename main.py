import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

import PyPDF2

from agents.agent_0a_profiler import build_candidate_profile
from agents.agent_0b_scraper import scrape_jobs
from agents.agent_0c_ranker import rank_and_filter_jobs
from agents.agent_1 import run_gatekeeper
from agents.agent_2 import run_tailor
from agents.agent_3 import publish_to_s3

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("main")

RESUME_PDF_PATH = Path("data/master_resume.pdf")


def extract_resume_text(pdf_path: Path) -> str:
    """Extracts plain text from the master resume PDF for Agent 2 caching."""
    text_parts = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
    return "\n".join(text_parts)


def main():
    logger.info("=" * 60)
    logger.info("AI Job Application Pipeline — Starting")
    logger.info("=" * 60)

    # Validate resume exists
    if not RESUME_PDF_PATH.exists():
        logger.error(
            f"Master resume not found at {RESUME_PDF_PATH}. "
            "Please upload data/master_resume.pdf to the repo."
        )
        sys.exit(1)

    # ── Agent 0A: Build candidate profile ────────────────────────────
    logger.info("\n[AGENT 0A] Building candidate profile from resume PDF...")
    profile = build_candidate_profile(str(RESUME_PDF_PATH))

    # Extract resume text once for Agent 2 caching
    master_resume_text = extract_resume_text(RESUME_PDF_PATH)
    logger.info(f"Resume text extracted: {len(master_resume_text)} characters")

    # ── Agent 0B: Scrape jobs ─────────────────────────────────────────
    logger.info("\n[AGENT 0B] Scraping jobs from Google Jobs via SerpAPI...")
    raw_jobs = scrape_jobs(profile, target_raw=150)

    if not raw_jobs:
        logger.warning("No jobs found. Pipeline complete (nothing to process).")
        sys.exit(0)

    # ── Agent 0C: Rank and filter ─────────────────────────────────────
    logger.info("\n[AGENT 0C] Ranking and filtering jobs...")
    ranked_jobs = rank_and_filter_jobs(raw_jobs, profile, target_output=55, min_score=45)

    if not ranked_jobs:
        logger.warning("No jobs passed the ranking threshold. Try lowering min_score.")
        sys.exit(0)

    logger.info(f"{len(ranked_jobs)} jobs passed ranking. Processing top jobs...")

    # ── Process each job: Agent 1 → 2 → 3 ───────────────────────────
    results_summary = []

    for job, score in ranked_jobs:
        logger.info(f"\n{'─'*50}")
        logger.info(f"Processing: {job.title} @ {job.company} (score: {score}/100)")

        try:
            # Agent 1: Gatekeeper
            gate_result = run_gatekeeper(job, max_yoe=profile.max_yoe_applying_for)
            if not gate_result.passed:
                results_summary.append({
                    "job": f"{job.title} @ {job.company}",
                    "status": "SKIPPED",
                    "reason": f"YOE requirement {gate_result.minimum_yoe} > threshold"
                })
                continue

            # Agent 2: Tailor
            tailored = run_tailor(job, profile, master_resume_text)

            # Agent 3: Publish
            uploads = publish_to_s3(job, tailored, profile.full_name)

            results_summary.append({
                "job": f"{job.title} @ {job.company}",
                "status": "PUBLISHED",
                "files": uploads
            })

        except Exception as e:
            logger.error(f"Pipeline failed for '{job.title}' @ {job.company}: {e}")
            results_summary.append({
                "job": f"{job.title} @ {job.company}",
                "status": "ERROR",
                "reason": str(e)
            })
            # Continue to next job — one failure doesn't kill the run
            continue

    # ── Final summary ─────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE — Summary")
    logger.info("=" * 60)

    published = [r for r in results_summary if r["status"] == "PUBLISHED"]
    skipped = [r for r in results_summary if r["status"] == "SKIPPED"]
    errors = [r for r in results_summary if r["status"] == "ERROR"]

    logger.info(f"✅ Published: {len(published)}")
    logger.info(f"⏭️  Skipped (YOE filter): {len(skipped)}")
    logger.info(f"❌ Errors: {len(errors)}")

    for r in published:
        logger.info(f"  → {r['job']}")


if __name__ == "__main__":
    main()