"""
Test script that mocks Agent 0A (profiler) to test Agent 0B (scraper)
and Agent 0C (ranker) without paying for Sonnet API calls.

Usage: python test_pipeline.py
"""

import sys
import logging
from dotenv import load_dotenv

from agents.agent_0a_profiler import CandidateProfile
from agents.agent_0b_scraper import scrape_jobs
from agents.agent_0c_ranker import rank_and_filter_jobs

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("test_pipeline")

# ── Mock profile (replace with values that match your actual resume) ──────
MOCK_PROFILE = CandidateProfile(
    full_name="Ankur Bajaj",
    current_title="Software Engineer",
    total_yoe=2.0,
    core_skills=["Python", "React", "Node.js", "AWS", "TypeScript"],
    frameworks=["FastAPI", "Next.js", "Django", "Express"],
    domains=["HealthTech", "IoT"],
    seniority="junior",
    companies=[],
    education="B.Tech Computer Science",
    preferred_locations=["Bengaluru", "Remote"],
    company_type_preference="product",
    max_yoe_applying_for=3,
    search_keywords=[
        "Software Engineer",
        "Backend Developer",
        "Full Stack Developer",
        "React Developer",
        "Python Developer",
        "SDE",
    ],
    raw_summary=(
        "Junior software engineer with 2 years of experience building "
        "full-stack applications using Python, React, and AWS. "
        "Comfortable with backend services, REST APIs, and cloud deployments."
    ),
)


def main():
    logger.info("=" * 60)
    logger.info("TEST PIPELINE — Mocked Agent 0A, testing 0B + 0C")
    logger.info("=" * 60)

    logger.info(f"Mock profile: {MOCK_PROFILE.full_name} | {MOCK_PROFILE.seniority}")
    logger.info(f"Search keywords: {MOCK_PROFILE.search_keywords}")

    # ── Agent 0B: Scrape ──────────────────────────────────────────────
    logger.info("\n[AGENT 0B] Scraping jobs...")
    raw_jobs = scrape_jobs(MOCK_PROFILE, target_raw=150)

    if not raw_jobs:
        logger.warning("No jobs found. Check SerpAPI key and search queries.")
        sys.exit(0)

    logger.info(f"\nScraper returned {len(raw_jobs)} jobs. Sample titles:")
    for job in raw_jobs[:10]:
        logger.info(f"  • {job.title} @ {job.company} — {job.location}")

    # ── Agent 0C: Rank ────────────────────────────────────────────────
    logger.info("\n[AGENT 0C] Ranking and filtering...")
    ranked = rank_and_filter_jobs(raw_jobs, MOCK_PROFILE, target_output=55, min_score=45)

    if not ranked:
        logger.warning("No jobs passed ranking. Check hard filters / min_score.")
        sys.exit(0)

    logger.info(f"\n{len(ranked)} jobs passed ranking:")
    for job, score in ranked[:15]:
        logger.info(f"  [{score:3d}] {job.title} @ {job.company}")

    logger.info("\nTest complete.")


if __name__ == "__main__":
    main()
