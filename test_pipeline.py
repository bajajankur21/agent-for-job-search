"""
Configurable test runner for the job application pipeline.

Each agent runs in one of three modes:
  mock   — use fixture data (zero API cost)
  live   — run the real agent as-is
  gemini — swap Claude Sonnet → Gemini Flash (free tier, same prompts)

Usage:
  # Default: mock profiler + publisher, live scraper + ranker, gemini tailor
  python test_pipeline.py

  # All Gemini, no Claude/AWS at all:
  python test_pipeline.py --gemini-only

  # Mock specific agents:
  python test_pipeline.py --mock agent_0a,agent_0b,agent_2

  # Limit jobs processed (default 5):
  python test_pipeline.py --max-jobs 3
"""

import sys
import logging
import argparse
from pathlib import Path
from dotenv import load_dotenv

import PyPDF2

from agents.agent_0a_profiler import build_candidate_profile
from agents.agent_0b_scraper import scrape_jobs
from agents.agent_0c_ranker import rank_and_filter_jobs
from agents.agent_1 import run_tailor
from agents.agent_2 import publish_to_s3

from tests.fixtures import (
    mock_agent_0a, mock_agent_0b, mock_agent_0c,
    mock_agent_1, mock_agent_2,
)
from tests.gemini_overrides import (
    build_candidate_profile_gemini,
    run_tailor_gemini,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("test_pipeline")

RESUME_PDF_PATH = Path("data/master_resume.pdf")

ALL_AGENTS = ["agent_0a", "agent_0b", "agent_0c", "agent_1", "agent_2"]

# Defaults: mock expensive/infra agents, live the free ones, gemini for Claude
DEFAULT_MODES = {
    "agent_0a": "mock",    # Claude Sonnet — expensive
    "agent_0b": "live",    # SerpAPI — free tier
    "agent_0c": "live",    # Gemini Flash Lite — free tier
    "agent_1":  "gemini",  # Claude Sonnet → Gemini override
    "agent_2":  "mock",    # AWS S3 — needs credentials
}


def parse_args():
    parser = argparse.ArgumentParser(description="Test the job pipeline with configurable agent modes.")
    parser.add_argument(
        "--max-jobs", type=int, default=5,
        help="Max jobs to process through agent_1 + agent_2 (default 5). 0 = unlimited.",
    )
    parser.add_argument(
        "--mock", type=str, default=None,
        help="Comma-separated agents to mock, e.g. --mock agent_0a,agent_2",
    )
    parser.add_argument(
        "--live", type=str, default=None,
        help="Comma-separated agents to force live, e.g. --live agent_0b",
    )
    parser.add_argument(
        "--gemini-only", action="store_true",
        help="Swap all Claude agents to Gemini, mock S3. No Anthropic/AWS costs.",
    )
    return parser.parse_args()


def build_mode_config(args) -> dict[str, str]:
    if args.gemini_only:
        modes = {
            "agent_0a": "gemini",
            "agent_0b": "live",
            "agent_0c": "live",
            "agent_1":  "gemini",
            "agent_2":  "mock",
        }
    else:
        modes = dict(DEFAULT_MODES)

    if args.mock:
        for name in args.mock.split(","):
            name = name.strip()
            if name in ALL_AGENTS:
                modes[name] = "mock"

    if args.live:
        for name in args.live.split(","):
            name = name.strip()
            if name in ALL_AGENTS:
                modes[name] = "live"

    return modes


def extract_resume_text(pdf_path: Path) -> str:
    text_parts = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
    return "\n".join(text_parts)


def main():
    args = parse_args()
    modes = build_mode_config(args)

    logger.info("=" * 60)
    logger.info("TEST PIPELINE — Configurable agent modes")
    logger.info("=" * 60)
    for agent, mode in modes.items():
        tag = {"mock": "MOCK", "live": "LIVE", "gemini": "GEMINI"}[mode]
        logger.info(f"  {agent}: [{tag}]")
    logger.info("")

    # ── Agent 0A: Profiler ────────────────────────────────────────────
    if modes["agent_0a"] == "mock":
        logger.info("[AGENT 0A] Using mock profile")
        profile = mock_agent_0a()
    elif modes["agent_0a"] == "gemini":
        logger.info("[AGENT 0A] Running profiler via Gemini (override)")
        profile = build_candidate_profile_gemini(str(RESUME_PDF_PATH))
    else:
        logger.info("[AGENT 0A] Running profiler via Claude (live)")
        profile = build_candidate_profile(str(RESUME_PDF_PATH))

    logger.info(f"Profile: {profile.full_name} | {profile.seniority} | keywords={profile.search_keywords}")

    master_resume_text = ""
    if modes["agent_1"] != "mock" and RESUME_PDF_PATH.exists():
        master_resume_text = extract_resume_text(RESUME_PDF_PATH)

    # ── Agent 0B: Scraper ─────────────────────────────────────────────
    if modes["agent_0b"] == "mock":
        logger.info("[AGENT 0B] Using mock jobs")
        raw_jobs = mock_agent_0b()
    else:
        logger.info("[AGENT 0B] Scraping jobs via SerpAPI (live)")
        raw_jobs = scrape_jobs(profile, target_raw=150)

    if not raw_jobs:
        logger.warning("No jobs found. Check SerpAPI key and search queries.")
        sys.exit(0)

    logger.info(f"Scraper: {len(raw_jobs)} jobs. Sample:")
    for job in raw_jobs[:5]:
        logger.info(f"  - {job.title} @ {job.company} — {job.location}")

    # ── Agent 0C: Ranker ──────────────────────────────────────────────
    if modes["agent_0c"] == "mock":
        logger.info("[AGENT 0C] Using mock ranking (all jobs score 75)")
        ranked_jobs = mock_agent_0c(raw_jobs)
    else:
        logger.info("[AGENT 0C] Ranking via Gemini Flash Lite (live)")
        ranked_jobs = rank_and_filter_jobs(raw_jobs, profile, target_output=55, min_score=45)

    if not ranked_jobs:
        logger.warning("No jobs passed ranking. Check hard filters / min_score.")
        sys.exit(0)

    logger.info(f"Ranker: {len(ranked_jobs)} jobs passed:")
    for job, score in ranked_jobs[:10]:
        logger.info(f"  [{score:3d}] {job.title} @ {job.company}")

    # ── Per-job: Agent 1 → Agent 2 ───────────────────────────────────
    max_jobs = args.max_jobs if args.max_jobs > 0 else len(ranked_jobs)
    if max_jobs < len(ranked_jobs):
        logger.info(f"Limiting to top {max_jobs} jobs (--max-jobs {max_jobs})")
        ranked_jobs = ranked_jobs[:max_jobs]

    results = []
    for job, score in ranked_jobs:
        logger.info(f"\n{'─' * 50}")
        logger.info(f"Processing: {job.title} @ {job.company} (score: {score})")

        try:
            # Agent 1: Tailor
            if modes["agent_1"] == "mock":
                tailored = mock_agent_1(job)
            elif modes["agent_1"] == "gemini":
                tailored = run_tailor_gemini(job, profile, master_resume_text)
            else:
                tailored = run_tailor(job, profile, master_resume_text)

            # Agent 2: Publish
            if modes["agent_2"] == "mock":
                uploads = mock_agent_2(job)
            else:
                uploads = publish_to_s3(job, tailored, profile.full_name)

            logger.info(f"  PUBLISHED: {uploads}")
            results.append({"job": f"{job.title} @ {job.company}", "status": "PUBLISHED"})

        except Exception as e:
            logger.error(f"  ERROR: {e}")
            results.append({"job": f"{job.title} @ {job.company}", "status": "ERROR", "reason": str(e)})
            continue

    # ── Summary ───────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    published = sum(1 for r in results if r["status"] == "PUBLISHED")
    errors = sum(1 for r in results if r["status"] == "ERROR")
    logger.info(f"DONE — Published: {published} | Errors: {errors}")


if __name__ == "__main__":
    main()
