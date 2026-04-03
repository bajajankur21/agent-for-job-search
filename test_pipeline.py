"""
Configurable test runner for the job application pipeline.

Each agent runs in one of three modes:
  mock   — use fixture data (zero API cost)
  live   — run the real agent as-is
  gemini — swap Claude Sonnet → Gemini Flash (free tier, same prompts)

Usage:
  # Default: mock profiler + publisher, live everything else (costs only SerpAPI + Gemini)
  python test_pipeline.py

  # All Gemini, no Claude/AWS at all:
  python test_pipeline.py --gemini-only

  # Mock specific agents:
  python test_pipeline.py --mock agent_0a,agent_0b,agent_3

  # Override a single agent to live:
  python test_pipeline.py --gemini-only --live agent_0b
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from dotenv import load_dotenv

from agents.agent_0a_profiler import CandidateProfile, build_candidate_profile
from agents.agent_0b_scraper import JobListing, scrape_jobs
from agents.agent_0c_ranker import rank_and_filter_jobs
from agents.agent_1 import run_gatekeeper
from agents.agent_2 import run_tailor
from agents.agent_3 import publish_to_s3

from tests.fixtures import (
    mock_agent_0a, mock_agent_0b, mock_agent_0c,
    mock_agent_1, mock_agent_2, mock_agent_3,
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

ALL_AGENTS = ["agent_0a", "agent_0b", "agent_0c", "agent_1", "agent_2", "agent_3"]

# ── Defaults: mock the expensive / external-infra agents, live the rest ───
DEFAULT_MODES = {
    "agent_0a": "mock",     # Claude Sonnet — expensive
    "agent_0b": "live",     # SerpAPI — free tier
    "agent_0c": "live",     # Gemini Flash — free tier
    "agent_1":  "live",     # Gemini Flash — free tier
    "agent_2":  "gemini",   # Claude Sonnet → Gemini override
    "agent_3":  "mock",     # AWS S3 — needs credentials
}


def parse_args():
    parser = argparse.ArgumentParser(description="Test the job pipeline with configurable agent modes.")
    parser.add_argument(
        "--max-jobs", type=int, default=5,
        help="Max number of jobs to process per-job agents (default 5). Use 0 for unlimited.",
    )
    parser.add_argument(
        "--mock", type=str, default=None,
        help="Comma-separated agents to mock, e.g. --mock agent_0a,agent_3",
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
    """Resolve the final mode for each agent from CLI flags."""
    if args.gemini_only:
        modes = {
            "agent_0a": "gemini",
            "agent_0b": "live",
            "agent_0c": "live",
            "agent_1":  "live",
            "agent_2":  "gemini",
            "agent_3":  "mock",
        }
    else:
        modes = dict(DEFAULT_MODES)

    # --mock overrides
    if args.mock:
        for name in args.mock.split(","):
            name = name.strip()
            if name in ALL_AGENTS:
                modes[name] = "mock"

    # --live overrides (applied last, highest priority)
    if args.live:
        for name in args.live.split(","):
            name = name.strip()
            if name in ALL_AGENTS:
                modes[name] = "live"

    return modes


def extract_resume_text(pdf_path: Path) -> str:
    import PyPDF2
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
        logger.info("[AGENT 0A] Running profiler via Gemini Flash")
        profile = build_candidate_profile_gemini(str(RESUME_PDF_PATH))
    else:
        logger.info("[AGENT 0A] Running profiler via Claude Sonnet (live)")
        profile = build_candidate_profile(str(RESUME_PDF_PATH))

    logger.info(f"Profile: {profile.full_name} | {profile.seniority} | keywords={profile.search_keywords}")

    # Extract resume text (needed for Agent 2 if not mocked)
    master_resume_text = ""
    if modes["agent_2"] != "mock" and RESUME_PDF_PATH.exists():
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
        logger.info("[AGENT 0C] Using mock ranking (all jobs pass with score 75)")
        ranked_jobs = mock_agent_0c(raw_jobs)
    else:
        logger.info("[AGENT 0C] Ranking via Gemini Flash (live)")
        ranked_jobs = rank_and_filter_jobs(raw_jobs, profile, target_output=55, min_score=45)

    if not ranked_jobs:
        logger.warning("No jobs passed ranking. Check hard filters / min_score.")
        sys.exit(0)

    logger.info(f"Ranker: {len(ranked_jobs)} jobs passed:")
    for job, score in ranked_jobs[:10]:
        logger.info(f"  [{score:3d}] {job.title} @ {job.company}")

    # ── Per-job: Agent 1 → 2 → 3 ─────────────────────────────────────
    max_jobs = args.max_jobs if args.max_jobs > 0 else len(ranked_jobs)
    if max_jobs < len(ranked_jobs):
        logger.info(f"Limiting to top {max_jobs} jobs (--max-jobs {max_jobs})")
        ranked_jobs = ranked_jobs[:max_jobs]

    results = []
    for job, score in ranked_jobs:
        logger.info(f"\n{'─' * 50}")
        logger.info(f"Processing: {job.title} @ {job.company} (score: {score})")

        try:
            # Agent 1: Gatekeeper
            if modes["agent_1"] == "mock":
                gate = mock_agent_1()
            else:
                gate = run_gatekeeper(job, max_yoe=profile.max_yoe_applying_for)

            if not gate.passed:
                logger.info(f"  SKIPPED (YOE: {gate.minimum_yoe} > {profile.max_yoe_applying_for})")
                results.append({"job": f"{job.title} @ {job.company}", "status": "SKIPPED"})
                continue

            # Agent 2: Tailor
            if modes["agent_2"] == "mock":
                tailored = mock_agent_2(job)
            elif modes["agent_2"] == "gemini":
                tailored = run_tailor_gemini(job, profile, master_resume_text)
            else:
                tailored = run_tailor(job, profile, master_resume_text)

            # Agent 3: Publisher
            if modes["agent_3"] == "mock":
                uploads = mock_agent_3(job)
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
    skipped = sum(1 for r in results if r["status"] == "SKIPPED")
    errors = sum(1 for r in results if r["status"] == "ERROR")
    logger.info(f"DONE — Published: {published} | Skipped: {skipped} | Errors: {errors}")


if __name__ == "__main__":
    main()
