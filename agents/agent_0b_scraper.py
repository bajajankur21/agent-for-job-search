import os
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

from agents.agent_0a_profiler import CandidateProfile
from agents.job_listing import JobListing, _make_job_id, _normalize_for_id  # noqa: F401 — re-exported
from agents.ats_scrapers import FETCHERS
from agents.ats_scrapers.common import is_india_or_remote

load_dotenv()
logger = logging.getLogger(__name__)

_COMPANIES_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ats_companies.json"
)


def _load_companies() -> dict[str, list[str]]:
    with open(_COMPANIES_FILE) as f:
        return json.load(f)


def scrape_jobs(profile: CandidateProfile, target_raw: int = 80) -> list[JobListing]:
    """
    Crawls every company token in data/ats_companies.json across Greenhouse,
    Lever, Ashby, and Workable, returning their full job boards as
    JobListings (each board's job_url is a direct application-form link).

    None of these ATS platforms expose a cross-company "search by title" API
    — each only returns one company's entire board. So the candidate pool
    comes from breadth of company coverage (see scripts/discover_ats_companies.py
    for how data/ats_companies.json is built/grown), and Agent 0C's hard
    filter (location/title/skill/YOE) does the title-relevance filtering
    that used to happen via JobSpy search terms.

    Env knobs:
    - ATS_FETCH_CONCURRENCY: thread pool size for per-company API calls, default 12
    """
    companies = _load_companies()
    concurrency = int(os.getenv("ATS_FETCH_CONCURRENCY", "12"))

    worklist = [
        (ats_name, token)
        for ats_name, tokens in companies.items()
        for token in tokens
    ]

    preferred_terms = [loc.lower() for loc in profile.preferred_locations]

    seen_ids: set[str] = set()
    all_jobs: list[JobListing] = []
    per_ats_counts: dict[str, int] = {}

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(FETCHERS[ats_name], token): (ats_name, token)
            for ats_name, token in worklist
        }
        for future in as_completed(futures):
            ats_name, token = futures[future]
            try:
                jobs = future.result()
            except Exception as e:
                logger.warning(f"  [{ats_name}] '{token}' failed: {e}")
                continue

            for job in jobs:
                if job.job_id in seen_ids:
                    continue
                loc_lower = job.location.lower()
                if not is_india_or_remote(loc_lower) and not any(t in loc_lower for t in preferred_terms):
                    continue
                seen_ids.add(job.job_id)
                all_jobs.append(job)
                per_ats_counts[ats_name] = per_ats_counts.get(ats_name, 0) + 1

    logger.info(
        f"Scraper complete: {len(all_jobs)} unique India/Remote jobs from "
        f"{len(worklist)} companies across {list(companies.keys())} (target was {target_raw}) "
        f"— breakdown: {per_ats_counts}"
    )

    return all_jobs[:target_raw]
