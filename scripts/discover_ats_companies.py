"""
Maintenance script — builds/grows data/ats_companies.json, the list of
ATS company tokens that agent_0b_scraper.scrape_jobs() crawls every run.

Greenhouse/Lever/Ashby don't expose a cross-company "search by title" API,
so coverage comes from crawling many companies and keeping the ones that
currently have at least one India/Remote-relevant, SDE-looking posting.

Usage:
    python scripts/discover_ats_companies.py --ats greenhouse --start 0 --limit 1000
    python scripts/discover_ats_companies.py --ats lever
    python scripts/discover_ats_companies.py --ats ashby
    python scripts/discover_ats_companies.py --ats workable

    # Resumable "auto" mode (used by the scheduled discovery workflow):
    # processes the next --chunk-size companies after wherever the last
    # --auto run for this ATS left off, tracked in data/discovery_progress.json,
    # and wraps back to the start of the seed list once it reaches the end.
    python scripts/discover_ats_companies.py --ats greenhouse --auto --chunk-size 200

Resumable: pass --start/--limit to process the big token lists in slices
across multiple runs. Survivors are merged into data/ats_companies.json
(existing entries are kept, duplicates de-duped).
"""
import os
import sys
import json
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.ats_scrapers import FETCHERS
from agents.ats_scrapers.common import get_json, is_india_or_remote
from agents.agent_0c_ranker import NON_SDE_TITLE_KEYWORDS

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPANIES_FILE = os.path.join(ROOT, "data", "ats_companies.json")
PROGRESS_FILE = os.path.join(ROOT, "data", "discovery_progress.json")

SEED_LIST_URLS = {
    "greenhouse": "https://raw.githubusercontent.com/Feashliaa/job-board-aggregator/main/data/greenhouse_companies.json",
    "lever": "https://raw.githubusercontent.com/Feashliaa/job-board-aggregator/main/data/lever_companies.json",
    "ashby": "https://raw.githubusercontent.com/Feashliaa/job-board-aggregator/main/data/ashby_companies.json",
}

# No public token list exists for Workable — small hand-seeded starter set
# of accounts known to be active. Extend as more are discovered.
WORKABLE_SEED = [
    "razorpay", "grammarly", "browserstack", "darwinbox", "whatfix",
    "springworks", "chargebee", "zluri", "miro", "figma", "canva",
    "deliveroo", "gitlab", "webflow", "loom", "postman", "zeta-suite",
    "innovaccer", "meesho", "curefit", "licious", "slice",
]


def _full_token_list(ats: str) -> list[str]:
    if ats == "workable":
        return WORKABLE_SEED
    return get_json(SEED_LIST_URLS[ats]) or []


def _candidate_tokens(ats: str, start: int, limit: int | None) -> list[str]:
    tokens = _full_token_list(ats)
    if limit is not None:
        return tokens[start:start + limit]
    return tokens[start:]


def _auto_candidate_tokens(ats: str, chunk_size: int, progress: dict[str, int]) -> tuple[list[str], int]:
    """Resumable slice: the next `chunk_size` tokens after wherever the last
    --auto run for this ATS left off, wrapping back to the start of the seed
    list once it reaches the end. Returns (candidates, next_start)."""
    tokens = _full_token_list(ats)
    total = len(tokens)
    if total == 0:
        return [], 0

    chunk = min(chunk_size, total)
    start = progress.get(ats, 0) % total
    end = start + chunk
    if end <= total:
        candidates = tokens[start:end]
        next_start = end % total
    else:
        candidates = tokens[start:] + tokens[:end - total]
        next_start = end - total
    return candidates, next_start


def _load_progress() -> dict[str, int]:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {}


def _save_progress(progress: dict[str, int]) -> None:
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2, sort_keys=True)
        f.write("\n")


def _looks_sde_relevant(title: str) -> bool:
    title_lower = title.lower()
    return not any(kw in title_lower for kw in NON_SDE_TITLE_KEYWORDS)


def _is_relevant_company(token: str, fetch_jobs) -> bool:
    jobs = fetch_jobs(token)
    return any(
        is_india_or_remote(job.location) and _looks_sde_relevant(job.title)
        for job in jobs
    )


def _load_companies() -> dict[str, list[str]]:
    if os.path.exists(COMPANIES_FILE):
        with open(COMPANIES_FILE) as f:
            return json.load(f)
    return {"greenhouse": [], "lever": [], "ashby": [], "workable": []}


def _save_companies(companies: dict[str, list[str]]) -> None:
    with open(COMPANIES_FILE, "w") as f:
        json.dump(companies, f, indent=2, sort_keys=True)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ats", required=True, choices=["greenhouse", "lever", "ashby", "workable"])
    parser.add_argument("--start", type=int, default=0, help="Start offset into the candidate token list")
    parser.add_argument("--limit", type=int, default=None, help="Number of candidates to probe")
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--auto", action="store_true",
                         help="Resumable mode: process the next --chunk-size tokens after "
                              "wherever the last --auto run for this ATS left off "
                              "(tracked in data/discovery_progress.json), wrapping around "
                              "at the end of the seed list. Overrides --start/--limit.")
    parser.add_argument("--chunk-size", type=int, default=200,
                         help="Number of candidates to probe per --auto run (default 200)")
    args = parser.parse_args()

    fetch_jobs = FETCHERS[args.ats]
    progress = _load_progress()

    if args.auto:
        candidates, next_start = _auto_candidate_tokens(args.ats, args.chunk_size, progress)
        logger.info(f"[auto] Probing {len(candidates)} {args.ats} candidates "
                     f"(resuming from progress, chunk_size={args.chunk_size})...")
    else:
        candidates = _candidate_tokens(args.ats, args.start, args.limit)
        logger.info(f"Probing {len(candidates)} {args.ats} candidates (start={args.start}, limit={args.limit})...")

    companies = _load_companies()
    existing = set(companies.setdefault(args.ats, []))
    kept = []

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(_is_relevant_company, token, fetch_jobs): token for token in candidates}
        for i, future in enumerate(as_completed(futures), 1):
            token = futures[future]
            try:
                if future.result():
                    kept.append(token)
                    logger.info(f"  [{i}/{len(candidates)}] KEEP  {token}")
            except Exception as e:
                logger.warning(f"  [{i}/{len(candidates)}] ERROR {token}: {e}")

    new_tokens = set(kept) - existing
    companies[args.ats] = sorted(existing | new_tokens)
    _save_companies(companies)

    if args.auto:
        progress[args.ats] = next_start
        _save_progress(progress)

    logger.info(
        f"\n{args.ats}: kept {len(kept)}/{len(candidates)} candidates "
        f"({len(new_tokens)} new). Total {args.ats} companies: {len(companies[args.ats])}"
    )


if __name__ == "__main__":
    main()
