import os
import logging
import hashlib
from typing import Optional
from pydantic import BaseModel, Field
from serpapi import GoogleSearch
from dotenv import load_dotenv
from agents.agent_0a_profiler import CandidateProfile

load_dotenv()
logger = logging.getLogger(__name__)


class JobListing(BaseModel):
    job_id: str
    title: str
    company: str
    location: str
    description: str
    date_posted: Optional[str] = None
    job_url: Optional[str] = None
    source: str = "google_jobs"


def _make_job_id(company: str, title: str) -> str:
    raw = f"{company.lower().strip()}-{title.lower().strip()}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _build_search_queries(profile: CandidateProfile) -> list[str]:
    """
    Generates 8 SDE-specific queries from the candidate profile.
    
    Design principle: every query contains at least one hard technical
    keyword so Google Jobs never returns marketing/finance/sales roles.
    Queries vary across skill combinations and seniority signals to
    maximise coverage without duplicating results.
    """
    # Pull the top skills from profile — these are the hard technical anchors
    skills = profile.search_keywords or ["React", "TypeScript", "frontend"]
    top3 = skills[:3]
    mid3 = skills[3:6] if len(skills) >= 6 else skills[:3]

    # Seniority label Google Jobs understands
    seniority_label = {
        "junior": "junior software engineer",
        "mid": "software engineer II",
        "senior": "senior software engineer",
        "staff": "staff engineer"
    }.get(profile.seniority, "software engineer")

    queries = [
        # Core role + primary skills + location
        f"{' '.join(top3)} {seniority_label} Bengaluru product company",
        f"{' '.join(top3)} {seniority_label} remote India product",

        # Alt skills variation
        f"{' '.join(mid3)} software engineer Bengaluru",
        f"{' '.join(mid3)} developer remote India",

        # Domain-specific (HealthTech, Design Systems, etc.)
        *[
            f"{domain} {seniority_label} Bengaluru"
            for domain in (profile.domains or [])[:2]
        ],

        # Title-driven queries — catches jobs that list the exact role
        f"{profile.current_title} Bengaluru startup",
        f"{profile.current_title} remote India product based",

        # Broad fallback with hard SDE anchor to avoid non-tech roles
        f"software engineer {top3[0]} Bengaluru OR remote India",
    ]

    # Deduplicate and cap at 8 queries
    seen = set()
    final = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            final.append(q)
        if len(final) == 8:
            break

    logger.info(f"Generated {len(final)} search queries:")
    for i, q in enumerate(final, 1):
        logger.info(f"  {i}. {q}")

    return final


def _search_single_query(query: str, serp_api_key: str) -> list[dict]:
    """
    Calls SerpAPI Google Jobs for one query.
    num=20 is the max results per call on the free tier.
    chips=date_posted:today ensures last 24h only.
    """
    params = {
        "engine": "google_jobs",
        "q": query,
        "location": "India",
        "hl": "en",
        "gl": "in",
        "chips": "date_posted:today",
        "num": "20",
        "api_key": serp_api_key,
    }

    try:
        search = GoogleSearch(params)
        results = search.get_dict()

        # SerpAPI returns an error key if the query fails
        if "error" in results:
            logger.error(f"SerpAPI error for query '{query}': {results['error']}")
            return []

        jobs = results.get("jobs_results", [])
        logger.info(f"  Query '{query[:55]}...' → {len(jobs)} raw results")
        return jobs

    except Exception as e:
        logger.error(f"SerpAPI call failed for query '{query}': {e}")
        return []


def scrape_jobs(profile: CandidateProfile, target_raw: int = 150) -> list[JobListing]:
    """
    Scrapes Google Jobs across multiple queries.
    Returns up to target_raw deduplicated raw jobs BEFORE any ranking.
    Location filter is intentionally loose here — ranker handles precision.
    
    Args:
        profile: Candidate profile built by Agent 0A
        target_raw: How many raw jobs to collect before stopping (default 150)
    """
    serp_api_key = os.getenv("SERP_API_KEY")
    if not serp_api_key:
        raise EnvironmentError(
            "SERP_API_KEY not set. "
            "Get a free key at serpapi.com and add it to GitHub Secrets."
        )

    queries = _build_search_queries(profile)
    seen_ids: set[str] = set()
    all_jobs: list[JobListing] = []

    # Preferred locations — used for soft pre-filter only
    location_terms = [loc.lower() for loc in profile.preferred_locations] + \
                     ["remote", "work from home", "hybrid", "india"]

    for query in queries:
        if len(all_jobs) >= target_raw:
            logger.info(f"Reached target of {target_raw} raw jobs. Stopping scrape.")
            break

        raw_results = _search_single_query(query, serp_api_key)

        for raw in raw_results:
            if len(all_jobs) >= target_raw:
                break

            company = raw.get("company_name", "Unknown Company")
            title = raw.get("title", "Unknown Title")
            job_id = _make_job_id(company, title)

            # Deduplicate
            if job_id in seen_ids:
                continue
            seen_ids.add(job_id)

            # Soft location pre-filter — keep anything India-adjacent
            # Hard location enforcement happens in the ranker
            location = raw.get("location", "")
            location_ok = any(term in location.lower() for term in location_terms)
            if not location_ok:
                logger.debug(f"Pre-filter drop (location): '{title}' @ '{company}' — '{location}'")
                continue

            # Build description from all available fields
            description = raw.get("description", "")
            if not description:
                extensions = raw.get("detected_extensions", {})
                description = " | ".join(f"{k}: {v}" for k, v in extensions.items())

            # Get job URL from related links if available
            related_links = raw.get("related_links", [])
            job_url = related_links[0].get("link") if related_links else None

            all_jobs.append(JobListing(
                job_id=job_id,
                title=title,
                company=company,
                location=location,
                description=description,
                date_posted=raw.get("detected_extensions", {}).get("posted_at"),
                job_url=job_url,
            ))

    logger.info(
        f"Scraper complete: {len(all_jobs)} unique jobs collected "
        f"from {len(queries)} queries (target was {target_raw})"
    )
    return all_jobs 