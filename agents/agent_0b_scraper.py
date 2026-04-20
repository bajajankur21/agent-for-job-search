import os
import re
import logging
import hashlib
from typing import Optional
import pandas as pd
from pydantic import BaseModel
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
    source: str = "jobspy"


def _normalize_for_id(s: str) -> str:
    # Strip pipe-separated qualifiers ("| 6 YoE | Remote | Immediate Joiner"),
    # then collapse any non-alphanumeric run to a single space. Catches reposts
    # that differ only in punctuation/whitespace/trailing tags.
    s = s.lower().strip().split(" | ")[0]
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def _make_job_id(company: str, title: str) -> str:
    raw = f"{_normalize_for_id(company)}-{_normalize_for_id(title)}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _build_search_terms(profile: CandidateProfile) -> list[str]:
    """
    JobSpy expects ONE search_term per scrape call (not Google-style multi-query).
    Pick the most-yielding seniority-appropriate role titles from the profile.
    """
    keywords = profile.search_keywords or ["Software Engineer", "Backend Engineer"]
    max_terms = int(os.getenv("JOBSPY_MAX_TERMS", "4"))

    seen = set()
    terms = []
    for kw in keywords:
        key = kw.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        terms.append(kw)
        if len(terms) >= max_terms:
            break

    logger.info(f"JobSpy will run {len(terms)} search terms:")
    for i, t in enumerate(terms, 1):
        logger.info(f"  {i}. {t}")
    return terms


def _scrape_one_term(
    term: str,
    location: str,
    results_per_site: int,
    hours_old: int,
    sites: list[str],
) -> pd.DataFrame:
    """
    Calls JobSpy for one search term. Runs sites separately because
    LinkedIn only honors `location` while Indeed needs `country_indeed`.
    """
    from jobspy import scrape_jobs as jobspy_scrape

    frames: list[pd.DataFrame] = []

    if "linkedin" in sites:
        try:
            df = jobspy_scrape(
                site_name=["linkedin"],
                search_term=term,
                location=location,
                results_wanted=results_per_site,
                hours_old=hours_old,
                linkedin_fetch_description=True,
            )
            logger.info(f"  [linkedin] '{term}' → {len(df)} results")
            frames.append(df)
        except Exception as e:
            logger.error(f"  [linkedin] '{term}' failed: {e}")

    if "indeed" in sites:
        try:
            df = jobspy_scrape(
                site_name=["indeed"],
                search_term=term,
                location=location,
                country_indeed="india",
                results_wanted=results_per_site,
                hours_old=hours_old,
            )
            logger.info(f"  [indeed] '{term}' → {len(df)} results")
            frames.append(df)
        except Exception as e:
            logger.error(f"  [indeed] '{term}' failed: {e}")

    if "glassdoor" in sites:
        try:
            df = jobspy_scrape(
                site_name=["glassdoor"],
                search_term=term,
                location=location,
                country_indeed="india",
                results_wanted=results_per_site,
                hours_old=hours_old,
            )
            logger.info(f"  [glassdoor] '{term}' → {len(df)} results")
            frames.append(df)
        except Exception as e:
            logger.error(f"  [glassdoor] '{term}' failed: {e}")

    if "naukri" in sites:
        try:
            df = jobspy_scrape(
                site_name=["naukri"],
                search_term=term,
                location=location,
                results_wanted=results_per_site,
                hours_old=hours_old,
            )
            logger.info(f"  [naukri] '{term}' → {len(df)} results")
            frames.append(df)
        except Exception as e:
            logger.error(f"  [naukri] '{term}' failed: {e}")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _row_to_listing(row: pd.Series) -> Optional[JobListing]:
    """Map a JobSpy DataFrame row to a JobListing. Returns None on missing essentials."""
    company = str(row.get("company") or "").strip() or "Unknown Company"
    title = str(row.get("title") or "").strip()
    if not title:
        return None

    location_parts = [
        str(row.get("city") or "").strip(),
        str(row.get("state") or "").strip(),
        str(row.get("country") or "").strip(),
    ]
    location = ", ".join(p for p in location_parts if p and p.lower() != "nan")
    if not location:
        location = str(row.get("location") or "").strip()
    if row.get("is_remote") is True:
        location = (location + " (Remote)").strip() if location else "Remote"

    description = str(row.get("description") or "").strip()
    if description.lower() == "nan":
        description = ""

    date_posted = row.get("date_posted")
    if pd.isna(date_posted):
        date_posted = None
    else:
        date_posted = str(date_posted)

    job_url = row.get("job_url")
    if pd.isna(job_url):
        job_url = row.get("job_url_direct")
    if pd.isna(job_url):
        job_url = None
    else:
        job_url = str(job_url)

    site = str(row.get("site") or "jobspy").strip().lower() or "jobspy"

    return JobListing(
        job_id=_make_job_id(company, title),
        title=title,
        company=company,
        location=location,
        description=description,
        date_posted=date_posted,
        job_url=job_url,
        source=site,
    )


def scrape_jobs(profile: CandidateProfile, target_raw: int = 80) -> list[JobListing]:
    """
    Scrapes LinkedIn + Indeed (and optionally Glassdoor) via JobSpy.
    No API key required. Returns up to target_raw deduplicated raw jobs.

    Env knobs:
    - JOBSPY_SITES: comma-separated, default "linkedin,indeed"
    - JOBSPY_RESULTS_PER_SITE: per-site cap per term, default 25
    - JOBSPY_HOURS_OLD: recency filter in hours, default 168 (7 days)
    - JOBSPY_MAX_TERMS: how many search_keywords to use, default 4
    """
    sites = [
        s.strip().lower()
        for s in os.getenv("JOBSPY_SITES", "linkedin,indeed").split(",")
        if s.strip()
    ]
    results_per_site = int(os.getenv("JOBSPY_RESULTS_PER_SITE", "25"))
    hours_old = int(os.getenv("JOBSPY_HOURS_OLD", "168"))

    primary_location = (profile.preferred_locations or ["Bengaluru"])[0]
    terms = _build_search_terms(profile)

    location_terms = [loc.lower() for loc in profile.preferred_locations] + [
        "remote", "work from home", "hybrid", "india"
    ]

    seen_ids: set[str] = set()
    all_jobs: list[JobListing] = []

    for term in terms:
        if len(all_jobs) >= target_raw:
            logger.info(f"Reached target of {target_raw} raw jobs. Stopping scrape.")
            break

        df = _scrape_one_term(
            term=term,
            location=primary_location,
            results_per_site=results_per_site,
            hours_old=hours_old,
            sites=sites,
        )
        if df.empty:
            continue

        for _, row in df.iterrows():
            if len(all_jobs) >= target_raw:
                break

            listing = _row_to_listing(row)
            if listing is None:
                continue
            if listing.job_id in seen_ids:
                continue
            seen_ids.add(listing.job_id)

            if not any(t in listing.location.lower() for t in location_terms):
                logger.debug(f"Pre-filter drop (location): '{listing.title}' @ '{listing.company}' — '{listing.location}'")
                continue

            all_jobs.append(listing)

    logger.info(
        f"Scraper complete: {len(all_jobs)} unique jobs collected "
        f"from {len(terms)} search terms across {sites} (target was {target_raw})"
    )
    return all_jobs
