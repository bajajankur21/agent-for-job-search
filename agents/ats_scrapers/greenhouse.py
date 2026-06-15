from agents.job_listing import JobListing, _make_job_id
from .common import get_json, html_to_text, company_name_from_token


def fetch_jobs(token: str) -> list[JobListing]:
    """Fetch all open jobs for a Greenhouse-hosted company board."""
    data = get_json(
        f"https://boards-api.greenhouse.io/v1/boards/{token}/jobs",
        params={"content": "true"},
    )
    if not data:
        return []

    listings = []
    for job in data.get("jobs", []):
        title = (job.get("title") or "").strip()
        if not title:
            continue

        company = job.get("company_name") or company_name_from_token(token)
        location = (job.get("location") or {}).get("name") or ""

        listings.append(JobListing(
            job_id=_make_job_id(company, title),
            title=title,
            company=company,
            location=location,
            description=html_to_text(job.get("content") or ""),
            date_posted=job.get("updated_at"),
            job_url=job.get("absolute_url"),
            source="greenhouse",
        ))
    return listings
