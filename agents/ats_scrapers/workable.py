from agents.job_listing import JobListing, _make_job_id
from .common import get_json, html_to_text, company_name_from_token


def fetch_jobs(token: str) -> list[JobListing]:
    """Fetch all open jobs for a Workable-hosted company job board."""
    data = get_json(
        f"https://apply.workable.com/api/v1/widget/accounts/{token}",
        params={"details": "true"},
    )
    if not data:
        return []

    company = data.get("name") or company_name_from_token(token)

    listings = []
    for job in data.get("jobs", []):
        title = (job.get("title") or "").strip()
        if not title:
            continue

        loc = job.get("location") or {}
        location_parts = [loc.get(key) for key in ("city", "region", "country")]
        location = ", ".join(p for p in location_parts if p)
        if loc.get("telecommute") or loc.get("remote"):
            location = (location + " (Remote)").strip() if location else "Remote"

        description = html_to_text(job.get("description") or "")

        listings.append(JobListing(
            job_id=_make_job_id(company, title),
            title=title,
            company=company,
            location=location,
            description=description,
            date_posted=job.get("published_on") or job.get("created_at"),
            job_url=job.get("url") or job.get("shortlink"),
            source="workable",
        ))
    return listings
