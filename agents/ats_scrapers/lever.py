from datetime import datetime, timezone
from agents.job_listing import JobListing, _make_job_id
from .common import get_json, html_to_text, company_name_from_token


def fetch_jobs(token: str) -> list[JobListing]:
    """Fetch all open jobs for a Lever-hosted company postings page."""
    data = get_json(
        f"https://api.lever.co/v0/postings/{token}",
        params={"mode": "json"},
    )
    if not isinstance(data, list):
        return []

    company = company_name_from_token(token)

    listings = []
    for job in data:
        title = (job.get("text") or "").strip()
        if not title:
            continue

        categories = job.get("categories") or {}
        all_locations = categories.get("allLocations") or []
        location = ", ".join(all_locations) if all_locations else (categories.get("location") or "")

        description = job.get("descriptionPlain") or html_to_text(job.get("description") or "")

        date_posted = None
        created_at = job.get("createdAt")
        if created_at:
            date_posted = datetime.fromtimestamp(created_at / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

        listings.append(JobListing(
            job_id=_make_job_id(company, title),
            title=title,
            company=company,
            location=location,
            description=description,
            date_posted=date_posted,
            job_url=job.get("hostedUrl"),
            source="lever",
        ))
    return listings
