from agents.job_listing import JobListing, _make_job_id
from .common import get_json, html_to_text, company_name_from_token


def fetch_jobs(token: str) -> list[JobListing]:
    """Fetch all open jobs for an Ashby-hosted company job board."""
    data = get_json(f"https://api.ashbyhq.com/posting-api/job-board/{token}")
    if not data:
        return []

    company = company_name_from_token(token)

    listings = []
    for job in data.get("jobs", []):
        title = (job.get("title") or "").strip()
        if not title:
            continue

        locations = [job.get("location") or ""]
        for secondary in job.get("secondaryLocations") or []:
            loc = secondary.get("location")
            if loc:
                locations.append(loc)
        location = ", ".join(loc for loc in locations if loc)

        description = job.get("descriptionPlain") or html_to_text(job.get("descriptionHtml") or "")

        listings.append(JobListing(
            job_id=_make_job_id(company, title),
            title=title,
            company=company,
            location=location,
            description=description,
            date_posted=(job.get("publishedAt") or "")[:10] or None,
            job_url=job.get("jobUrl") or job.get("applyUrl"),
            source="ashby",
        ))
    return listings
