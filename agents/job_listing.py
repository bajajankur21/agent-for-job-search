import re
import hashlib
from typing import Optional
from pydantic import BaseModel


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
