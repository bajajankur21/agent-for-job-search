import re
import html
import logging
import requests

logger = logging.getLogger(__name__)

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "job-search-agent/1.0 (+ats-scraper)"})

DEFAULT_TIMEOUT = 15

# Same location terms Agent 0C's hard filter checks for, used by the
# discovery script to decide whether a company is relevant to keep.
# "india" needs a word boundary — otherwise "Indianapolis, IN" matches.
INDIA_OR_REMOTE_TERMS = (
    "bengaluru", "bangalore", "remote", "work from home", "hybrid",
)
_INDIA_RE = re.compile(r"\bindia\b")


def get_json(url: str, params: dict | None = None):
    """GET a JSON endpoint. Returns the parsed body, or None on any failure
    (404/network error/non-JSON) so a dead or renamed company token never
    breaks the run."""
    try:
        resp = _SESSION.get(url, params=params, timeout=DEFAULT_TIMEOUT)
        if resp.status_code != 200:
            logger.debug(f"  GET {url} -> HTTP {resp.status_code}")
            return None
        return resp.json()
    except (requests.RequestException, ValueError) as e:
        logger.debug(f"  GET {url} failed: {e}")
        return None


_TAG_RE = re.compile(r"<[^>]+>")


def html_to_text(content: str) -> str:
    """Strip HTML tags and unescape entities, collapsing whitespace."""
    if not content:
        return ""
    text = html.unescape(_TAG_RE.sub(" ", content))
    return " ".join(text.split())


def company_name_from_token(token: str) -> str:
    """Best-effort human-readable company name when the ATS API doesn't
    return one (Lever, Ashby don't include a company name per posting)."""
    words = re.split(r"[-_]+", token)
    return " ".join(w.capitalize() for w in words if w)


def is_india_or_remote(location: str) -> bool:
    loc_lower = (location or "").lower()
    return _INDIA_RE.search(loc_lower) is not None or any(
        term in loc_lower for term in INDIA_OR_REMOTE_TERMS
    )
