import os
import re
import json
import logging
from agents.agent_0a_profiler import CandidateProfile
from agents.agent_0b_scraper import JobListing
from dotenv import load_dotenv


load_dotenv()
logger = logging.getLogger(__name__)

# ── Hard blacklists — zero cost to apply ─────────────────────────────────────

SERVICE_COMPANY_BLACKLIST = {
    "tcs", "tata consultancy services",
    "infosys",
    "wipro",
    "hcl", "hcl technologies",
    "cognizant",
    "accenture",
    "capgemini",
    "tech mahindra",
    "mphasis",
    "hexaware",
    "ltimindtree", "lti", "ltm", "larsen & toubro infotech",
    "persistent", "persistent systems",
    "coforge",
    "niit",
    "zensar",
    "ibm",
    "dxc", "dxc technology",
    "unisys",
    "syntel",
    "mastech",
    "igate",
    "genpact",
    "wipro",
    "birlasoft",
    "mindtree",
    "happiest minds",
    "cyient",
    "kpit",
    "sonata software",
    "kellton tech",
}

# ── Personal exclusion list ───────────────────────────────────────────────────
# Exact lowercase company names to skip (current employer, companies with offers).
EXCLUDED_COMPANIES: set[str] = {
    "philips",                  # current employer
    "ge healthcare",            # offer in hand
}

# Prefix patterns (lowercase) — any company whose name starts with one of these
# is excluded. Handles "GE Healthcare", "GE Digital", "GE Vernova", etc.
EXCLUDED_COMPANY_PREFIXES: tuple[str, ...] = (
    "ge ",      # "ge digital", "ge vernova", …
    "ge-",      # hyphenated variants
)

NON_SDE_TITLE_KEYWORDS = {
    "marketing", "sales", "finance", "accounting", "hr ", "human resource",
    "recruiter", "talent", "analyst", "business development", "bd ",
    "content writer", "seo", "social media", "graphic design",
    "product manager", "program manager", "project manager",
    "scrum master", "agile coach", "customer success", "support engineer",
    "technical support", "it support", "helpdesk", "network engineer",
    "system administrator", "sysadmin", "database administrator", "dba",
    # QA / Testing roles
    "qa ", "quality assurance", "automation engineer", "test engineer",
    "sdet", "testing engineer", "manual tester", "qa analyst",
    # Senior / non-IC roles a junior candidate should never match
    "vice president", "vp ", "vp,", " vp", "director", "head of",
    "engineering manager", "em ", "chief", "cto", "cio", "cfo",
    "general manager", "gm ", "avp", "svp", "evp",
}

# ── YOE regex pre-filter ──────────────────────────────────────────────────────
# Runs before Gemini to eliminate obviously over-levelled jobs from the batch,
# reducing token cost and improving scoring accuracy.

_YOE_PATTERNS = [
    # Explicit requirement phrases — high confidence
    (r'\b(?:minimum|min\.?|at\s+least|around|approx\.?)\s+(?:of\s+)?(\d+)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:experience|exp|professional|relevant|hands[\s-]?on|working)', 1),
    (r'\b(?:require[sd]?|expecting|need[sd]?|looking\s+for)\s+(?:of\s+)?(\d+)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:experience|exp|professional|relevant|hands[\s-]?on|working)', 1),
    # "N+ years of experience"
    (r'\b(\d+)\+\s*(?:years?|yrs?)\s+(?:of\s+)?(?:experience|exp|relevant|professional|hands[\s-]?on|working)', 1),
    # "N years of experience" (no plus, but clearly a requirement)
    (r'\b(\d+)\s+(?:years?|yrs?)\s+(?:of\s+)?(?:experience|exp|relevant|professional|hands[\s-]?on|working)\b', 1),
    # Range: "2-5 years experience"
    (r'\b(\d+)\s*(?:to|-|–)\s*\d+\s*(?:years?|yrs?)\s+(?:of\s+)?(?:experience|exp|relevant|professional|hands[\s-]?on|working)', 1),
    # "experience of N years"
    (r'\bexperience\s+(?:of\s+)?(\d+)\s+(?:years?|yrs?)\b', 1),
    # "exp: 5" or "experience: 5"
    (r'\bexp(?:erience)?\s*[:\-]\s*(\d+)', 1),
]

_TITLE_SENIORITY_MAP = [
    (r'\b(staff|principal|distinguished|fellow)\b', 8),
    (r'\b(lead|architect)\b', 6),
    (r'\b(senior|sr\.?|specialist|l[3-6]|sde[\s-]?iii)\b', 5),
    (r'\b(mid[\s-]?level|intermediate|l2|sde[\s-]?ii(?!i))\b', 3),
    (r'\b(junior|jr\.?|entry[\s-]?level|fresher|trainee|intern|l1|sde[\s-]?i(?!i))\b', 0),
]


def _regex_min_yoe(job: JobListing) -> int | None:
    """
    Extracts minimum YOE from title + description using regex.
    Returns an int if found confidently, None if ambiguous.
    None means: do not pre-filter — let Gemini score it.
    """
    text = f"{job.title}\n{job.description}".lower()

    for pattern, group in _YOE_PATTERNS:
        match = re.search(pattern, text)
        if match:
            return int(match.group(group))

    # Title-based fallback — only the job title (first line)
    title_lower = job.title.lower()
    for pattern, yoe in _TITLE_SENIORITY_MAP:
        if re.search(pattern, title_lower):
            return yoe

    return None


def _passes_hard_filter(job: JobListing, profile: CandidateProfile) -> tuple[bool, str]:
    """
    Zero-cost Python pre-filter. Returns (passes, reason_if_rejected).
    Eliminates non-SDE roles, service companies, wrong locations,
    jobs with no technical keywords, and — critically — over-levelled
    jobs where regex can confidently determine YOE > threshold.
    This keeps the Gemini batch small, cheap, and focused.
    """
    title_lower = job.title.lower()
    company_lower = job.company.lower().strip()

    # 1. Title must look like an SDE role
    for kw in NON_SDE_TITLE_KEYWORDS:
        if kw in title_lower:
            return False, f"Non-SDE title keyword: '{kw}'"

    # 2. Service company blacklist
    if company_lower in SERVICE_COMPANY_BLACKLIST:
        return False, f"Service company: {job.company}"

    # 3. Personal exclusion list (current employer + companies with existing offers)
    if company_lower in EXCLUDED_COMPANIES:
        return False, f"Excluded company: {job.company}"
    if company_lower.startswith(EXCLUDED_COMPANY_PREFIXES):
        return False, f"Excluded company (prefix match): {job.company}"

    # 4. Location must be Bengaluru or Remote
    loc_lower = job.location.lower()
    location_ok = any(
        term in loc_lower
        for term in ["bengaluru", "bangalore", "remote", "work from home", "hybrid"]
    )
    if not location_ok:
        return False, f"Location not Bengaluru/Remote: '{job.location}'"

    # 5. Description must mention at least one of the candidate's technical keywords
    desc_lower = job.description.lower()
    tech_anchor_found = any(
        skill.lower() in desc_lower
        for skill in (profile.core_skills + profile.frameworks + profile.search_keywords)
    )
    if not tech_anchor_found:
        return False, "No technical keywords found in description"

    # 6. YOE pre-filter — drop clearly over-levelled jobs before Gemini sees them
    yoe = _regex_min_yoe(job)
    if yoe is not None and yoe > profile.max_yoe_applying_for:
        return False, f"Over-levelled: regex found {yoe} YOE > threshold {profile.max_yoe_applying_for}"

    return True, ""


GEMINI_BATCH_SCORING_PROMPT = """
You are a technical recruiter scoring job listings for a specific candidate.

Candidate profile:
{profile_summary}

Core skills: {skills}
Seniority: {seniority}
Domains: {domains}
Max YOE the candidate is applying for: {max_yoe}

HARD RULE — apply before any other scoring:
If a job explicitly requires more than {max_yoe} years of experience, assign score = 0.
This overrides all other factors. Do not reward skill match or company quality for over-levelled roles.

Score each remaining job on a scale of 0-100 based on:
- Skill match (40 points): How many of the candidate's core skills does this job require?
- Seniority fit (25 points): Is the role level appropriate for the candidate?
- Domain relevance (20 points): Does the domain match the candidate's background?
- Company quality (15 points): Product company, funded startup, or known tech brand scores higher than unknown.

Return ONLY a raw JSON array — no markdown, no explanation, no code fences.
One object per job, in the same order as input.

Schema: [{{"job_id": "string", "score": integer, "reason": "max 8 words"}}]

Jobs to score:
{jobs_json}
"""


def _extract_json_array(text: str) -> str:
    """Pull the first top-level JSON array out of a possibly fenced/prose response."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        text = text.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end <= start:
        raise ValueError(f"No JSON array found in response: {text[:300]}")
    return text[start:end + 1]


def _score_one_chunk(
    jobs_chunk: list[JobListing],
    profile: CandidateProfile,
    chunk_idx: int,
    chunk_count: int,
) -> dict[str, tuple[int, str]]:
    """Score a single chunk of jobs via Gemma. Returns {job_id: (score, reason)}.

    Failures here degrade gracefully — a parse error returns a default score=50
    for that chunk so we don't lose the whole run to one bad response.
    """
    from agents.agent_1 import gemma_generate

    jobs_for_scoring = [
        {
            "job_id": job.job_id,
            "title": job.title,
            "company": job.company,
            "location": job.location,
            "description_preview": job.description[:600],
        }
        for job in jobs_chunk
    ]

    prompt = GEMINI_BATCH_SCORING_PROMPT.format(
        profile_summary=profile.raw_summary,
        skills=", ".join(profile.core_skills + profile.frameworks),
        seniority=profile.seniority,
        domains=", ".join(profile.domains),
        max_yoe=profile.max_yoe_applying_for,
        jobs_json=json.dumps(jobs_for_scoring, indent=2),
    )

    logger.info(
        f"Ranker chunk {chunk_idx}/{chunk_count}: scoring {len(jobs_chunk)} jobs"
    )

    try:
        raw = gemma_generate(
            prompt,
            max_output_tokens=4096,
            temperature=0.0,
            label=f"ranker:{chunk_idx}/{chunk_count}",
        )
        scored_list = json.loads(_extract_json_array(raw))
    except Exception as e:
        # Includes JSONDecodeError, ValueError, DeadlineExceeded, NotFound, etc.
        # Default-score the chunk so the run survives a single bad response.
        logger.error(f"Ranker chunk {chunk_idx} failed ({type(e).__name__}): {e}")
        return {job.job_id: (50, "chunk failed — default score") for job in jobs_chunk}

    scores: dict[str, tuple[int, str]] = {}
    for item in scored_list:
        try:
            scores[item["job_id"]] = (int(item["score"]), item.get("reason", ""))
        except (KeyError, ValueError, TypeError):
            continue
    return scores


# How many jobs per Gemma scoring call. Empirically: 84-job batches blow past
# the server-side deadline on gemma-4-31b-it (>10 min). 20 keeps each request
# well under a minute and lets a single bad chunk be retried/defaulted in
# isolation. Tune via env without touching code.
_RANKER_CHUNK_SIZE_DEFAULT = 20


def _gemini_batch_score(
    jobs: list[JobListing],
    profile: CandidateProfile
) -> dict[str, tuple[int, str]]:
    """Chunked batch-score across all jobs. Returns {job_id: (score, reason)}.

    Chunks are sized to keep each Gemma call short enough to avoid the
    server-side 504 deadline. The shared gemma_generate helper handles
    primary→fallback model selection and RPM throttling per call.
    """
    chunk_size = int(os.getenv("RANKER_CHUNK_SIZE", _RANKER_CHUNK_SIZE_DEFAULT))
    chunks = [jobs[i:i + chunk_size] for i in range(0, len(jobs), chunk_size)]
    logger.info(
        f"Sending {len(jobs)} jobs to Gemma in {len(chunks)} chunk(s) "
        f"of up to {chunk_size} (prompt-driven JSON)..."
    )

    scores: dict[str, tuple[int, str]] = {}
    for i, chunk in enumerate(chunks, start=1):
        scores.update(_score_one_chunk(chunk, profile, i, len(chunks)))

    logger.info(f"Gemma scored {len(scores)}/{len(jobs)} jobs across {len(chunks)} chunk(s)")
    return scores


def rank_and_filter_jobs(
    jobs: list[JobListing],
    profile: CandidateProfile,
    target_output: int = 55,
    min_score: int = 45
) -> list[tuple[JobListing, int]]:
    """
    Full ranking pipeline:
    1. Hard Python filter (free) — removes obvious non-fits
    2. Gemini batch score (one API call) — intelligent profile matching
    3. Sort by score, return top target_output jobs above min_score
    
    Args:
        jobs: Raw jobs from Agent 0B (typically ~150)
        profile: Candidate profile from Agent 0A
        target_output: How many jobs to pass to Agent 1 (default 55,
                       gives ~50 after YOE kill switch)
        min_score: Minimum Gemini score to be considered (0-100)
    """
    logger.info(f"Ranker input: {len(jobs)} jobs")

    # Step 1: Hard filter (free)
    passed_hard: list[JobListing] = []
    for job in jobs:
        passes, reason = _passes_hard_filter(job, profile)
        if passes:
            passed_hard.append(job)
        else:
            logger.debug(f"Hard filter drop: '{job.title}' @ '{job.company}' — {reason}")

    logger.info(
        f"Hard filter: {len(jobs)} → {len(passed_hard)} "
        f"({len(jobs) - len(passed_hard)} dropped)"
    )

    if not passed_hard:
        logger.warning("All jobs dropped by hard filter. Check your search queries.")
        return []

    # Step 2: Gemini batch score (one API call for all survivors)
    scores = _gemini_batch_score(passed_hard, profile)

    # Step 3: Attach scores, filter by min_score, sort descending
    scored_jobs: list[tuple[JobListing, int]] = []
    for job in passed_hard:
        score, reason = scores.get(job.job_id, (0, "not scored"))
        if score >= min_score:
            scored_jobs.append((job, score))
            logger.info(f"  [{score:3d}] {job.title} @ {job.company} — {reason}")
        else:
            logger.debug(f"  [{score:3d}] DROP: {job.title} @ {job.company} — {reason}")

    scored_jobs.sort(key=lambda x: x[1], reverse=True)

    # Cap at target_output — don't send more than needed to Claude
    final = scored_jobs[:target_output]

    logger.info(
        f"Ranker output: {len(final)} jobs "
        f"(from {len(passed_hard)} post-hard-filter, "
        f"min score {min_score}/100, "
        f"capped at {target_output})"
    )
    return final