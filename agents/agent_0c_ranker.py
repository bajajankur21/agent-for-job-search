import os
import json
import logging
from agents.agent_0a_profiler import CandidateProfile
from agents.agent_0b_scraper import JobListing
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── Hard blacklists — zero cost to apply ─────────────────────────────────────

SERVICE_COMPANY_BLACKLIST = {
    "tcs", "infosys", "wipro", "hcl", "cognizant", "accenture",
    "capgemini", "tech mahindra", "mphasis", "hexaware", "ltimindtree",
    "persistent", "coforge", "niit", "zensar", "ibm", "dxc",
    "unisys", "syntel", "mastech", "igate"
}

# If any of these appear in the job title, it's not an SDE role
NON_SDE_TITLE_KEYWORDS = {
    "marketing", "sales", "finance", "accounting", "hr ", "human resource",
    "recruiter", "talent", "analyst", "business development", "bd ",
    "content writer", "seo", "social media", "graphic design",
    "product manager", "program manager", "project manager",
    "scrum master", "agile coach", "customer success", "support engineer",
    "technical support", "it support", "helpdesk", "network engineer",
    "system administrator", "sysadmin", "database administrator", "dba"
}


def _passes_hard_filter(job: JobListing, profile: CandidateProfile) -> tuple[bool, str]:
    """
    Zero-cost Python pre-filter. Returns (passes, reason_if_rejected).
    Cuts ~30-40% of jobs before any API call.
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

    # 3. Location must be Bengaluru or Remote (hard constraint)
    loc_lower = job.location.lower()
    location_ok = any(
        term in loc_lower
        for term in ["bengaluru", "bangalore", "remote", "work from home", "hybrid"]
    )
    if not location_ok:
        return False, f"Location not Bengaluru/Remote: '{job.location}'"

    # 4. Description must mention at least one technical keyword
    # (catches jobs scraped accidentally with no tech content)
    desc_lower = job.description.lower()
    tech_anchor_found = any(
        skill.lower() in desc_lower
        for skill in (profile.core_skills + profile.frameworks + profile.search_keywords)
    )
    if not tech_anchor_found:
        return False, "No technical keywords found in description"

    return True, ""


GEMINI_BATCH_SCORING_PROMPT = """
You are a technical recruiter scoring job listings for a specific candidate.

Candidate profile:
{profile_summary}

Core skills: {skills}
Seniority: {seniority}
Domains: {domains}
Applying for max YOE requirement: {max_yoe}

Score each job listing below on a scale of 0-100 based on:
- Skill match (40 points): How many of the candidate's core skills does this job require?
- Seniority fit (25 points): Is the role level appropriate for the candidate?
- Domain relevance (20 points): Does the domain match the candidate's background?
- Company quality (15 points): Product company, funded startup, or known tech brand scores higher than unknown.

IMPORTANT: Return ONLY a JSON array. No explanation, no markdown. One object per job, in the same order as input.

Schema: [{{"job_id": "string", "score": integer, "reason": "max 8 words"}}]

Jobs to score:
{jobs_json}
"""


def _gemini_batch_score(
    jobs: list[JobListing],
    profile: CandidateProfile
) -> dict[str, tuple[int, str]]:
    """
    Sends ALL jobs to Gemini Flash in a single API call.
    Returns dict of job_id → (score, reason).
    
    Batching is the key cost optimization: 100 jobs in one call costs
    the same input tokens as the prompt overhead, not 100× the overhead.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    # Build compact job representations — enough for scoring, not full JD
    jobs_for_scoring = [
        {
            "job_id": job.job_id,
            "title": job.title,
            "company": job.company,
            "location": job.location,
            # Truncate description to 300 chars — enough for scoring,
            # saves tokens significantly at 100 jobs
            "description_preview": job.description[:300]
        }
        for job in jobs
    ]

    prompt = GEMINI_BATCH_SCORING_PROMPT.format(
        profile_summary=profile.raw_summary,
        skills=", ".join(profile.core_skills + profile.frameworks),
        seniority=profile.seniority,
        domains=", ".join(profile.domains),
        max_yoe=profile.max_yoe_applying_for,
        jobs_json=json.dumps(jobs_for_scoring, indent=2)
    )

    logger.info(f"Sending {len(jobs)} jobs to Gemini Flash for batch scoring...")

    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.0,
            max_output_tokens=16384,  # Gemini 2.5 Flash uses thinking tokens — need headroom
        )
    )

    raw = response.text.strip()
    # Gemini 2.5 Flash may prepend thinking tokens — extract the JSON array directly
    start = raw.find('[')
    end = raw.rfind(']')
    if start != -1 and end > start:
        raw = raw[start:end + 1]

    try:
        scored_list = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"Gemini batch scoring parse failed: {e}\nRaw: {raw[:500]}")
        # Graceful degradation: return score 50 for all jobs so pipeline continues
        return {job.job_id: (50, "parse failed — default score") for job in jobs}

    scores: dict[str, tuple[int, str]] = {}
    for item in scored_list:
        try:
            scores[item["job_id"]] = (int(item["score"]), item.get("reason", ""))
        except (KeyError, ValueError):
            continue

    logger.info(f"Gemini scored {len(scores)}/{len(jobs)} jobs successfully")
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