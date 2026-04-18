"""
Drop-in Gemini replacements for agents that normally use Claude Sonnet.
Same prompts, same Pydantic output models — just routed through Gemini Flash (free tier).

Only Agent 0A (profiler) and Agent 1 (tailor) need overrides since they are
the only agents that use Claude. All other agents already use Gemini or SerpAPI.
"""

import os
import json
import logging

import google.generativeai as genai
from dotenv import load_dotenv

from agents.agent_0a_profiler import (
    CandidateProfile,
    extract_text_from_pdf,
    PROFILE_EXTRACTION_PROMPT,
)
from agents.agent_0b_scraper import JobListing
from agents.agent_1 import TailoredAssets, SYSTEM_PROMPT

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"


def _get_gemini_model(env_var: str = "GEMINI_MODEL") -> genai.GenerativeModel:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set")
    genai.configure(api_key=api_key)
    model_name = os.getenv(env_var, DEFAULT_GEMINI_MODEL)
    return genai.GenerativeModel(model_name)


def _extract_json(raw: str, array: bool = False) -> str:
    """Extract JSON object or array from a response that may contain thinking preamble."""
    raw = raw.strip()
    if array:
        start, end = raw.find('['), raw.rfind(']')
    else:
        start, end = raw.find('{'), raw.rfind('}')
    if start != -1 and end > start:
        return raw[start:end + 1]
    return raw


# ── Agent 0A override: Profiler via Gemini ────────────────────────────────────

def build_candidate_profile_gemini(resume_pdf_path: str) -> CandidateProfile:
    """Same as agent_0a_profiler.build_candidate_profile but uses Gemini Flash."""
    logger.info(f"[GEMINI OVERRIDE] Extracting text from resume: {resume_pdf_path}")
    resume_text = extract_text_from_pdf(resume_pdf_path)
    logger.info(f"Extracted {len(resume_text)} characters from PDF")

    from datetime import date
    model = _get_gemini_model("MODEL_PROFILER_GEMINI")
    today_date = date.today().strftime("%B %d, %Y")
    prompt = PROFILE_EXTRACTION_PROMPT.format(resume_text=resume_text, today_date=today_date)

    logger.info("[GEMINI OVERRIDE] Building candidate profile with Gemini Flash (structured output)...")
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.0,
            max_output_tokens=4096,
            response_mime_type="application/json",
            response_schema=CandidateProfile,
        ),
    )

    try:
        data = json.loads(response.text)
        profile = CandidateProfile(**data)
    except Exception as e:
        logger.error(f"Profile extraction failed: {e}\nRaw response: {response.text[:500]}")
        raise ValueError(f"Could not parse candidate profile: {e}")

    logger.info(
        f"Profile built: {profile.full_name} | "
        f"{profile.total_yoe} YOE | {profile.seniority} | "
        f"Keywords: {profile.search_keywords}"
    )
    return profile


# ── Agent 1 override: Tailor via Gemini ──────────────────────────────────────

def run_tailor_gemini(
    job: JobListing,
    profile: CandidateProfile,
    master_resume_text: str,
) -> TailoredAssets:
    """Same as agent_1.run_tailor but uses Gemini Flash with native structured output."""
    model = _get_gemini_model("MODEL_TAILORING_GEMINI")

    user_prompt = f"""{SYSTEM_PROMPT}

Here is my complete master resume for reference:

{master_resume_text}

Here is the job I am applying to:

Company: {job.company}
Role: {job.title}
Location: {job.location}
Job Description:
---
{job.description}
---

My candidate profile summary: {profile.raw_summary}
Total years of experience: {profile.total_yoe}

Produce a complete tailored resume and application materials.

Rules:
- CRITICAL: The candidate has exactly {profile.total_yoe} years of experience. Always use this exact number. Do NOT calculate, estimate, or round YOE from resume dates — use {profile.total_yoe} as-is.
- Copy all experience entries and projects EXACTLY as they appear in the master resume (company names, titles, dates, locations). Do NOT invent or omit any.
- For each experience entry, write 3-5 achievement bullets reframed toward this specific JD. Use action verbs + tech from JD + quantified impact from master resume. NEVER invent numbers.
- summary: 2-3 sentences positioning me for THIS specific role. Do NOT include the words 'tailored to' or 'tailored for'.
- skills: group into 3-4 categories (e.g. Languages, Frameworks, Tools & Cloud, Databases). Include all skills from master resume; highlight those relevant to JD first.
- projects: include all projects from master resume. Write 1-2 bullets per project emphasising relevance to this JD.
- education: single line in format "Degree | Institution | Graduation Year".
- form_answers: include describe_last_role, describe_second_last_role, why_this_company, biggest_achievement (STAR format), notice_period, expected_ctc.
- job_title_used and company_name_used: use the exact values from the listing above.
"""

    logger.info(f"[GEMINI OVERRIDE] Calling Gemini for tailoring: '{job.title}' @ {job.company} (structured output)")

    import time
    from google.api_core.exceptions import ResourceExhausted

    for attempt in range(3):
        try:
            response = model.generate_content(
                user_prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=8192,
                    response_mime_type="application/json",
                    response_schema=TailoredAssets,
                ),
            )
            break
        except ResourceExhausted:
            if attempt == 2:
                raise
            wait = 60
            logger.warning(f"Rate limit hit, retrying in {wait}s... (attempt {attempt + 1}/3)")
            time.sleep(wait)

    try:
        data = json.loads(response.text)
        return TailoredAssets(**data)
    except Exception as e:
        raise ValueError(f"Gemini tailor parse failed for '{job.title}': {e}\nRaw: {response.text[:500]}")
