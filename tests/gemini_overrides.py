"""
Drop-in Gemini replacements for agents that normally use Claude Sonnet.
Same prompts, same Pydantic output models — just routed through Gemini Flash (free tier).

Only Agent 0A (profiler) and Agent 2 (tailor) need overrides since they are
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
from agents.agent_2 import TailoredAssets, SYSTEM_PROMPT

load_dotenv()
logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.0-flash"


def _get_gemini_model() -> genai.GenerativeModel:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(GEMINI_MODEL)


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1])
    return raw


# ── Agent 0A override: Profiler via Gemini ────────────────────────────────

def build_candidate_profile_gemini(resume_pdf_path: str) -> CandidateProfile:
    """Same as agent_0a_profiler.build_candidate_profile but uses Gemini Flash."""
    logger.info(f"[GEMINI OVERRIDE] Extracting text from resume: {resume_pdf_path}")
    resume_text = extract_text_from_pdf(resume_pdf_path)
    logger.info(f"Extracted {len(resume_text)} characters from PDF")

    model = _get_gemini_model()
    prompt = PROFILE_EXTRACTION_PROMPT.format(resume_text=resume_text)

    logger.info("[GEMINI OVERRIDE] Building candidate profile with Gemini Flash...")
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(temperature=0.0, max_output_tokens=1024),
    )

    raw = _strip_fences(response.text)

    try:
        data = json.loads(raw)
        profile = CandidateProfile(**data)
    except Exception as e:
        logger.error(f"Profile extraction failed: {e}\nRaw response: {raw}")
        raise ValueError(f"Could not parse candidate profile: {e}")

    logger.info(
        f"Profile built: {profile.full_name} | "
        f"{profile.total_yoe} YOE | {profile.seniority} | "
        f"Keywords: {profile.search_keywords}"
    )
    return profile


# ── Agent 2 override: Tailor via Gemini ───────────────────────────────────

def run_tailor_gemini(
    job: JobListing,
    profile: CandidateProfile,
    master_resume_text: str,
) -> TailoredAssets:
    """Same as agent_2.run_tailor but uses Gemini Flash instead of Claude Sonnet."""
    model = _get_gemini_model()

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

Please produce the following as a single JSON object (no markdown, no explanation):

{{
  "resume_bullets": [
    "6 to 8 bullet points that reframe my experience for THIS specific job.",
    "Each bullet: Action verb + specific technology/skill from JD + quantified impact.",
    "Pull metrics and specifics from my master resume — never invent numbers."
  ],
  "cover_letter": "Full 3-paragraph cover letter. Para 1: why this role + company. Para 2: most relevant 2-3 experiences mapped to their requirements. Para 3: closing with specific value proposition.",
  "form_answers": {{
    "describe_last_role": "2-3 sentences about most recent job, tailored to this JD",
    "describe_second_last_role": "2-3 sentences about second most recent job, tailored to this JD",
    "why_this_company": "2 sentences specific to this company",
    "biggest_achievement": "One STAR-format achievement most relevant to this JD",
    "notice_period": "Immediate to 30 days",
    "expected_ctc": "Open to discussion based on role scope"
  }},
  "job_title_used": "{job.title}",
  "company_name_used": "{job.company}"
}}
"""

    logger.info(f"[GEMINI OVERRIDE] Calling Gemini for tailoring: '{job.title}' @ {job.company}")

    response = model.generate_content(
        user_prompt,
        generation_config=genai.GenerationConfig(temperature=0.3, max_output_tokens=2048),
    )

    raw = _strip_fences(response.text)

    try:
        data = json.loads(raw)
        return TailoredAssets(**data)
    except Exception as e:
        raise ValueError(f"Gemini tailor JSON parse failed for '{job.title}': {e}\nRaw: {raw}")
