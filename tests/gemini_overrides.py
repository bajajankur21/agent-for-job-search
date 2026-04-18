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

GEMINI_MODEL = "gemini-2.5-flash-lite"


def _get_gemini_model() -> genai.GenerativeModel:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(GEMINI_MODEL)


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
    model = _get_gemini_model()
    today_date = date.today().strftime("%B %d, %Y")
    prompt = PROFILE_EXTRACTION_PROMPT.format(resume_text=resume_text, today_date=today_date)

    logger.info("[GEMINI OVERRIDE] Building candidate profile with Gemini Flash...")
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(temperature=0.0, max_output_tokens=4096),
    )

    raw = _extract_json(response.text)

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


# ── Agent 1 override: Tailor via Gemini ──────────────────────────────────────

def run_tailor_gemini(
    job: JobListing,
    profile: CandidateProfile,
    master_resume_text: str,
) -> TailoredAssets:
    """Same as agent_1.run_tailor but uses Gemini Flash instead of Claude Sonnet."""
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
Total years of experience: {profile.total_yoe}

Produce a complete tailored resume and application materials as a single JSON object.
No markdown, no explanation — raw JSON only.

Rules:
- CRITICAL: The candidate has exactly {profile.total_yoe} years of experience. Always use this exact number. Do NOT calculate, estimate, or round YOE from resume dates — use {profile.total_yoe} as-is.
- Copy all experience entries and projects EXACTLY as they appear in the master resume (company names, titles, dates, locations). Do NOT invent or omit any.
- For each experience entry, write 3-5 achievement bullets reframed toward this specific JD. Use action verbs + tech from JD + quantified impact from master resume. NEVER invent numbers.
- The summary must be 2-3 sentences positioning me for THIS specific role.
- Skills: group into 3-4 categories (e.g. Languages, Frameworks, Tools & Cloud, Databases). Include all skills from master resume; highlight those relevant to JD first.
- Projects: include all projects from master resume. Write 1-2 bullets per project emphasising relevance to this JD.
- Education: single line in format "Degree | Institution | Graduation Year".

Required JSON schema:
{{
  "summary": "WRITE a 2-3 sentence professional summary positioning the candidate for this specific role. Do NOT include the words 'tailored to' or 'tailored for'.",
  "experience": [
    {{
      "company": "Company name from resume",
      "title": "Job title from resume",
      "dates": "Date range from resume e.g. Jan 2023 – Present",
      "location": "City, Country",
      "bullets": [
        "Achievement bullet 1 reframed toward this JD",
        "Achievement bullet 2",
        "Achievement bullet 3"
      ]
    }}
  ],
  "skills": {{
    "Languages": ["Python", "Java"],
    "Frameworks": ["React", "FastAPI"],
    "Tools & Cloud": ["AWS", "Docker", "Git"]
  }},
  "projects": [
    {{
      "name": "Project name from resume",
      "tech_stack": "comma-separated tech stack",
      "bullets": ["What it does and its impact, framed for this JD"]
    }}
  ],
  "education": "Degree | Institution | Year",
  "cover_letter": "Full 3-paragraph cover letter. Para 1: why this role and company specifically. Para 2: 2-3 concrete experiences mapped to their requirements. Para 3: closing with specific value proposition.",
  "form_answers": {{
    "describe_last_role": "2-3 sentences about most recent role relevant to this JD",
    "describe_second_last_role": "2-3 sentences about second most recent role relevant to this JD",
    "why_this_company": "2 sentences specific to this company",
    "biggest_achievement": "One STAR-format achievement most relevant to this JD",
    "notice_period": "Immediate to 30 days",
    "expected_ctc": "Open to discussion based on role scope"
  }},
  "job_title_used": "the exact job title from the listing above",
  "company_name_used": "the exact company name from the listing above"
}}
"""

    logger.info(f"[GEMINI OVERRIDE] Calling Gemini for tailoring: '{job.title}' @ {job.company}")

    import time
    from google.api_core.exceptions import ResourceExhausted

    for attempt in range(3):
        try:
            response = model.generate_content(
                user_prompt,
                generation_config=genai.GenerationConfig(temperature=0.3, max_output_tokens=8192),
            )
            break
        except ResourceExhausted:
            if attempt == 2:
                raise
            wait = 60
            logger.warning(f"Rate limit hit, retrying in {wait}s... (attempt {attempt + 1}/3)")
            time.sleep(wait)

    raw = _extract_json(response.text)

    try:
        data = json.loads(raw)
        return TailoredAssets(**data)
    except Exception as e:
        raise ValueError(f"Gemini tailor JSON parse failed for '{job.title}': {e}\nRaw: {raw[:500]}")
