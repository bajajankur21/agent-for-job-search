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
from agents.agent_1 import run_tailor_gemini  # noqa: F401 — re-export for test_pipeline.py

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"


def _get_gemini_model(env_var: str = "GEMINI_MODEL") -> genai.GenerativeModel:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set")
    genai.configure(api_key=api_key)
    model_name = os.getenv(env_var) or DEFAULT_GEMINI_MODEL
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


# Agent 1 tailor override now lives in agents/agent_1.py as run_tailor_gemini
# (promoted to production path so main.py can use it for the long-tail tier).
# It is re-exported at the top of this file for test_pipeline.py's imports.
