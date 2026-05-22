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
    build_candidate_profile as _build_candidate_profile_gemma,
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
    """Deprecated alias — production build_candidate_profile is now Gemma-based.

    Kept so test_pipeline.py imports keep working; routes straight through to
    the production builder (gemma-4-31b-it with gemma-3-27b-it fallback).
    """
    logger.info("[GEMINI OVERRIDE] Routing to production Gemma profiler.")
    return _build_candidate_profile_gemma(resume_pdf_path)


# Agent 1 tailor override now lives in agents/agent_1.py as run_tailor_gemini
# (promoted to production path so main.py can use it for the long-tail tier).
# It is re-exported at the top of this file for test_pipeline.py's imports.
