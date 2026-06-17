import os
import json
import logging
from typing import Optional
from pathlib import Path

import PyPDF2
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class CandidateProfile(BaseModel):
    """
    Format-agnostic candidate profile. Every field has a sensible default
    so the model never fails even on a sparse resume.
    """
    full_name: str = Field(default="Candidate")
    current_title: str = Field(default="Software Engineer")
    total_yoe: float = Field(default=2.0, description="Total years of experience")
    core_skills: list[str] = Field(default_factory=list)
    frameworks: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list, description="e.g. HealthTech, IoT, Design Systems")
    seniority: str = Field(default="mid", description="junior / mid / senior / staff")
    companies: list[str] = Field(default_factory=list)
    education: Optional[str] = Field(default=None)
    # Contact info — extracted from resume for PDF header
    email: Optional[str] = Field(default=None)
    phone: Optional[str] = Field(default=None)
    linkedin: Optional[str] = Field(default=None)
    github: Optional[str] = Field(default=None)
    location_city: Optional[str] = Field(default=None, description="City shown in resume header")
    # Hard constraints for job matching — these are filters, not preferences
    preferred_locations: list[str] = Field(default=["Bengaluru", "Remote"])
    company_type_preference: str = Field(
        default="product",
        description="product / service / both"
    )
    max_yoe_applying_for: int = Field(
        default=4,
        description="Kill switch threshold. Don't apply to jobs requiring more than this."
    )
    search_keywords: list[str] = Field(
        default_factory=list,
        description="Auto-generated keywords for job board queries"
    )
    raw_summary: str = Field(
        default="",
        description="One paragraph summary of the candidate for use in prompts"
    )


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts raw text from a PDF. Works with any layout —
    single column, multi-column, or plain text PDFs.
    Falls back gracefully if a page can't be read.
    """
    text_parts = []
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(reader.pages):
                try:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
                except Exception as e:
                    logger.warning(f"Could not read page {i}: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF at {pdf_path}: {e}")

    if not text_parts:
        raise RuntimeError(
            "PDF extraction returned empty text. "
            "This can happen with image-only PDFs. "
            "Please export your resume as a text-based PDF."
        )

    return "\n".join(text_parts)


_PROFILE_JSON_TEMPLATE = """{
  "full_name": "<string>",
  "current_title": "<string>",
  "total_yoe": <number>,
  "core_skills": ["<skill>"],
  "frameworks": ["<framework>"],
  "domains": ["<domain>"],
  "seniority": "junior | mid | senior | staff",
  "companies": ["<company>"],
  "education": "<string or null>",
  "email": "<string or null>",
  "phone": "<string or null>",
  "linkedin": "<string or null>",
  "github": "<string or null>",
  "location_city": "<string or null>",
  "preferred_locations": ["Bengaluru", "Remote"],
  "company_type_preference": "product | service | both",
  "max_yoe_applying_for": <integer>,
  "search_keywords": ["<8-12 short keywords>"],
  "raw_summary": "<one paragraph summary>"
}"""

_PROFILE_OVERRIDES_PATH = Path("data/candidate_profile_overrides.json")


def _load_profile_overrides() -> dict:
    try:
        return json.loads(_PROFILE_OVERRIDES_PATH.read_text())
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse {_PROFILE_OVERRIDES_PATH}: {e}") from e


def _apply_profile_overrides(profile: CandidateProfile, overrides: dict) -> CandidateProfile:
    data = profile.model_dump()
    for key, value in overrides.items():
        if key in data:
            data[key] = value
    return CandidateProfile(**data)


PROFILE_EXTRACTION_PROMPT = """
You are an expert technical recruiter. Extract a structured candidate profile from the resume text below.

Today's date is: {today_date}

Rules:
- Work with whatever information is present. Do not require specific sections or formats.
- If a field cannot be determined, use a sensible default — never omit a field.
- Extract contact info from the resume header/contact block when present: email, phone, LinkedIn URL, GitHub URL, and city.
- For linkedin/github, preserve the exact profile URL or handle as written in the resume. Do not invent URLs.
- For total_yoe: calculate from work history dates. "Present" means today ({today_date}). Include ALL roles — full-time AND internships — in the total. Round to nearest 0.5.
- For seniority: infer from titles, YOE, and scope of responsibilities.
- For search_keywords: generate 8-12 SHORT job title keywords (1-3 words each) for Google Jobs search. These will be used directly as search queries, so make them seniority-appropriate. For junior/entry-level candidates, include role variants like "Junior Developer", "SDE 1", "Associate Engineer", "Graduate Engineer". For mid-level, include both plain and "SDE II" style. For senior, include "Senior", "Lead". Always include plain role titles (e.g. "Backend Engineer") alongside seniority-qualified ones. Each keyword must work as a standalone search term.
- For company_type_preference: always set to "product" unless the resume strongly signals service/consulting background.
- For preferred_locations: default to ["Bengaluru", "Remote"] unless other locations are stated.
- For max_yoe_applying_for: always set to 4. This is a strict ceiling to filter out all Senior, Lead, and 5+ YOE roles, ensuring the candidate only sees mid-level and junior opportunities.

Output rules:
- Output ONE JSON object only. No prose, no markdown fences. Start with {{ and end with }}.
- Every field in the schema must be present. Use null only where the schema allows it.
- search_keywords must contain 8 to 12 short keywords (1-3 words each).

Schema (fill every field with real content, not placeholders):
{json_template}

Resume text:
---
{resume_text}
---
"""


def build_candidate_profile(resume_pdf_path: str) -> CandidateProfile:
    """
    Main entry point. Reads the PDF, asks Gemma (primary gemma-4-31b-it, fallback
    gemma-3-27b-it) for a structured profile, validates with Pydantic, returns it.
    Called once per pipeline run — not once per job.
    """
    # Local import avoids the agent_1 → agent_0a circular at module load time.
    from agents.agent_1 import gemma_generate, _extract_json_object
    from pydantic import ValidationError

    logger.info(f"Extracting text from resume: {resume_pdf_path}")
    resume_text = extract_text_from_pdf(resume_pdf_path)
    logger.info(f"Extracted {len(resume_text)} characters from PDF")

    from datetime import date
    today_date = date.today().strftime("%B %d, %Y")
    prompt = PROFILE_EXTRACTION_PROMPT.format(
        resume_text=resume_text,
        today_date=today_date,
        json_template=_PROFILE_JSON_TEMPLATE,
    )

    logger.info("Building candidate profile with Gemma (prompt-driven JSON)...")
    overrides = _load_profile_overrides()

    last_err: Exception | None = None
    last_raw = ""
    for attempt in range(3):
        try:
            last_raw = gemma_generate(
                prompt,
                max_output_tokens=2048,
                temperature=0.0,
                label="profiler",
            )
            data = json.loads(_extract_json_object(last_raw))
            profile = CandidateProfile(**data)
            profile = _apply_profile_overrides(profile, overrides)
            break
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            last_err = e
            if attempt == 2:
                logger.error(f"Profile extraction failed: {e}\nRaw response: {last_raw[:500]}")
                raise ValueError(f"Could not validate candidate profile after 3 attempts: {e}")
            logger.warning(f"Profile JSON invalid, retrying ({attempt + 1}/3): {e}")
    else:
        raise RuntimeError(f"Unexpected fallthrough in build_candidate_profile: {last_err}")

    logger.info(
        f"Profile built: {profile.full_name} | "
        f"{profile.total_yoe} YOE | "
        f"{profile.seniority} | "
        f"Keywords: {profile.search_keywords}"
    )
    return profile
