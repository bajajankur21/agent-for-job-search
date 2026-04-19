import os
import json
import logging
from typing import Optional
from pathlib import Path

import anthropic
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


PROFILE_EXTRACTION_PROMPT = """
You are an expert technical recruiter. Extract a structured candidate profile from the resume text below by calling the `extract_candidate_profile` tool.

Today's date is: {today_date}

Rules:
- Work with whatever information is present. Do not require specific sections or formats.
- If a field cannot be determined, use a sensible default — never omit a field.
- For total_yoe: calculate from work history dates. "Present" means today ({today_date}). Include ALL roles — full-time AND internships — in the total. Round to nearest 0.5.
- For seniority: infer from titles, YOE, and scope of responsibilities.
- For search_keywords: generate 8-12 SHORT job title keywords (1-3 words each) for Google Jobs search. These will be used directly as search queries, so make them seniority-appropriate. For junior/entry-level candidates, include role variants like "Junior Developer", "SDE 1", "Associate Engineer", "Graduate Engineer". For mid-level, include both plain and "SDE II" style. For senior, include "Senior", "Lead". Always include plain role titles (e.g. "Backend Engineer") alongside seniority-qualified ones. Each keyword must work as a standalone search term.
- For company_type_preference: always set to "product" unless the resume strongly signals service/consulting background.
- For preferred_locations: default to ["Bengaluru", "Remote"] unless other locations are stated.
- For max_yoe_applying_for: set to total_yoe + 1, capped at 5.

Resume text:
---
{resume_text}
---
"""

# Tool schema for Claude — forces the model to return structured data matching CandidateProfile.
# This eliminates fragile JSON parsing entirely.
PROFILE_EXTRACTION_TOOL = {
    "name": "extract_candidate_profile",
    "description": "Extract a structured candidate profile from a resume. Always call this tool with all fields populated.",
    "input_schema": {
        "type": "object",
        "properties": {
            "full_name": {"type": "string"},
            "current_title": {"type": "string"},
            "total_yoe": {"type": "number", "description": "Total years of experience including internships, rounded to nearest 0.5"},
            "core_skills": {"type": "array", "items": {"type": "string"}},
            "frameworks": {"type": "array", "items": {"type": "string"}},
            "domains": {"type": "array", "items": {"type": "string"}},
            "seniority": {"type": "string", "enum": ["junior", "mid", "senior", "staff"]},
            "companies": {"type": "array", "items": {"type": "string"}},
            "education": {"type": ["string", "null"]},
            "email": {"type": ["string", "null"]},
            "phone": {"type": ["string", "null"]},
            "linkedin": {"type": ["string", "null"]},
            "github": {"type": ["string", "null"]},
            "location_city": {"type": ["string", "null"]},
            "preferred_locations": {"type": "array", "items": {"type": "string"}},
            "company_type_preference": {"type": "string", "enum": ["product", "service", "both"]},
            "max_yoe_applying_for": {"type": "integer"},
            "search_keywords": {"type": "array", "items": {"type": "string"}, "minItems": 8, "maxItems": 12},
            "raw_summary": {"type": "string", "description": "One paragraph professional summary"},
        },
        "required": [
            "full_name", "current_title", "total_yoe", "core_skills", "frameworks",
            "domains", "seniority", "companies", "preferred_locations",
            "company_type_preference", "max_yoe_applying_for", "search_keywords", "raw_summary"
        ],
    },
}


def build_candidate_profile(resume_pdf_path: str) -> CandidateProfile:
    """
    Main entry point. Reads PDF, calls Claude, returns validated profile.
    This is called once per pipeline run — not once per job.
    """
    logger.info(f"Extracting text from resume: {resume_pdf_path}")
    resume_text = extract_text_from_pdf(resume_pdf_path)
    logger.info(f"Extracted {len(resume_text)} characters from PDF")

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    from datetime import date
    today_date = date.today().strftime("%B %d, %Y")
    prompt = PROFILE_EXTRACTION_PROMPT.format(resume_text=resume_text, today_date=today_date)

    model = os.getenv("MODEL_PROFILER") or "claude-sonnet-4-5-20250929"
    logger.info(f"Building candidate profile with {model} (tool-use mode)...")
    response = client.messages.create(
        model=model,
        max_tokens=2048,
        tools=[PROFILE_EXTRACTION_TOOL],
        tool_choice={"type": "tool", "name": "extract_candidate_profile"},
        messages=[{"role": "user", "content": prompt}],
    )

    # Tool use response — Claude returns the extracted data as a structured dict, no JSON parsing needed.
    tool_use_block = next((b for b in response.content if b.type == "tool_use"), None)
    if tool_use_block is None:
        raise ValueError(f"Claude did not return a tool_use block. Got: {response.content}")

    try:
        profile = CandidateProfile(**tool_use_block.input)
    except Exception as e:
        logger.error(f"Profile validation failed: {e}\nInput: {tool_use_block.input}")
        raise ValueError(f"Could not validate candidate profile: {e}")

    logger.info(
        f"Profile built: {profile.full_name} | "
        f"{profile.total_yoe} YOE | "
        f"{profile.seniority} | "
        f"Keywords: {profile.search_keywords}"
    )
    return profile
