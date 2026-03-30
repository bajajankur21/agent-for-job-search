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
    # Hard constraints for job matching — these are filters, not preferences
    preferred_locations: list[str] = Field(default=["Bengaluru", "Remote"])
    company_type_preference: str = Field(
        default="product",
        description="product / service / both"
    )
    max_yoe_applying_for: int = Field(
        default=3,
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
You are an expert technical recruiter. Extract a structured candidate profile from the resume text below.

Rules:
- Work with whatever information is present. Do not require specific sections or formats.
- If a field cannot be determined, use a sensible default — never omit a field.
- For total_yoe: calculate from work history dates if present, otherwise estimate from seniority signals.
- For seniority: infer from titles, YOE, and scope of responsibilities.
- For search_keywords: generate 6-10 job search keywords this person should use on job boards. Think like a recruiter — what would they search to find this person's next role?
- For company_type_preference: always set to "product" unless the resume strongly signals service/consulting background.
- For preferred_locations: default to ["Bengaluru", "Remote"] unless other locations are stated.
- For max_yoe_applying_for: set to total_yoe + 1, capped at 4.

Return ONLY a valid JSON object matching this exact schema. No markdown, no explanation, no code fences.

Schema:
{
  "full_name": string,
  "current_title": string,
  "total_yoe": float,
  "core_skills": [string],
  "frameworks": [string],
  "domains": [string],
  "seniority": "junior" | "mid" | "senior" | "staff",
  "companies": [string],
  "education": string | null,
  "preferred_locations": [string],
  "company_type_preference": "product" | "service" | "both",
  "max_yoe_applying_for": integer,
  "search_keywords": [string],
  "raw_summary": "one paragraph professional summary"
}

Resume text:
---
{resume_text}
---
"""


def build_candidate_profile(resume_pdf_path: str) -> CandidateProfile:
    """
    Main entry point. Reads PDF, calls Claude, returns validated profile.
    This is called once per pipeline run — not once per job.
    """
    logger.info(f"Extracting text from resume: {resume_pdf_path}")
    resume_text = extract_text_from_pdf(resume_pdf_path)
    logger.info(f"Extracted {len(resume_text)} characters from PDF")

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    prompt = PROFILE_EXTRACTION_PROMPT.format(resume_text=resume_text)

    logger.info("Building candidate profile with Claude...")
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1])

    try:
        data = json.loads(raw)
        profile = CandidateProfile(**data)
    except Exception as e:
        logger.error(f"Profile extraction failed: {e}\nRaw response: {raw}")
        raise ValueError(f"Could not parse candidate profile: {e}")

    logger.info(
        f"Profile built: {profile.full_name} | "
        f"{profile.total_yoe} YOE | "
        f"{profile.seniority} | "
        f"Keywords: {profile.search_keywords}"
    )
    return profile