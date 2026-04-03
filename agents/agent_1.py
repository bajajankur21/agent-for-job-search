import os
import sys
import json
import logging
import re
from typing import Optional

import google.generativeai as genai
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv
from agents.agent_0b_scraper import JobListing

load_dotenv()
logger = logging.getLogger(__name__)


class GatekeeperResult(BaseModel):
    minimum_yoe: int = Field(default=0, ge=0, le=30)
    job_title: str = Field(default="Unknown Role")
    company_name: Optional[str] = Field(default=None)
    reasoning: str = Field(default="")
    passed: bool = Field(default=True)

    @field_validator("minimum_yoe", mode="before")
    @classmethod
    def coerce_yoe(cls, v):
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            nums = re.findall(r"\d+", v)
            if nums:
                return int(nums[0])
        return 0


EXTRACTION_PROMPT = """
Analyze this job description and return ONLY raw JSON — no markdown, no explanation.

Schema:
{{
  "minimum_yoe": <integer>,
  "job_title": "<string>",
  "company_name": "<string or null>",
  "reasoning": "<one sentence>"
}}

Rules: ranges like "2-4 years" → use lower bound (2). "senior" with no number → 5. "lead" → 6. "mid" → 2. "junior" → 0.

Job Description:
---
{job_description}
---
"""


def run_gatekeeper(job: JobListing, max_yoe: int = 3) -> GatekeeperResult:
    """
    Returns GatekeeperResult with passed=True if job clears the YOE threshold.
    Returns passed=False (does NOT sys.exit) so the pipeline can continue
    to the next job instead of dying entirely.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = EXTRACTION_PROMPT.format(
        job_description=f"{job.title} at {job.company}\n\n{job.description}"
    )

    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(temperature=0.0, max_output_tokens=1024)
    )

    raw = response.text.strip()
    # Gemini 2.5 Flash may prepend thinking tokens — extract the JSON object directly
    start = raw.find('{')
    end = raw.rfind('}')
    if start != -1 and end > start:
        raw = raw[start:end + 1]

    try:
        data = json.loads(raw)
        result = GatekeeperResult(**data)
    except Exception as e:
        logger.warning(f"Gatekeeper parse failed for '{job.title}': {e}. Defaulting to pass.")
        result = GatekeeperResult(
            job_title=job.title,
            company_name=job.company,
            reasoning="Parse failed — defaulting to pass"
        )

    result.passed = result.minimum_yoe <= max_yoe

    if not result.passed:
        logger.warning(
            f"KILL SWITCH: '{job.title}' @ {job.company} "
            f"requires {result.minimum_yoe} YOE > threshold {max_yoe}. Skipping."
        )
    else:
        logger.info(
            f"PASSED: '{job.title}' @ {job.company} "
            f"requires {result.minimum_yoe} YOE ≤ threshold {max_yoe}."
        )

    return result