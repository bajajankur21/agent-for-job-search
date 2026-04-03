import os
import json
import logging
from pydantic import BaseModel, Field
from anthropic import Anthropic
from dotenv import load_dotenv
from agents.agent_0a_profiler import CandidateProfile
from agents.agent_0b_scraper import JobListing

load_dotenv()
logger = logging.getLogger(__name__)
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


class TailoredAssets(BaseModel):
    resume_bullets: list[str] = Field(
        description="6-8 tailored resume bullet points for this specific job"
    )
    cover_letter: str = Field(
        description="Full cover letter text, 3 paragraphs"
    )
    form_answers: dict = Field(
        description="Structured answers for common application form fields"
    )
    job_title_used: str
    company_name_used: str


SYSTEM_PROMPT = """You are an expert technical resume writer and career coach.
You tailor resumes and cover letters to specific job descriptions.
Your output is always precise, achievement-oriented, and passes ATS scanners.
Never fabricate experience. Only amplify and reframe what exists in the master resume."""


def run_tailor(
    job: JobListing,
    profile: CandidateProfile,
    master_resume_text: str
) -> TailoredAssets:
    """
    Calls Claude 3.5 Sonnet with prompt caching on the master resume.
    The master_resume_text block is cached — paid at full price on first call,
    ~10% cost on all subsequent calls within the 5-minute cache window.
    """

    user_prompt = f"""
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

    logger.info(f"Calling Claude for tailoring: '{job.title}' @ {job.company}")

    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    # This block is cached — Anthropic stores the processed
                    # version for 5 minutes after first call
                    {
                        "type": "text",
                        "text": f"Here is my complete master resume for reference:\n\n{master_resume_text}",
                        "cache_control": {"type": "ephemeral"}
                    },
                    {
                        "type": "text",
                        "text": user_prompt
                    }
                ]
            }
        ]
    )

    # Log cache performance
    usage = response.usage
    logger.info(
        f"Token usage — input: {usage.input_tokens}, "
        f"output: {usage.output_tokens}, "
        f"cache_read: {getattr(usage, 'cache_read_input_tokens', 0)}, "
        f"cache_write: {getattr(usage, 'cache_creation_input_tokens', 0)}"
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:-1])

    try:
        data = json.loads(raw)
        return TailoredAssets(**data)
    except Exception as e:
        raise ValueError(f"Agent 2 JSON parse failed for '{job.title}': {e}\nRaw: {raw}")