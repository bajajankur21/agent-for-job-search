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


class ExperienceEntry(BaseModel):
    company: str
    title: str
    dates: str
    location: str = ""
    bullets: list[str]


class ProjectEntry(BaseModel):
    name: str
    tech_stack: str = ""
    bullets: list[str]


class TailoredAssets(BaseModel):
    # Full resume sections
    summary: str = Field(description="2-3 sentence professional summary tailored to this role")
    experience: list[ExperienceEntry] = Field(description="Work experience entries with tailored bullets")
    skills: dict[str, list[str]] = Field(description="Categorised skills, e.g. {'Languages': ['Python']}")
    projects: list[ProjectEntry] = Field(description="Key projects from the master resume")
    education: str = Field(description="Education line, e.g. 'B.Tech CS | BITS Pilani | 2022'")
    # Application materials
    cover_letter: str = Field(description="Full 3-paragraph cover letter")
    form_answers: dict = Field(description="Structured answers for common application form fields")
    job_title_used: str
    company_name_used: str


SYSTEM_PROMPT = """You are an expert technical resume writer and career coach.
You tailor resumes and cover letters to specific job descriptions.
Your output is always precise, achievement-oriented, and passes ATS scanners.
Never fabricate experience or invent numbers. Only amplify and reframe what exists in the master resume."""


def run_tailor(
    job: JobListing,
    profile: CandidateProfile,
    master_resume_text: str
) -> TailoredAssets:
    """
    Calls Claude with prompt caching on the master resume.
    Returns a fully structured TailoredAssets object ready for PDF rendering.
    Cache hit on all calls after the first within the 5-minute window.
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

    logger.info(f"Calling Claude for tailoring: '{job.title}' @ {job.company}")

    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    # Cached block — paid once, ~10% cost on subsequent calls
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

    # Robust JSON extraction in case Claude adds any preamble
    start = raw.find('{')
    end = raw.rfind('}')
    if start != -1 and end > start:
        raw = raw[start:end + 1]

    try:
        data = json.loads(raw)
        return TailoredAssets(**data)
    except Exception as e:
        raise ValueError(f"Agent 1 JSON parse failed for '{job.title}': {e}\nRaw: {raw[:500]}")
