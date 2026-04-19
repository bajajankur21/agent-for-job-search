import os
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
    summary: str = Field(description="2-3 sentence professional summary tailored to this role")
    experience: list[ExperienceEntry] = Field(description="Work experience entries with tailored bullets")
    skills: dict[str, list[str]] = Field(description="Categorised skills, e.g. {'Languages': ['Python']}")
    projects: list[ProjectEntry] = Field(description="Key projects from the master resume")
    education: str = Field(description="Education line, e.g. 'B.Tech CS | BITS Pilani | 2022'")
    form_answers: dict = Field(description="Structured answers for common application form fields")
    job_title_used: str
    company_name_used: str


SYSTEM_PROMPT = """You are an expert technical resume writer and career coach.
You tailor resumes and cover letters to specific job descriptions.
Your output is always precise, achievement-oriented, and passes ATS scanners.
Never fabricate experience or invent numbers. Only amplify and reframe what exists in the master resume."""


# Tool schema mirrors TailoredAssets so Claude returns structured data directly.
# Eliminates fragile JSON parsing, markdown fences, and preamble stripping.
TAILORING_TOOL = {
    "name": "produce_tailored_resume",
    "description": "Produce a fully tailored resume, cover letter, and form answers for a specific job application. Always call this tool with all fields populated.",
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "2-3 sentence professional summary positioning the candidate for this specific role. Do NOT include the words 'tailored to' or 'tailored for'.",
            },
            "experience": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "company": {"type": "string"},
                        "title": {"type": "string"},
                        "dates": {"type": "string", "description": "e.g. Jan 2023 – Present"},
                        "location": {"type": "string"},
                        "bullets": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 3,
                            "maxItems": 5,
                            "description": "3-5 achievement bullets reframed toward this JD. Action verbs + tech from JD + quantified impact from master resume. NEVER invent numbers.",
                        },
                    },
                    "required": ["company", "title", "dates", "location", "bullets"],
                },
            },
            "skills": {
                "type": "object",
                "description": "3-4 skill categories (e.g. Languages, Frameworks, Tools & Cloud, Databases). Each value is a list of skill names. Highlight JD-relevant skills first.",
                "additionalProperties": {"type": "array", "items": {"type": "string"}},
            },
            "projects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "tech_stack": {"type": "string", "description": "comma-separated tech stack"},
                        "bullets": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "maxItems": 2,
                        },
                    },
                    "required": ["name", "tech_stack", "bullets"],
                },
            },
            "education": {
                "type": "string",
                "description": "Single line: 'Degree | Institution | Graduation Year'",
            },
            "form_answers": {
                "type": "object",
                "properties": {
                    "describe_last_role": {"type": "string"},
                    "describe_second_last_role": {"type": "string"},
                    "why_this_company": {"type": "string"},
                    "biggest_achievement": {"type": "string", "description": "STAR-format achievement most relevant to this JD"},
                    "notice_period": {"type": "string"},
                    "expected_ctc": {"type": "string"},
                },
                "required": [
                    "describe_last_role", "describe_second_last_role",
                    "why_this_company", "biggest_achievement",
                    "notice_period", "expected_ctc",
                ],
            },
            "job_title_used": {"type": "string", "description": "exact job title from the listing"},
            "company_name_used": {"type": "string", "description": "exact company name from the listing"},
        },
        "required": [
            "summary", "experience", "skills", "projects", "education",
            "form_answers", "job_title_used", "company_name_used",
        ],
    },
}


def run_tailor(
    job: JobListing,
    profile: CandidateProfile,
    master_resume_text: str
) -> TailoredAssets:
    """
    Calls Claude with prompt caching on the master resume and tool-use for structured output.
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

Call the `produce_tailored_resume` tool with a complete tailored resume and application materials.

Rules:
- CRITICAL: The candidate has exactly {profile.total_yoe} years of experience. Always use this exact number. Do NOT calculate, estimate, or round YOE from resume dates — use {profile.total_yoe} as-is.
- Copy all experience entries and projects EXACTLY as they appear in the master resume (company names, titles, dates, locations). Do NOT invent or omit any.
- For each experience entry, write 3-5 achievement bullets reframed toward this specific JD. Use action verbs + tech from JD + quantified impact from master resume. NEVER invent numbers.
- The summary must be 2-3 sentences positioning me for THIS specific role.
- Skills: group into 3-4 categories. Include all skills from master resume; highlight those relevant to JD first.
- Projects: include all projects from master resume. Write 1-2 bullets per project emphasising relevance to this JD.
- Education: single line in format "Degree | Institution | Graduation Year".
"""

    model = os.getenv("MODEL_TAILORING") or "claude-sonnet-4-5-20250929"
    logger.info(f"Calling {model} for tailoring: '{job.title}' @ {job.company} (tool-use mode)")

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        tools=[TAILORING_TOOL],
        tool_choice={"type": "tool", "name": "produce_tailored_resume"},
        messages=[
            {
                "role": "user",
                "content": [
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

    tool_use_block = next((b for b in response.content if b.type == "tool_use"), None)
    if tool_use_block is None:
        raise ValueError(f"Claude did not return a tool_use block for '{job.title}'. Got: {response.content}")

    try:
        return TailoredAssets(**tool_use_block.input)
    except Exception as e:
        logger.error(f"Tailor validation failed for '{job.title}': {e}\nInput: {tool_use_block.input}")
        raise ValueError(f"Could not validate tailored assets for '{job.title}': {e}")
