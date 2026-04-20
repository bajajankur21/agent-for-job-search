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
    location: str
    bullets: list[str]


class ProjectEntry(BaseModel):
    name: str
    tech_stack: str
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
                            "minItems": 2,
                            "maxItems": 3,
                            "description": "2-3 achievement bullets reframed toward this JD. Action verbs + tech from JD + quantified impact from master resume. NEVER invent numbers. Keep each bullet under ~20 words so the resume fits on one page.",
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
                "maxItems": 2,
                "description": "Top 2 most JD-relevant projects from the master resume. Drop the rest to keep the resume on one page.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "tech_stack": {"type": "string", "description": "comma-separated tech stack"},
                        "bullets": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "maxItems": 1,
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
- CRITICAL: The resume MUST fit on a single A4 page. Be ruthless — drop low-signal content rather than overflow.
- CRITICAL: The candidate has exactly {profile.total_yoe} years of experience. Always use this exact number. Do NOT calculate, estimate, or round YOE from resume dates — use {profile.total_yoe} as-is.
- Copy all experience entries EXACTLY as they appear in the master resume (company names, titles, dates, locations). Do NOT invent or omit any experience entry.
- For each experience entry, write 2-3 achievement bullets reframed toward this specific JD. Each bullet under ~20 words. Action verbs + tech from JD + quantified impact from master resume. NEVER invent numbers.
- The summary must be 2 concise sentences positioning me for THIS specific role.
- Skills: group into 3-4 categories. Include JD-relevant skills first; trim less relevant ones.
- Projects: include ONLY the top 2 projects from the master resume most relevant to this JD. Write 1 bullet per project. Drop the rest — single-page constraint takes priority over completeness.
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


# ── Gemini Flash tier for long-tail jobs ─────────────────────────────────────
# Top-scored jobs go through run_tailor (Claude Sonnet, cached master resume).
# Lower-scored jobs go through run_tailor_gemini (free tier, ~1500 RPD on Flash).
#
# Gemini's response_schema does not support `additionalProperties`, so the
# open-shape TailoredAssets fields (`skills: dict[str, list[str]]` and
# `form_answers: dict`) can't be fed to Gemini directly. We mirror the model
# with fixed-key equivalents, ask Gemini for that shape, then map back.

class _GeminiSkills(BaseModel):
    languages: list[str] = Field(description="Programming languages relevant to the JD")
    frameworks: list[str] = Field(description="Frameworks and libraries relevant to the JD")
    tools_and_cloud: list[str] = Field(description="Build tools, cloud platforms, infrastructure")
    databases: list[str] = Field(description="SQL and NoSQL databases / storage systems")


class _GeminiFormAnswers(BaseModel):
    describe_last_role: str
    describe_second_last_role: str
    why_this_company: str
    biggest_achievement: str = Field(description="STAR-format achievement most relevant to this JD")
    notice_period: str
    expected_ctc: str


class _GeminiTailoredAssets(BaseModel):
    summary: str
    experience: list[ExperienceEntry]
    skills: _GeminiSkills
    projects: list[ProjectEntry]
    education: str
    form_answers: _GeminiFormAnswers
    job_title_used: str
    company_name_used: str


_GEMINI_SKILL_DISPLAY = {
    "languages": "Languages",
    "frameworks": "Frameworks",
    "tools_and_cloud": "Tools & Cloud",
    "databases": "Databases",
}


def _gemini_to_tailored(raw: _GeminiTailoredAssets) -> TailoredAssets:
    skills_dict = {
        _GEMINI_SKILL_DISPLAY[key]: values
        for key, values in raw.skills.model_dump().items()
        if values
    }
    return TailoredAssets(
        summary=raw.summary,
        experience=raw.experience,
        skills=skills_dict,
        projects=raw.projects,
        education=raw.education,
        form_answers=raw.form_answers.model_dump(),
        job_title_used=raw.job_title_used,
        company_name_used=raw.company_name_used,
    )


def run_tailor_gemini(
    job: JobListing,
    profile: CandidateProfile,
    master_resume_text: str,
) -> TailoredAssets:
    """Gemini Flash variant of run_tailor — used for the lower-scored long tail."""
    import json
    import time
    import google.generativeai as genai
    from google.api_core.exceptions import ResourceExhausted

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set — required for Gemini tailor tier")
    genai.configure(api_key=api_key)
    model_name = os.getenv("MODEL_TAILORING_GEMINI") or "gemini-2.5-flash"
    model = genai.GenerativeModel(model_name)

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

Produce a complete tailored resume and application materials.

Rules:
- CRITICAL: The resume MUST fit on a single A4 page. Be ruthless — drop low-signal content rather than overflow.
- CRITICAL: The candidate has exactly {profile.total_yoe} years of experience. Always use this exact number. Do NOT calculate, estimate, or round YOE from resume dates — use {profile.total_yoe} as-is.
- Copy all experience entries EXACTLY as they appear in the master resume (company names, titles, dates, locations). Do NOT invent or omit any experience entry.
- For each experience entry, write 2-3 achievement bullets reframed toward this specific JD. Each bullet under ~20 words. Action verbs + tech from JD + quantified impact from master resume. NEVER invent numbers.
- summary: 2 concise sentences positioning me for THIS specific role. Do NOT include the words 'tailored to' or 'tailored for'.
- skills: populate the four fixed categories — `languages`, `frameworks`, `tools_and_cloud`, `databases`. JD-relevant items first in each list. Return `[]` for a category that genuinely doesn't apply.
- projects: include ONLY the top 2 projects from the master resume most relevant to this JD. Write 1 bullet per project. Drop the rest — single-page constraint takes priority over completeness.
- education: single line in format "Degree | Institution | Graduation Year".
- form_answers: fill every key — describe_last_role, describe_second_last_role, why_this_company, biggest_achievement (STAR format), notice_period, expected_ctc.
- job_title_used and company_name_used: use the exact values from the listing above.
"""

    logger.info(f"Calling {model_name} for tailoring (Gemini tier): '{job.title}' @ {job.company}")

    response = None
    for attempt in range(3):
        try:
            response = model.generate_content(
                user_prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=8192,
                    response_mime_type="application/json",
                    response_schema=_GeminiTailoredAssets,
                ),
            )
            break
        except ResourceExhausted:
            if attempt == 2:
                raise
            wait = 60
            logger.warning(f"Gemini rate limit hit, retrying in {wait}s... (attempt {attempt + 1}/3)")
            time.sleep(wait)

    try:
        raw = _GeminiTailoredAssets(**json.loads(response.text))
        return _gemini_to_tailored(raw)
    except Exception as e:
        raise ValueError(f"Gemini tailor parse failed for '{job.title}': {e}\nRaw: {response.text[:500]}")
