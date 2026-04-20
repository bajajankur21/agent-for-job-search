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


# ── Gemma tier for long-tail jobs ────────────────────────────────────────────
# Top-scored jobs go through run_tailor (Claude Sonnet, cached master resume).
# Lower-scored jobs go through run_tailor_gemini → Gemma 4 31B on AI Studio
# (free tier: 15 RPM, UNLIMITED TPM, UNLIMITED RPD). We self-throttle to 14
# RPM (sliding window) to stay one below the ceiling; with no daily cap, the
# only pacing constraint is per-minute. For ~40 jobs this adds ~3 min of
# total wall-clock. `gemma-3-27b-it` also works if the user prefers it
# (higher RPM, lower RPD ceiling) — set MODEL_TAILORING_GEMINI to override.
#
# Gemma on AI Studio does NOT support response_schema or response_mime_type
# (JSON mode is disabled: "JSON mode is not enabled for models/gemma-*-it").
# So we drive the JSON shape purely through a strict prompt + robust parser,
# and validate with Pydantic. Fixed-key `_GeminiSkills` + `_GeminiFormAnswers`
# are retained because a literal-key schema is easier for the model to hit.

# Sliding-window RPM limiter shared across all run_tailor_gemini calls in the
# process. Module-level so it persists across the per-job loop in main.py.
from collections import deque as _deque
import time as _time

_GEMMA_CALL_TIMES: _deque = _deque()


def _wait_for_gemma_rpm_slot() -> None:
    """Block until a new request would stay under the rolling-60s RPM ceiling."""
    limit = int(os.getenv("GEMMA_RPM_LIMIT", "14"))
    now = _time.monotonic()
    while _GEMMA_CALL_TIMES and now - _GEMMA_CALL_TIMES[0] > 60:
        _GEMMA_CALL_TIMES.popleft()
    if len(_GEMMA_CALL_TIMES) >= limit:
        wait = 60 - (now - _GEMMA_CALL_TIMES[0]) + 0.5
        if wait > 0:
            logger.info(
                f"Gemma RPM ceiling reached ({limit}/min) — pausing {wait:.1f}s before next call"
            )
            _time.sleep(wait)
    _GEMMA_CALL_TIMES.append(_time.monotonic())

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


def _extract_json_object(text: str) -> str:
    """Strip markdown fences / prose and return the first top-level JSON object."""
    text = text.strip()
    if text.startswith("```"):
        # Drop opening fence (```json or ```) and trailing ```
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"No JSON object found in response: {text[:300]}")
    return text[start:end + 1]


_GEMMA_JSON_TEMPLATE = """{
  "summary": "<exactly 2 sentences positioning the candidate for THIS role>",
  "experience": [
    {
      "company": "<string, exact from master resume>",
      "title": "<string, exact from master resume>",
      "dates": "<e.g. Jan 2023 - Present, exact from master resume>",
      "location": "<string, exact from master resume>",
      "bullets": ["<bullet under 20 words>", "<bullet under 20 words>"]
    }
  ],
  "skills": {
    "languages": ["<lang>"],
    "frameworks": ["<framework>"],
    "tools_and_cloud": ["<tool>"],
    "databases": ["<db>"]
  },
  "projects": [
    {
      "name": "<string>",
      "tech_stack": "<comma-separated>",
      "bullets": ["<single bullet>"]
    }
  ],
  "education": "<Degree | Institution | Graduation Year>",
  "form_answers": {
    "describe_last_role": "<string>",
    "describe_second_last_role": "<string>",
    "why_this_company": "<string>",
    "biggest_achievement": "<STAR-format>",
    "notice_period": "<string>",
    "expected_ctc": "<string>"
  },
  "job_title_used": "<exact title from the listing>",
  "company_name_used": "<exact company from the listing>"
}"""


def run_tailor_gemini(
    job: JobListing,
    profile: CandidateProfile,
    master_resume_text: str,
) -> TailoredAssets:
    """Gemma 4 31B variant of run_tailor — long-tail tier on AI Studio free quota.

    Uses prompt-enforced JSON + robust parser because Gemma on AI Studio does
    not expose JSON mode / response_schema. Client-side RPM throttle keeps us
    under the 15/min ceiling (unlimited RPD on Gemma 4). Retries on rate-limit
    AND on JSON/schema validation failures (the model usually recovers on
    retry once it sees that its first output was malformed).
    """
    import json
    import time
    import google.generativeai as genai
    from google.api_core.exceptions import ResourceExhausted
    from pydantic import ValidationError

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set — required for Gemma tailor tier")
    genai.configure(api_key=api_key)
    model_name = os.getenv("MODEL_TAILORING_GEMINI") or "gemma-4-31b-it"
    model = genai.GenerativeModel(model_name)

    prompt = f"""You are an expert technical resume writer and career coach. You tailor resumes to specific job descriptions. Your output is precise, achievement-oriented, and passes ATS scanners. You never fabricate experience or invent numbers — you only amplify and reframe what exists in the master resume.

Your SINGLE task: output one JSON object matching the schema below. No prose. No markdown fences. No explanations. Start with {{ and end with }}.

=== MASTER RESUME ===
{master_resume_text}
=== END MASTER RESUME ===

=== TARGET JOB ===
Company: {job.company}
Role: {job.title}
Location: {job.location}
Job Description:
{job.description}
=== END TARGET JOB ===

=== CANDIDATE META ===
Summary: {profile.raw_summary}
Total years of experience: {profile.total_yoe}
=== END CANDIDATE META ===

=== HARD RULES — VIOLATING ANY OF THESE IS A FAILURE ===
R1. Output is ONE JSON object. No text before {{. No text after }}. No ```json fences.
R2. The candidate has EXACTLY {profile.total_yoe} years of experience. Use this number verbatim if you mention YOE anywhere. Do not recompute from dates.
R3. Copy every experience entry from the master resume EXACTLY — same company, title, dates, location. Do not invent, omit, reorder, or merge entries.
R4. Every experience bullet is under 20 words, starts with an action verb, and references tech or responsibilities from the target JD. Quantified impact only if the number appears in the master resume. NEVER invent numbers.
R5. Each experience entry has 2 or 3 bullets — no more, no less.
R6. `summary` is EXACTLY 2 sentences. Never use the phrases "tailored to", "tailored for", "seeking to", "passionate about".
R7. `skills` has exactly these 4 keys: "languages", "frameworks", "tools_and_cloud", "databases". Each value is an array of strings. JD-relevant items first. Use [] for a category with nothing relevant — do NOT omit the key.
R8. `projects` has EXACTLY 2 entries — the 2 most JD-relevant projects from the master resume. Each project has exactly 1 bullet. Drop the rest.
R9. `education` is a single line: "Degree | Institution | Graduation Year".
R10. `form_answers` has all 6 keys populated with non-empty strings: describe_last_role, describe_second_last_role, why_this_company, biggest_achievement (STAR format), notice_period, expected_ctc.
R11. `job_title_used` = "{job.title}" and `company_name_used` = "{job.company}" — copy these exact strings.
R12. Total output must fit on a single A4 resume page when rendered. Be ruthless — drop low-signal content.

=== OUTPUT SCHEMA (fill every field with real content, not placeholders) ===
{_GEMMA_JSON_TEMPLATE}

Reminder: output the JSON object only. Your first character must be {{ and your last character must be }}."""

    logger.info(f"Calling {model_name} for tailoring (Gemma tier): '{job.title}' @ {job.company}")

    last_err: Exception | None = None
    for attempt in range(3):
        try:
            _wait_for_gemma_rpm_slot()
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=8192,
                ),
            )
            raw_json = _extract_json_object(response.text)
            parsed = _GeminiTailoredAssets(**json.loads(raw_json))
            return _gemini_to_tailored(parsed)

        except ResourceExhausted as e:
            last_err = e
            if attempt == 2:
                raise
            logger.warning(f"Gemma rate limit hit, retrying in 60s... (attempt {attempt + 1}/3)")
            time.sleep(60)

        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            last_err = e
            if attempt == 2:
                preview = getattr(response, "text", "")[:500] if "response" in dir() else "<no response>"
                raise ValueError(
                    f"Gemma output failed JSON/schema validation after 3 attempts for '{job.title}': {e}\n"
                    f"Last raw output: {preview}"
                )
            logger.warning(
                f"Gemma JSON/schema invalid, retrying... (attempt {attempt + 1}/3): {e}"
            )

    # Unreachable — loop either returns or raises. Guard to satisfy type checkers.
    raise RuntimeError(f"Unexpected fallthrough in run_tailor_gemini: {last_err}")
