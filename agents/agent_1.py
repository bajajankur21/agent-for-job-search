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
    bullets: list[str] = Field(
        description="Bullets shaped as '**Lead-in:** body with **bold tech** and **bold metrics**.'"
    )


class EducationEntry(BaseModel):
    institution: str
    degree: str
    date: str
    bullets: list[str] = Field(description="Bullets with bold lead-ins, same shape as experience.")


class TailoredAssets(BaseModel):
    experience: list[ExperienceEntry]
    skills: dict[str, list[str]] = Field(
        description="Exactly 4 keys: 'Languages & Backend', 'Frontend & Architecture', "
                    "'Cloud & DevOps', 'Testing & Design'. JD-relevant items first."
    )
    interests: str = Field(description="Comma-separated interests line from master resume")
    education: EducationEntry
    form_answers: dict = Field(description="Structured answers for common application form fields")
    job_title_used: str
    company_name_used: str


SYSTEM_PROMPT = """You are an expert technical resume writer and career coach.
You tailor resumes and cover letters to specific job descriptions.
Your output is always precise, achievement-oriented, and passes ATS scanners.
Never fabricate experience or invent numbers. Only amplify and reframe what exists in the master resume."""


# Tool schema mirrors TailoredAssets so Claude returns structured data directly.
# Bullet shape: "**Lead-in:** body with **bold tech** and **bold metrics**."
# The docx renderer splits on ** markers to create alternating bold/normal runs
# that match the master resume's inline-emphasis pattern.
TAILORING_TOOL = {
    "name": "produce_tailored_resume",
    "description": "Produce a fully tailored resume and form answers for a specific job application. Always call this tool with all fields populated.",
    "input_schema": {
        "type": "object",
        "properties": {
            "experience": {
                "type": "array",
                "description": "One entry per role in the master resume, in the same order. Most recent role gets 5-6 bullets; older roles get 3 bullets.",
                "items": {
                    "type": "object",
                    "properties": {
                        "company": {"type": "string", "description": "exact from master resume"},
                        "title": {"type": "string", "description": "exact from master resume — do NOT promote, rename, or embellish"},
                        "dates": {"type": "string", "description": "exact from master resume, e.g. 'Aug. 2023 – Present'"},
                        "location": {"type": "string", "description": "exact from master resume"},
                        "bullets": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Each bullet shaped as '**Lead-in Phrase:** body text with **bold tech names** and **bold metrics** from master resume.' Markdown ** markers must appear literally. Lead-in is 1-3 words ending in colon. NEVER invent numbers.",
                        },
                    },
                    "required": ["company", "title", "dates", "location", "bullets"],
                },
            },
            "skills": {
                "type": "object",
                "description": "Exactly 4 keys required, each an array of skills with JD-relevant items first.",
                "properties": {
                    "Languages & Backend": {"type": "array", "items": {"type": "string"}},
                    "Frontend & Architecture": {"type": "array", "items": {"type": "string"}},
                    "Cloud & DevOps": {"type": "array", "items": {"type": "string"}},
                    "Testing & Design": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["Languages & Backend", "Frontend & Architecture", "Cloud & DevOps", "Testing & Design"],
            },
            "interests": {
                "type": "string",
                "description": "Comma-separated interests line from master resume (may be trimmed, but do not invent).",
            },
            "education": {
                "type": "object",
                "properties": {
                    "institution": {"type": "string"},
                    "degree": {"type": "string"},
                    "date": {"type": "string", "description": "e.g. 'July 2023'"},
                    "bullets": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Same bold-lead-in shape as experience bullets. Copy master's education bullets; do not invent.",
                    },
                },
                "required": ["institution", "degree", "date", "bullets"],
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
            "experience", "skills", "interests", "education",
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
- R1. The candidate has EXACTLY {profile.total_yoe} years of experience. Use this number verbatim. Do NOT recompute from resume dates.
- R2. Copy every experience entry from the master resume EXACTLY — same company, title, dates, location. Do NOT invent, omit, reorder, rename, promote, or merge entries.
- R3. Bullet shape is mandatory: "**Lead-in Phrase:** body text with **bold tech names** and **bold metrics/numbers** copied from the master resume." Markdown ** markers must appear literally in the output string. Lead-in is 1-3 words ending in a colon.
- R4. The MOST RECENT role has 5 or 6 bullets. Every older role has exactly 3 bullets. Drop low-signal master bullets on older roles; never invent new ones.
- R5. Every bullet starts with a short bold lead-in, references tech or responsibilities from the target JD, and uses only numbers that appear in the master resume. NEVER invent numbers.
- R6. `skills` has EXACTLY these 4 keys: "Languages & Backend", "Frontend & Architecture", "Cloud & DevOps", "Testing & Design". Each value is a non-empty array. JD-relevant items come first within each category.
- R7. `interests` is a single comma-separated string copied from the master's Interests line (may be trimmed).
- R8. `education` is a structured object: institution, degree, date, plus bullets that use the same bold-lead-in shape as experience.
- R9. `form_answers` must reference exact titles from the master resume — do NOT promote, rename, or embellish titles (e.g. "Software Development Engineer" is NOT "Lead SDE").
- R10. `job_title_used` = "{job.title}" and `company_name_used` = "{job.company}" — copy these exact strings.
- R11. Target single A4 page density: 5-6 bullets on primary role, 3 on older roles, 4 skill categories fully populated.
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

class _GeminiFormAnswers(BaseModel):
    describe_last_role: str
    describe_second_last_role: str
    why_this_company: str
    biggest_achievement: str = Field(description="STAR-format achievement most relevant to this JD")
    notice_period: str
    expected_ctc: str


class _GeminiTailoredAssets(BaseModel):
    experience: list[ExperienceEntry]
    skills: dict[str, list[str]]
    interests: str
    education: EducationEntry
    form_answers: _GeminiFormAnswers
    job_title_used: str
    company_name_used: str


_REQUIRED_SKILL_KEYS = (
    "Languages & Backend",
    "Frontend & Architecture",
    "Cloud & DevOps",
    "Testing & Design",
)


def _gemini_to_tailored(raw: _GeminiTailoredAssets) -> TailoredAssets:
    missing = [k for k in _REQUIRED_SKILL_KEYS if k not in raw.skills]
    if missing:
        raise ValueError(f"Gemma skills missing required keys: {missing}")
    return TailoredAssets(
        experience=raw.experience,
        skills={k: raw.skills[k] for k in _REQUIRED_SKILL_KEYS},
        interests=raw.interests,
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
  "experience": [
    {
      "company": "<exact from master>",
      "title": "<exact from master>",
      "dates": "<exact from master>",
      "location": "<exact from master>",
      "bullets": [
        "**Lead-in:** body with **bold tech** and **bold metrics** ...",
        "**Lead-in:** ..."
      ]
    }
  ],
  "skills": {
    "Languages & Backend": ["<skill>"],
    "Frontend & Architecture": ["<skill>"],
    "Cloud & DevOps": ["<skill>"],
    "Testing & Design": ["<skill>"]
  },
  "interests": "<comma-separated from master>",
  "education": {
    "institution": "<string>",
    "degree": "<string>",
    "date": "<string>",
    "bullets": ["**Leadership:** ...", "**Management:** ..."]
  },
  "form_answers": {
    "describe_last_role": "<string>",
    "describe_second_last_role": "<string>",
    "why_this_company": "<string>",
    "biggest_achievement": "<STAR format>",
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
R2. The candidate has EXACTLY {profile.total_yoe} years of experience. Do not recompute from dates.
R3. Copy every experience entry from the master resume EXACTLY — same company, title, dates, location. Do not invent, omit, reorder, rename, promote, or merge entries.
R4. Every experience bullet MUST follow this shape: "**Lead-in Phrase:** body text with **bold tech names** and **bold metrics/numbers** copied from the master resume." Markdown bold markers (**) must appear literally in the output string. Lead-in is 1-3 words ending in colon.
R5. The MOST RECENT role has 5 or 6 bullets. Every older role has exactly 3 bullets. Drop low-signal master bullets on older roles; never invent new ones.
R6. Every bullet references tech or responsibilities from the target JD, and uses only numbers that appear in the master resume. NEVER invent numbers.
R7. Do NOT include a "summary" field or a "projects" field. The master resume has neither — the output must not either.
R8. `skills` has EXACTLY these 4 keys: "Languages & Backend", "Frontend & Architecture", "Cloud & DevOps", "Testing & Design". Each value is a non-empty array. JD-relevant items first.
R9. `interests` is a single comma-separated string copied from the master's Interests line (may be trimmed, but do not invent).
R10. `education` is an object: {{ "institution": "...", "degree": "...", "date": "...", "bullets": ["**Leadership:** ...", "**Management:** ..."] }}. Bullets use the same bold-lead-in shape as R4.
R11. `form_answers` has all 6 keys populated with non-empty strings: describe_last_role, describe_second_last_role, why_this_company, biggest_achievement (STAR format), notice_period, expected_ctc. Reference exact titles from the master resume — do NOT promote, rename, or embellish titles.
R12. `job_title_used` = "{job.title}" and `company_name_used` = "{job.company}" — copy these exact strings.
R13. Total output must fit on a single A4 resume page. Target density: 5-6 bullets on primary role, 3 on older roles, 4 skill categories fully populated.

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
