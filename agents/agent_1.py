import os
import json
import time
import logging
from collections import deque
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from agents.agent_0a_profiler import CandidateProfile
from agents.agent_0b_scraper import JobListing

load_dotenv()
logger = logging.getLogger(__name__)

# ── Shared Gemma client (primary + fallback) ─────────────────────────────────
# Every LLM call in the pipeline now goes through Gemma on AI Studio. Primary
# model is gemma-4-31b-it (unlimited TPM/RPD, 15 RPM). On primary failure we
# fall back to gemma-3-27b-it (30 RPM, 14.4K RPD ceiling). Both models share
# the same RPM throttle window since they share project quota.

_PRIMARY_MODEL_ENV = "MODEL_TAILORING_GEMINI"
_PRIMARY_MODEL_DEFAULT = "gemma-4-31b-it"
_FALLBACK_MODEL_ENV = "MODEL_FALLBACK_GEMINI"
# Default fallback is the only other Gemma exposed on AI Studio v1beta for this
# key (verified via `python list_models.py`). Gemma 3 IDs are not reachable on
# this account — do not put a Gemma 3 ID here without confirming list_models.
_FALLBACK_MODEL_DEFAULT = "gemma-4-26b-a4b-it"

_GEMMA_CALL_TIMES: deque = deque()


def _wait_for_gemma_rpm_slot() -> None:
    """Block until a new request would stay under the rolling-60s RPM ceiling."""
    limit = int(os.getenv("GEMMA_RPM_LIMIT", "14"))
    now = time.monotonic()
    while _GEMMA_CALL_TIMES and now - _GEMMA_CALL_TIMES[0] > 60:
        _GEMMA_CALL_TIMES.popleft()
    if len(_GEMMA_CALL_TIMES) >= limit:
        wait = 60 - (now - _GEMMA_CALL_TIMES[0]) + 0.5
        if wait > 0:
            logger.info(
                f"Gemma RPM ceiling reached ({limit}/min) — pausing {wait:.1f}s before next call"
            )
            time.sleep(wait)
    _GEMMA_CALL_TIMES.append(time.monotonic())


def _call_model_with_retries(
    model_name: str,
    prompt: str,
    gen_config,
    label: str,
    max_attempts: int = 3,
) -> str:
    """One model + bounded retries on transient server-side failures.

    Transient = ResourceExhausted (429), InternalServerError (500),
    ServiceUnavailable (503), DeadlineExceeded (504). All other exceptions
    propagate immediately so the caller can decide whether to fall back.
    Backoff: 30s, 60s, 120s.
    """
    import google.generativeai as genai
    from google.api_core.exceptions import (
        ResourceExhausted,
        InternalServerError,
        ServiceUnavailable,
        DeadlineExceeded,
    )

    transient = (ResourceExhausted, InternalServerError, ServiceUnavailable, DeadlineExceeded)
    backoffs = [30, 60, 120]

    model = genai.GenerativeModel(model_name)
    last_err: Exception | None = None
    for attempt in range(max_attempts):
        try:
            _wait_for_gemma_rpm_slot()
            logger.info(f"[{label}] calling {model_name} (attempt {attempt + 1}/{max_attempts})")
            response = model.generate_content(prompt, generation_config=gen_config)
            return response.text
        except transient as e:
            last_err = e
            if attempt == max_attempts - 1:
                logger.warning(
                    f"[{label}] {model_name} transient {type(e).__name__} exhausted after {max_attempts} attempts: {e}"
                )
                raise
            wait = backoffs[min(attempt, len(backoffs) - 1)]
            logger.warning(
                f"[{label}] {model_name} transient {type(e).__name__}, sleeping {wait}s: {e}"
            )
            time.sleep(wait)
    raise RuntimeError(f"Unreachable retry loop fall-through: {last_err}")


def gemma_generate(
    prompt: str,
    max_output_tokens: int = 4096,
    temperature: float = 0.2,
    label: str = "gemma",
) -> str:
    """Call Gemma with primary model, fall back to secondary on failure.

    Tries MODEL_TAILORING_GEMINI (default gemma-4-31b-it) with bounded
    retries on transient server errors; on persistent failure or any
    non-transient exception, falls back to MODEL_FALLBACK_GEMINI (default
    gemma-4-26b-a4b-it) with its own retry budget. RPM throttle is
    applied on every attempt.

    Gemma on AI Studio does NOT support JSON mode (response_schema /
    response_mime_type are rejected), so callers must enforce JSON shape
    via prompt + a robust extractor on the returned string.
    """
    import google.generativeai as genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set — required for Gemma calls")
    genai.configure(api_key=api_key)

    primary = os.getenv(_PRIMARY_MODEL_ENV) or _PRIMARY_MODEL_DEFAULT
    fallback = os.getenv(_FALLBACK_MODEL_ENV) or _FALLBACK_MODEL_DEFAULT

    gen_config = genai.GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )

    try:
        return _call_model_with_retries(primary, prompt, gen_config, label, max_attempts=3)
    except Exception as e:
        logger.warning(
            f"[{label}] primary {primary} exhausted ({type(e).__name__}: {e}); "
            f"falling back to {fallback}"
        )

    return _call_model_with_retries(fallback, prompt, gen_config, label, max_attempts=3)


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


# Legacy Claude tool schema. Unused now that every tailor call goes through
# Gemma (no JSON mode → prompt-driven JSON below), but kept as documentation
# of the canonical TailoredAssets shape.
_LEGACY_TAILORING_TOOL = {
    "name": "produce_tailored_resume",
    "description": "Produce a fully tailored resume and form answers for a specific job application. Always call this tool with all fields populated.",
    "input_schema": {
        "type": "object",
        "properties": {
            "experience": {
                "type": "array",
                "description": "One entry per role in the master resume, in the same order. Most recent role gets exactly 5 bullets; older roles get 3 bullets.",
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
    """Tailor a single job's assets via Gemma (primary gemma-4-31b-it, fallback gemma-3-27b-it).

    Uses prompt-enforced JSON + robust parser because Gemma on AI Studio does
    not expose JSON mode / response_schema. Retries on JSON/schema validation
    failures — the model usually recovers on retry. Per-attempt model fallback
    (4-31b → 3-27b) is handled inside ``gemma_generate``.
    """
    from pydantic import ValidationError

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
R5. The MOST RECENT role has exactly 5 bullets. Every older role has exactly 3 bullets. Drop low-signal master bullets on older roles; never invent new ones.
R6. Every bullet references tech or responsibilities from the target JD, and uses only numbers that appear in the master resume. NEVER invent numbers.
R7. Do NOT include a "summary" field or a "projects" field. The master resume has neither — the output must not either.
R8. `skills` has EXACTLY these 4 keys: "Languages & Backend", "Frontend & Architecture", "Cloud & DevOps", "Testing & Design". Each value is a non-empty array. JD-relevant items first.
R9. `interests` is a single comma-separated string copied from the master's Interests line (may be trimmed, but do not invent).
R10. `education` is an object: {{ "institution": "...", "degree": "...", "date": "...", "bullets": ["**Leadership:** ...", "**Management:** ..."] }}. Bullets use the same bold-lead-in shape as R4.
R11. `form_answers` has all 6 keys populated with non-empty strings: describe_last_role, describe_second_last_role, why_this_company, biggest_achievement (STAR format), notice_period, expected_ctc. Reference exact titles from the master resume — do NOT promote, rename, or embellish titles.
R12. `job_title_used` = "{job.title}" and `company_name_used` = "{job.company}" — copy these exact strings.
R13. Total output must fit on a single A4 resume page. Target density: exactly 5 bullets on primary role, 3 on older roles, 4 skill categories fully populated.

=== OUTPUT SCHEMA (fill every field with real content, not placeholders) ===
{_GEMMA_JSON_TEMPLATE}

Reminder: output the JSON object only. Your first character must be {{ and your last character must be }}."""

    label = f"tailor:{job.company}"
    last_err: Exception | None = None
    last_raw: str = ""
    for attempt in range(3):
        try:
            last_raw = gemma_generate(
                prompt,
                max_output_tokens=8192,
                temperature=0.2,
                label=label,
            )
            raw_json = _extract_json_object(last_raw)
            parsed = _GeminiTailoredAssets(**json.loads(raw_json))
            return _gemini_to_tailored(parsed)

        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            last_err = e
            if attempt == 2:
                raise ValueError(
                    f"Gemma output failed JSON/schema validation after 3 attempts for '{job.title}': {e}\n"
                    f"Last raw output: {last_raw[:500]}"
                )
            logger.warning(
                f"Gemma JSON/schema invalid, retrying... (attempt {attempt + 1}/3): {e}"
            )

    raise RuntimeError(f"Unexpected fallthrough in run_tailor_gemini: {last_err}")


# Backwards-compatible alias — every tailor call now routes through Gemma.
run_tailor = run_tailor_gemini
