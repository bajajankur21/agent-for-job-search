# Gemini.md

This file provides guidance to Gemini Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Full production run (needs ANTHROPIC, GEMINI, and AWS credentials)
python main.py

# Configurable test runner — per-agent mode selection (mock | live | gemini)
python test_pipeline.py                               # defaults: mock 0a+2, live 0b+0c, gemini 1
python test_pipeline.py --gemini-only                 # no Claude/AWS cost — gemini for 0a+1, mock 2
python test_pipeline.py --mock agent_0a,agent_0b      # force specific agents to mocks
python test_pipeline.py --live agent_0c               # force specific agents live
python test_pipeline.py --max-jobs 3                  # cap per-job tailor/publish loop

# Discover the correct Gemini model ID for this API key
python list_models.py
```

There is no unit-test framework — `test_pipeline.py` is an end-to-end driver with per-agent mode switches defined in `tests/fixtures.py` (mocks) and `tests/gemini_overrides.py` (Gemini swaps for Claude agents).

## Architecture

Linear 5-agent pipeline. Each agent's Pydantic output model feeds the next agent's input. `main.py` orchestrates; `agents/` holds the stages.

```text
Agent 0A (profiler)  →  CandidateProfile
Agent 0B (scraper)   →  list[JobListing]
       ├─ seen-jobs dedup (S3 state/seen_jobs.json) happens BEFORE 0C
Agent 0C (ranker)    →  list[(JobListing, score)]
Agent 1  (tailor)    →  TailoredAssets          ← runs per-job
Agent 2  (publisher) →  uploads to S3           ← runs per-job
```

### Cost model (why agents use different LLMs)

The split is deliberate and drives most design decisions:

- **Agent 0A (profiler)** — Claude Sonnet via **tool-use** (`produce_tailored_resume`-style forced tool call). Runs once per pipeline invocation.
- **Agent 0B (scraper)** — `python-jobspy` hitting LinkedIn/Indeed/Glassdoor. Free, no API key. Each site is scraped separately per search term because LinkedIn honors `location` while Indeed needs `country_indeed`.
- **Agent 0C (ranker)** — Gemini Flash Lite via **structured output** (`response_schema=list[_JobScore]`). A zero-cost Python hard filter (service-company blacklist, non-SDE title keywords, location check, regex YOE extraction) runs first so Gemini only scores survivors, in **one batched call** for all survivors.
- **Agent 1 (tailor)** — Claude Sonnet with **prompt caching on the master resume** (`cache_control: ephemeral`). The master resume is the large, repeated block — it is cached once and reused across every job in the run (5-minute TTL). Uses tool-use for structured `TailoredAssets` output.
- **Agent 2 (publisher)** — ReportLab builds the resume PDF from `TailoredAssets`; boto3 uploads 3 files per job to S3 under `YYYY-MM-DD/Company_Title/{resume.pdf, form_answers.json, job_info.json}`.

### Dedup (important)

`main.py` loads `state/seen_jobs.json` from S3 before the ranker and filters jobs whose `job_id` is already recorded with `status="published"`. Jobs recorded as `status="failed"` fall through and get retried. Each outcome is recorded into the in-memory state **immediately** via `_record_result` (so a partial run still persists wins and failures), and the whole state is saved back at the end.

State shape (v2): `{"version": 2, "jobs": {job_id: {"status": "published"|"failed", "first_seen": "YYYY-MM-DD", "last_attempt": "YYYY-MM-DD", "company": ..., "title": ...}}}`. A legacy v1 file (`{"job_ids": [...]}`) is detected on load and discarded with a warning — the hash scheme changed alongside v2, so old IDs cannot match new hashes.

`job_id` is `md5(_normalize_for_id(company) + "-" + _normalize_for_id(title))[:12]` from `agent_0b_scraper._make_job_id`. `_normalize_for_id` strips `" | ..."` qualifiers and collapses non-alphanumerics so reposts that differ only in punctuation/whitespace/trailing tags (e.g. `"SDE II, Amazon Now"` vs `"SDE II Amazon Now"`) dedup correctly.

### Gemini override path

`tests/gemini_overrides.py` provides drop-in Gemini replacements **only for the Claude agents** (0A and 1). They reuse the original prompts (`PROFILE_EXTRACTION_PROMPT`, `SYSTEM_PROMPT`) but route through `genai.GenerativeModel(...).generate_content(..., response_schema=CandidateProfile/TailoredAssets)`. When modifying Agent 0A or 1 prompts or output schemas, update **both** the Claude path and the Gemini override to keep `--gemini-only` mode working.

### Models are overrideable via env

Every LLM call reads its model ID from env (`MODEL_PROFILER`, `MODEL_TAILORING`, `MODEL_SCORER`, `MODEL_PROFILER_GEMINI`, `MODEL_TAILORING_GEMINI`) with hardcoded fallbacks. Don't hardcode model strings in agent code — extend the env pattern.

## Env & secrets

Required at runtime (`.env` locally, GitHub Secrets in CI):

- `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`
- `AWS_S3_BUCKET`, `AWS_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`

Tuning knobs (GitHub repo Variables, not Secrets):

- `JOBSPY_SITES` (default `linkedin,indeed`), `JOBSPY_RESULTS_PER_SITE` (25), `JOBSPY_HOURS_OLD` (168), `JOBSPY_MAX_TERMS` (4)
- `MODEL_*` overrides listed above

## Scheduled execution

`.github/workflows/AI Job hunt.yaml` runs `python main.py` twice daily (03:30 and 12:30 UTC = 9 AM and 6 PM IST) and on `workflow_dispatch`. A commented-out `cleanup` job in the same file shows the pattern for nightly S3 cleanup that must **preserve `state/`** (deleting it would cause every job to be re-processed).

## Required input artifact

`data/master_resume.pdf` must exist. Agent 0A extracts text via PyPDF2 — image-only PDFs raise `RuntimeError`. When mocking Agent 0A in `tests/fixtures.py`, also update the mock profile to match the real resume (names, YOE, skills) so downstream tailoring matches reality.
