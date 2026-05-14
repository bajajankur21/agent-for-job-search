# Agent for Job Search

An autonomous AI pipeline that scrapes LinkedIn and Indeed daily, ranks matching jobs, tailors a resume and cover-letter answers per job using Claude/Gemma, and publishes the output to S3.

## Architecture

Linear 5-agent pipeline — each agent's Pydantic output feeds the next:

```text
Agent 0A (profiler)  →  CandidateProfile        — Claude Sonnet, tool-use; extracts structured profile from master_resume.pdf
Agent 0B (scraper)   →  list[JobListing]         — python-jobspy; hits LinkedIn + Indeed, no API key required
       ├─ seen-jobs dedup (S3 state/seen_jobs.json) runs BEFORE 0C
Agent 0C (ranker)    →  list[(JobListing, score)] — Gemini Flash Lite, structured output; Python hard filter runs first
Agent 1  (tailor)    →  TailoredAssets            — top jobs via Claude Sonnet (prompt-cached); long tail via Gemma 4 31B
Agent 2  (publisher) →  uploads to S3             — DOCX patch → PDF → 3 files per job to S3
```

## Resume rendering

Agent 2 patches `data/master_resume.docx` at fixed paragraph anchor indices rather than generating a new document from scratch. This preserves all formatting (bullet glyphs, tab stops, fonts) — only run-level text is replaced. The patched DOCX is converted to PDF via docx2pdf/LibreOffice and uploaded alongside `form_answers.json` and `job_info.json`.

**Anchor indices (tied to the master DOCX structure):**

```python
_ROLE_1_BULLETS = range(7, 12)       # P007–P011 (5 bullets, most-recent role)
_ROLE_2_BULLETS = range(14, 17)      # P014–P016 (3 bullets, intern role)
_EDUCATION_BULLETS = range(28, 30)   # P028–P029
_SKILL_LINE_INDICES = {"Languages & Backend": 32, "Frontend & Architecture": 33,
                       "Cloud & DevOps": 34, "Testing & Design": 35}
_INTERESTS_LINE = 36
```

If you edit the master DOCX (add/remove a bullet, role, or section), re-verify with `python scripts/inspect_master_docx.py` and update `agents/docx_renderer.py`.

## Setup

**Required secrets** (`.env` locally, GitHub Secrets in CI):

```env
ANTHROPIC_API_KEY
GEMINI_API_KEY
AWS_S3_BUCKET
AWS_REGION
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
```

**Tuning knobs** (GitHub repo Variables):

| Variable | Default | Notes |
| --- | --- | --- |
| `TOP_TIER_CLAUDE_COUNT` | `6` | Jobs above this rank use Gemma instead of Claude |
| `GEMMA_RPM_LIMIT` | `14` | Client-side cap; keep below model ceiling (15 for Gemma 4 31B) |
| `JOBSPY_SITES` | `linkedin,indeed` | Comma-separated scrape targets |
| `JOBSPY_RESULTS_PER_SITE` | `25` | Listings per site per search term |
| `JOBSPY_HOURS_OLD` | `168` | Max job age (1 week) |
| `MODEL_PROFILER` / `MODEL_TAILORING` / `MODEL_SCORER` | hardcoded fallbacks | Override any LLM without touching code |

## Running locally vs CI

```bash
# Full run (requires all credentials)
python main.py

# End-to-end test with per-agent mode switches
python test_pipeline.py                    # defaults: mock 0a+2, live 0b+0c, gemini 1
python test_pipeline.py --gemini-only      # zero Claude/AWS cost
python test_pipeline.py --max-jobs 3       # cap the tailor/publish loop
```

GitHub Actions (`.github/workflows/AI Job hunt.yaml`) runs `python main.py` twice daily at 03:30 and 12:30 UTC (9 AM and 6 PM IST) and on `workflow_dispatch`.

## Smoke testing

```bash
python scripts/smoke_render.py
```

Renders `scripts/smoke_resume.docx` and `scripts/smoke_resume.pdf` from a fixture payload without touching S3. Open the output files to verify: Role 1 has exactly 5 bullets, Role 2 has exactly 3, skills appear once, Projects section is untouched.
