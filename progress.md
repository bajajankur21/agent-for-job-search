# Resume HTML Migration — Progress

Tracking the phased build of the HTML-templating resume renderer. See
[Resume_generation_migration_architecture.md](Resume_generation_migration_architecture.md)
for the full design.

Status legend: ☐ todo · ◐ in progress · ☑ done

---

## Phase 0 — Setup & dependencies
- ☑ Add `jinja2>=3.1`, `weasyprint>=62.0` to `requirements.txt`
- ☑ Install macOS native libs: `brew install pango gdk-pixbuf libffi`
- ☑ Confirm import: `python -c "import weasyprint; print(weasyprint.__version__)"`
- ☑ Download EB Garamond (OFL) → `assets/fonts/EBGaramond-{Regular,Bold,Italic,BoldItalic}.ttf`
- **Exit criterion:** `import weasyprint` succeeds; 4 font files on disk.
- **Refs:** Appendix A.1.

## Phase 1 — Static data extraction
- ☑ Create `data/resume_base.json` — copy the full block from Appendix A.2
- ☑ Sanity-check project bullet text matches `data/master_resume.pdf` verbatim
- ☑ Confirm `CandidateProfile` fields exist: `full_name`, `email`, `phone`, `location_city`
      ([agents/agent_0a_profiler.py:19-33](agents/agent_0a_profiler.py#L19-L33))
- **Exit criterion:** every non-tailored string lives in JSON/profile, not the DOCX.
- **Refs:** Appendix A.2, A.3.

## Phase 2 — Template + CSS (fidelity pass)
- ☑ `templates/resume.html.j2` — from Appendix A.5
- ☑ `templates/resume.css` — from Appendix A.6
- ☑ Make `assets/fonts` reachable from `base_url` (copy/symlink into `templates/fonts` or use absolute `@font-face` URLs)
- ☑ Throwaway render with current master content (hardcode a sample view model)
- ☑ Visually diff against `data/master_resume.pdf`; tune CSS numbers
- ☑ Walk the Appendix A.9 fidelity checklist
- **Exit criterion:** rendered PDF visually indistinguishable from the golden; A.9 all checked.
- **Refs:** Appendix A.5, A.6, A.9.

## Phase 3 — Renderer module
- ☑ `agents/html_renderer.py` — skeleton from Appendix A.7
- ☑ `md_bold` filter (Appendix A.4) — verify `<`/`&` escape + unbalanced `**` stays literal
- ☑ `build_view_model(assets, profile, base)` (Appendix A.3)
- ☑ Single-page fit loop: render → `len(doc.pages)` → decrement `--scale` (≤3 tries)
- ☑ Decide `--scale` injection mechanism (CSS var override vs inline `<style>`)
- **Exit criterion:** a sample `TailoredAssets` → valid 1-page PDF bytes, in isolation.
- **Refs:** Appendix A.3, A.4, A.7.

## Phase 4 — Smoke test
- ☑ `scripts/smoke_render_html.py` (mirror `scripts/smoke_render.py`)
- ☑ Build a representative `TailoredAssets` (reuse `tests/fixtures.py` shapes)
- ☑ Assert: non-empty PDF bytes AND `len(doc.pages) == 1`
- ☑ Write output to `scripts/smoke_resume_html.pdf` for eyeballing
- **Exit criterion:** `python scripts/smoke_render_html.py` passes + PDF looks right.
- **Refs:** existing `scripts/smoke_render.py` for the pattern.

## Phase 5 — Pipeline integration (behind flag)
- ☑ Add `RESUME_RENDERER=html|docx` env switch (default `docx`) in `agents/agent_2.py`
- ☑ Wire `render_resume_pdf(assets, profile)` into the HTML branch (Appendix A.8)
- ☑ Remove `del profile`; keep DOCX branch intact for A/B
- ☑ Run `python test_pipeline.py --gemini-only --max-jobs 1` with `RESUME_RENDERER=html`
- ☑ Inspect the uploaded `resume.pdf` from a real run
- **Exit criterion:** full pipeline uploads an HTML-rendered PDF; DOCX path still works when flag unset.
- **Refs:** Appendix A.8.

## Phase 6 — Cutover & cleanup
- ☑ Flip default: `RESUME_RENDERER` defaults to `html`
- ☑ `grep -rn "import docx\|from docx\|docx2pdf\|render_tailored_docx\|convert_docx_to_pdf" .` → clean except retired files
- ☑ Delete `agents/docx_renderer.py`, `agents/pdf_converter.py`, `scripts/inspect_master_docx.py`, `scripts/smoke_render.py`
- ☑ Remove `python-docx`, `docx2pdf` from `requirements.txt` (keep `pypdf2` — Agent 0A needs it)
- ☑ CI: swap `apt-get install libreoffice` → WeasyPrint native libs (Appendix A.1)
- ☑ Update `CLAUDE.md`: replace "Resume renderer anchor map" section with template/CSS docs; note `master_resume.docx` is now a non-runtime reference
- **Exit criterion:** DOCX path deleted; `test_pipeline.py` green; CI green; docs updated.
- **Refs:** Appendix A.1.

---

## Decisions log
- **PDF engine:** WeasyPrint (pure-Python, no browser, embedded fonts). Playwright
  held in reserve if print fidelity falls short.
- **Font:** EB Garamond (OFL) as the Garamond match; replace with licensed
  Garamond TTF if exact match needed.
- **`TailoredAssets` schema:** unchanged — no LLM prompt/schema edits.
