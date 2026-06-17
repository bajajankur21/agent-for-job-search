# Resume Generation Migration — DOCX Anchor-Patching → HTML Templating

## 1. Goal

Replace the current map-based DOCX patching renderer with an HTML-templating
pipeline that produces a PDF **visually identical** to
[data/master_resume.pdf](data/master_resume.pdf), while removing the OOXML
hacks and the Word/LibreOffice runtime dependency.

## 2. Why migrate

The current renderer ([agents/docx_renderer.py](agents/docx_renderer.py)) opens
`data/master_resume.docx` and rewrites runs in fixed "anchor" paragraph indices.
This is brittle and carries a long tail of OOXML workarounds:

- Hard-coded paragraph indices (`_ROLE_1_BULLETS = range(7, 12)`, etc.) break
  whenever the master structure changes.
- Spacing tightening, paragraph-mark font normalization, and phantom-page
  collapsing all exist purely to fight Word's layout engine.
- Requires `docx2pdf` (Word) or `libreoffice` at render time; fonts depend on
  whatever is installed on the host.

HTML/CSS gives declarative control over layout, embedded fonts for deterministic
output, and deletes ~360 lines of OOXML manipulation.

## 3. Key insight — all content must become data

Today only the **tailored** fields live in `TailoredAssets`. The master DOCX
silently carries the rest as static content the renderer never touches:

| Content | Today's source | After migration |
| --- | --- | --- |
| Tailored experience bullets, skills, interests, education bullets | `TailoredAssets` (LLM) | `TailoredAssets` (unchanged) |
| Header — name, email, phone, city, `❖` separators | static in master DOCX | `CandidateProfile` + `resume_base.json` |
| Projects section (Spring Code Forger, TestForge) | static in master DOCX | `resume_base.json` |
| Section labels, order, `▪` bullets, header rules | static in master DOCX | template + CSS |

`CandidateProfile` already reaches `agent_2.py` but is discarded with
`del profile` ([agents/agent_2.py:41](agents/agent_2.py#L41)). The header will
now consume it.

**`TailoredAssets` does not change** — same schema, same `**bold**` convention.
No prompt or schema edits, so both the Claude and Gemini/Gemma tailor paths keep
working untouched.

## 4. Target architecture

```
TailoredAssets ─┐
CandidateProfile├─> build_view_model() ─> render_html() ─> render_pdf() ─> PDF ─> S3
resume_base.json┘    (merge into          (Jinja2 +         (WeasyPrint)
                      one flat dict)        md_bold filter)
```

### 4.1 Data layer

New `data/resume_base.json` holds the static, non-tailored content:

```json
{
  "header": { "separator": "❖" },
  "projects": [
    { "name": "Spring Code Forger", "bullets": ["**Agentic Codegen:** ...", "**LLM Orchestration:** ..."] },
    { "name": "TestForge",          "bullets": ["**Test Automation:** ...", "**Agentic Pipeline:** ..."] }
  ],
  "section_order": ["experience", "projects", "education", "skills"]
}
```

`build_view_model(assets, profile, base)` merges `TailoredAssets` +
`CandidateProfile` + `resume_base.json` into one flat dict for the template.

### 4.2 Template layer

- **`templates/resume.html.j2`** — semantic HTML mirroring the resume: header
  block, then sections looped per `section_order`. Each experience role is a
  two-row header (company ↔ dates, title ↔ location) plus a bullet `<ul>`.
- **`templates/resume.css`** — print CSS replacing the OOXML hacks:
  ```css
  @page { size: letter; margin: 0.5in 0.6in; }
  @font-face { font-family: "Garamond"; src: url("fonts/EBGaramond-Regular.ttf"); }
  /* + bold / italic / bold-italic faces */
  .role-row { display: flex; justify-content: space-between; }  /* company ↔ date */
  li::before { content: "▪"; }                                  /* bullet glyph */
  ```
- **`assets/fonts/EBGaramond-*.ttf`** — bundle the four faces. Embedding the
  font removes the Word/LibreOffice dependency *and* host-font variance. EB
  Garamond is the free OFL match for Microsoft Garamond; swap in a licensed
  Garamond TTF if exact match is required.

### 4.3 Rendering layer — new `agents/html_renderer.py` (replaces `docx_renderer.py`)

```python
def render_resume_html(view_model: dict) -> str         # Jinja2 + md_bold filter
def render_resume_pdf(assets, profile, base) -> bytes   # WeasyPrint .write_pdf()
```

- **`md_bold` Jinja filter** replaces `_parse_bold_segments`: HTML-escape first
  (so `<`/`&` in a bullet can't break the doc), then convert `**x**` →
  `<strong>x</strong>`. Same balanced-marker safety as today.
- **Single-page guarantee** becomes a bounded loop instead of magic indices:
  render, check `len(document.pages)`; if `> 1`, nudge a `--scale` CSS variable
  down and re-render (≤3 tries). Retires `_PARAGRAPHS_TO_TIGHTEN`,
  `_collapse_trailing_paragraph`, and the entire anchor map.

### 4.4 PDF engine decision — WeasyPrint

**Recommended: WeasyPrint.** Pure-Python API, no headless browser, strong
`@page` / `@font-face` / flexbox support, deterministic, free, CI-friendly. Its
only cost is native libs (`pango`, `cairo`, `gdk-pixbuf`) — a straight swap for
the `libreoffice` apt install already in CI, and it lets us drop `docx2pdf` and
`python-docx`.

Alternative considered: **Playwright/headless Chromium** — marginally higher CSS
fidelity but a ~150 MB browser download per CI run. Only worth it if
WeasyPrint's print rendering proves insufficient during the fidelity check.

## 5. Integration & cleanup

- **`agents/agent_2.py`**: swap `render_tailored_docx` + `convert_docx_to_pdf`
  for `render_resume_pdf(assets, profile, base)`; drop `del profile`.
- **Retire**: `agents/docx_renderer.py`, `agents/pdf_converter.py`,
  `scripts/inspect_master_docx.py`, `scripts/smoke_render.py`. Add
  `scripts/smoke_render_html.py`.
- **Deps**: add `jinja2`, `weasyprint`; remove `docx2pdf`, `python-docx` (after
  confirming nothing else imports them). CI: replace `apt-get install
  libreoffice` with WeasyPrint native deps.
- **CLAUDE.md**: replace the "Resume renderer anchor map" section with
  template/CSS docs. `data/master_resume.docx` becomes a non-runtime reference;
  `data/master_resume.pdf` stays as the fidelity golden.

## 6. Migration safety

1. Build the template, render with the *current* master content, and diff
   visually against [data/master_resume.pdf](data/master_resume.pdf) until it
   matches.
2. Gate behind `RESUME_RENDERER=html|docx` (default `docx`) for one transition
   period to A/B real runs.
3. Flip the default to `html`, then delete the DOCX path.

## 7. Out of scope (possible follow-ups)

- LLM-tailored Projects section (currently static).
- Editable DOCX output (today only `resume.pdf` is uploaded to S3).
- Multiple resume templates / themes.

---

# Appendix A — Implementation reference

Concrete code/specs to build from. Treat as a starting point, not gospel — the
fidelity pass (Phase 2) will drive the exact CSS numbers.

## A.1 `requirements.txt` changes

Add:

```
jinja2>=3.1
weasyprint>=62.0
```

Remove (only in Phase 6, after the HTML path is the default and nothing else
imports them — `grep -rn "import docx\|from docx\|docx2pdf" .` must come back
clean except the retired files):

```
python-docx>=1.1.2
docx2pdf>=0.1.8 ; sys_platform == "win32"
```

> Note: `pypdf2` stays — Agent 0A still uses it to read the master PDF.

WeasyPrint native libs (macOS dev): `brew install pango gdk-pixbuf libffi`.
CI (Debian/Ubuntu): replace the `libreoffice` install with
`apt-get install -y libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf-2.0-0 libffi-dev libcairo2`.

## A.2 `data/resume_base.json` (full, from current master)

Project bullet text is copied verbatim from [data/master_resume.pdf](data/master_resume.pdf).

```json
{
  "header": { "separator": "❖" },
  "projects": [
    {
      "name": "Spring Code Forger",
      "bullets": [
        "**Agentic Codegen:** Built a multi-agent system that ingests **Swagger/OpenAPI** specs and auto-generates production-ready **Spring Boot** CRUD scaffolding (controllers, services, repos, DTOs).",
        "**LLM Orchestration:** Designed an agent-per-layer pipeline using **LangChain + Claude**, cutting boilerplate authoring time by **~70%** for new microservices."
      ]
    },
    {
      "name": "TestForge",
      "bullets": [
        "**Test Automation:** Built a multi-agent workflow that generates **Selenium/Playwright** automation tests from plain-English user stories, reducing manual QA effort.",
        "**Agentic Pipeline:** Coordinated planner → writer → validator agents for self-healing test generation against live app DOM."
      ]
    }
  ],
  "section_order": ["experience", "projects", "education", "skills"]
}
```

Load it once in the renderer (module-level), not per-job.

## A.3 View model — `build_view_model(assets, profile, base)`

Produces the flat dict the template consumes. Header pulls from
`CandidateProfile`; static blocks from `base`; tailored blocks from `assets`.

```python
def build_view_model(assets, profile, base) -> dict:
    return {
        "name": profile.full_name,
        "contact": [c for c in (profile.email, profile.phone, profile.location_city) if c],
        "separator": base["header"]["separator"],          # ❖
        "experience": [e.model_dump() for e in assets.experience],
        "projects": base["projects"],
        "education": assets.education.model_dump(),
        "skills": assets.skills,                            # dict[str, list[str]]
        "interests": assets.interests,
        "section_order": base["section_order"],
    }
```

Field provenance reference:
- `CandidateProfile`: `full_name`, `email`, `phone`, `location_city`
  ([agents/agent_0a_profiler.py:19-33](agents/agent_0a_profiler.py#L19-L33)).
- `TailoredAssets`: `experience[]` (`company/title/dates/location/bullets`),
  `skills` (4 fixed keys), `interests`, `education`
  ([agents/agent_1.py:139-166](agents/agent_1.py#L139-L166)).

## A.4 `md_bold` filter (escape-then-bold)

Replaces `_parse_bold_segments`
([agents/docx_renderer.py:72-95](agents/docx_renderer.py#L72-L95)). **Escape
first**, then apply bold, and return `markupsafe.Markup` so Jinja won't
double-escape:

```python
import re
from markupsafe import Markup, escape

_BOLD = re.compile(r"\*\*(.+?)\*\*")

def md_bold(text: str) -> Markup:
    safe = str(escape(text))                       # < & > " become entities
    # Unbalanced ** → leave as literal (mirrors current renderer's warning path)
    if safe.count("**") % 2 != 0:
        return Markup(safe)
    return Markup(_BOLD.sub(r"<strong>\1</strong>", safe))
```

Register: `env.filters["md_bold"] = md_bold`.

## A.5 `templates/resume.html.j2`

```jinja
<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><link rel="stylesheet" href="resume.css"></head>
<body>
  <header class="masthead">
    <h1 class="name">{{ name }}</h1>
    <div class="contact">
      {% for c in contact %}{{ c }}{% if not loop.last %} <span class="sep">{{ separator }}</span> {% endif %}{% endfor %} <span class="sep">{{ separator }}</span>
    </div>
  </header>

  {% for section in section_order %}
    {% if section == "experience" %}
      <section><h2 class="section-title">Work Experience</h2>
        {% for role in experience %}
          <div class="role-row"><span class="org">{{ role.company }}</span><span class="dates">{{ role.dates }}</span></div>
          <div class="role-row"><span class="role-title">{{ role.title }}</span><span class="loc">{{ role.location }}</span></div>
          <ul>{% for b in role.bullets %}<li>{{ b | md_bold }}</li>{% endfor %}</ul>
        {% endfor %}
      </section>
    {% elif section == "projects" %}
      <section><h2 class="section-title">Projects</h2>
        {% for p in projects %}
          <div class="proj-name">{{ p.name }}</div>
          <ul>{% for b in p.bullets %}<li>{{ b | md_bold }}</li>{% endfor %}</ul>
        {% endfor %}
      </section>
    {% elif section == "education" %}
      <section><h2 class="section-title">Education</h2>
        <div class="edu-line">{{ education.institution }} | {{ education.degree }} | {{ education.date }}</div>
        <ul>{% for b in education.bullets %}<li>{{ b | md_bold }}</li>{% endfor %}</ul>
      </section>
    {% elif section == "skills" %}
      <section><h2 class="section-title">Certifications, Skills &amp; Interests</h2>
        <ul class="skills">
          {% for cat, items in skills.items() %}<li><strong>{{ cat }}:</strong> {{ items | join(", ") }}.</li>{% endfor %}
          <li><strong>Interests:</strong> {{ interests }}{% if not interests.endswith(".") %}.{% endif %}</li>
        </ul>
      </section>
    {% endif %}
  {% endfor %}
</body>
</html>
```

> The skills section keeps the master's exact 4-key order via `skills.items()`
> because `TailoredAssets.skills` is built with the fixed key order
> (`_REQUIRED_SKILL_KEYS` in [agents/agent_1.py](agents/agent_1.py)).

## A.6 `templates/resume.css` (starting point)

```css
:root { --scale: 1; }                 /* single-page fit loop decrements this */

@page { size: letter; margin: 0.5in 0.6in; }

@font-face { font-family: "Garamond"; font-weight: normal; font-style: normal;
             src: url("fonts/EBGaramond-Regular.ttf"); }
@font-face { font-family: "Garamond"; font-weight: bold; font-style: normal;
             src: url("fonts/EBGaramond-Bold.ttf"); }
@font-face { font-family: "Garamond"; font-weight: normal; font-style: italic;
             src: url("fonts/EBGaramond-Italic.ttf"); }
@font-face { font-family: "Garamond"; font-weight: bold; font-style: italic;
             src: url("fonts/EBGaramond-BoldItalic.ttf"); }

body { font-family: "Garamond", serif;
       font-size: calc(10.5pt * var(--scale)); line-height: 1.18; color: #000; }

.name { font-size: calc(24pt * var(--scale)); font-weight: bold; margin: 0; }
.contact { font-size: calc(11pt * var(--scale)); margin: 2pt 0 6pt; }
.sep { font-size: 0.85em; vertical-align: 1px; }

.section-title { font-weight: bold; text-transform: uppercase; font-size: 11pt;
                 border-bottom: 1px solid #000; margin: 8pt 0 4pt; padding-bottom: 1pt; }

.role-row { display: flex; justify-content: space-between; }
.org { font-weight: bold; }
.role-title, .loc { font-style: italic; }
.dates { font-weight: bold; }
.proj-name, .edu-line { font-weight: bold; margin-top: 3pt; }

ul { list-style: none; margin: 2pt 0; padding-left: 14pt; }
li { position: relative; margin: 1.5pt 0; }
li::before { content: "▪"; position: absolute; left: -12pt; font-size: 0.7em; top: 0.15em; }
ul.skills li { margin: 1pt 0; }
```

> Using `li::before { content: "▪" }` instead of native list markers is what
> lets us drop the Noto-Sans-Symbols paragraph-mark hacks
> ([agents/docx_renderer.py:98-129](agents/docx_renderer.py#L98-L129)).

## A.7 `agents/html_renderer.py` (skeleton)

```python
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML

_TEMPLATES = Path("templates")
_BASE = json.loads(Path("data/resume_base.json").read_text())

_env = Environment(loader=FileSystemLoader(_TEMPLATES), autoescape=True)
_env.filters["md_bold"] = md_bold

def render_resume_html(view_model: dict) -> str:
    return _env.get_template("resume.html.j2").render(**view_model)

def render_resume_pdf(assets, profile, base=_BASE) -> bytes:
    vm = build_view_model(assets, profile, base)
    for scale in (1.0, 0.95, 0.9):          # single-page fit loop
        html = render_resume_html({**vm, "scale": scale})
        # inject --scale; or pass via a <style> override / stylesheet variable
        doc = HTML(string=html, base_url=str(_TEMPLATES)).render()
        if len(doc.pages) <= 1:
            return doc.write_pdf()
    return doc.write_pdf()                    # last attempt wins
```

> `base_url=str(_TEMPLATES)` is what lets WeasyPrint resolve `resume.css` and the
> `fonts/*.ttf` relative URLs. Keep `assets/fonts` reachable from there (symlink,
> copy, or point the `@font-face` URLs at an absolute path).
> The `--scale` injection detail (CSS variable vs. inline `<style>`) is an
> implementation choice to settle in Phase 3.

## A.8 `agents/agent_2.py` integration

Replace ([agents/agent_2.py:59-61](agents/agent_2.py#L59-L61)):

```python
# 1. Resume PDF — patch master.docx in-memory, then convert to PDF.
docx_bytes = render_tailored_docx(assets, _MASTER_DOCX_PATH)
resume_bytes = convert_docx_to_pdf(docx_bytes)
```

with:

```python
# 1. Resume PDF — render HTML template, then WeasyPrint to PDF.
resume_bytes = render_resume_pdf(assets, profile)
```

Also remove `del profile` ([agents/agent_2.py:41](agents/agent_2.py#L41)) — the
header now consumes it — and drop the `_MASTER_DOCX_PATH` existence check.

During the transition (Phase 5), gate on env:

```python
if os.getenv("RESUME_RENDERER", "docx") == "html":
    resume_bytes = render_resume_pdf(assets, profile)
else:
    resume_bytes = convert_docx_to_pdf(render_tailored_docx(assets, _MASTER_DOCX_PATH))
```

## A.9 Fidelity checklist (Phase 2 gate)

Compare rendered output against [data/master_resume.pdf](data/master_resume.pdf):

- [ ] Single page.
- [ ] Name size/weight; contact line with `❖` separators and trailing `❖`.
- [ ] Section headers uppercase + full-width bottom rule.
- [ ] Company bold-left / dates bold-right on one line; title italic-left /
      location italic-right on the next.
- [ ] `▪` bullet glyph size and indent match.
- [ ] Inline bold renders for tech/metrics; no literal `**` leaks through.
- [ ] Skills block: 4 labeled lines in fixed order + Interests line.
- [ ] Overall vertical rhythm/margins read the same at a glance.
