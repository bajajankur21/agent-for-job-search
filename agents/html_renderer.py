import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader
from markupsafe import Markup, escape

if sys.platform == "darwin":
    homebrew_lib = Path("/opt/homebrew/lib")
    if homebrew_lib.exists():
        existing = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH")
        homebrew_path = str(homebrew_lib)
        paths = existing.split(os.pathsep) if existing else []
        can_reexec = sys.argv and sys.argv[0] != "-"
        if (
            can_reexec
            and homebrew_path not in paths
            and os.environ.get("RESUME_WEASYPRINT_REEXEC") != "1"
        ):
            env = os.environ.copy()
            env["DYLD_FALLBACK_LIBRARY_PATH"] = (
                homebrew_path if not existing else f"{homebrew_path}{os.pathsep}{existing}"
            )
            env["RESUME_WEASYPRINT_REEXEC"] = "1"
            os.execvpe(sys.executable, [sys.executable, *sys.argv], env)

from weasyprint import HTML

# --- Constants & Setup ---

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TEMPLATES = _PROJECT_ROOT / "templates"
_BASE_DATA_PATH = _PROJECT_ROOT / "data/resume_base.json"
_SCALE_FACTORS = (1.0, 0.95, 0.9, 0.86, 0.82)
_MIN_SCALE = min(_SCALE_FACTORS)

# Load static base data once at module level
try:
    _BASE = json.loads(_BASE_DATA_PATH.read_text())
except (FileNotFoundError, json.JSONDecodeError) as e:
    raise RuntimeError(f"Failed to load resume_base.json: {e}") from e

# Jinja2 Environment setup
_env = Environment(loader=FileSystemLoader(_TEMPLATES), autoescape=True)

# --- Filters ---

_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")

def md_bold(text: str) -> Markup:
    """
    Converts **bold** text to <strong>bold</strong>, mirroring the 
    current DOCX renderer's behavior of escaping first and 
    leaving unbalanced markers as literal.
    """
    if not text:
        return Markup("")
    
    safe = str(escape(text))
    # Unbalanced ** markers should be left as literal text
    if safe.count("**") % 2 != 0:
        return Markup(safe)
    
    return Markup(_BOLD_RE.sub(r"<strong>\1</strong>", safe))

_env.filters["md_bold"] = md_bold

# --- Logic ---

def build_view_model(assets: Any, profile: Any, base: dict = _BASE) -> dict:
    """
    Merges TailoredAssets, CandidateProfile, and resume_base.json 
    into a flat dictionary for the Jinja2 template.
    """
    # assets: TailoredAssets (Pydantic model)
    # profile: CandidateProfile (Pydantic model)
    
    return {
        "name": profile.full_name,
        "contact": [
            c for c in (
                profile.email,
                profile.phone,
                profile.linkedin,
                profile.github,
                profile.location_city,
            ) if c
        ],
        "separator": base["header"]["separator"],
        "experience": [e.model_dump() for e in assets.experience],
        "projects": base["projects"],
        "education": assets.education.model_dump(),
        "skills": assets.skills,
        "interests": assets.interests,
        "section_order": base["section_order"],
    }

def _scaled_pt(base: float, scale: float) -> str:
    return f"{base * scale:.2f}pt"


def _scale_override(scale: float) -> str:
    return (
        "<style>:root { "
        f"--body-font-size: {_scaled_pt(10.5, scale)}; "
        f"--name-font-size: {_scaled_pt(24, scale)}; "
        f"--contact-font-size: {_scaled_pt(11, scale)}; "
        f"--section-font-size: {_scaled_pt(11, scale)}; "
        f"--section-margin-top: {_scaled_pt(8, scale)}; "
        f"--section-margin-bottom: {_scaled_pt(4, scale)}; "
        f"--list-margin-y: {_scaled_pt(2, scale)}; "
        f"--li-margin-y: {_scaled_pt(1.5, scale)}; "
        "} </style>"
    )


def _inject_scale(html: str, scale: float) -> str:
    head = '<head><meta charset="utf-8"><link rel="stylesheet" href="resume.css"></head>'
    if head not in html:
        raise RuntimeError("resume.html.j2 head block changed; cannot inject render scale")
    return html.replace(head, head.replace("</head>", f"{_scale_override(scale)}</head>"))


def render_resume_html(view_model: dict) -> str:
    """
    Renders the HTML template using the provided view model.
    """
    return _env.get_template("resume.html.j2").render(**view_model)

def render_resume_pdf(assets: Any, profile: Any, base: dict = _BASE) -> bytes:
    """
    Renders the final PDF. Includes a bounded loop that adjusts the 
    --scale CSS variable to attempt to fit the content on a single page.
    """
    vm = build_view_model(assets, profile, base)
    
    last_doc = None
    for scale in _SCALE_FACTORS:
        styled_html = _inject_scale(render_resume_html(vm), scale)
        doc = HTML(string=styled_html, base_url=str(_TEMPLATES)).render()
        last_doc = doc
        
        if len(doc.pages) <= 1:
            return doc.write_pdf()

    raise RuntimeError(
        f"Rendered resume is {len(last_doc.pages)} pages even at scale {_MIN_SCALE:.2f}; "
        "shorten tailored content or tighten template spacing."
    )
