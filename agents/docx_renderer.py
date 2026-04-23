"""Render tailored resume bytes by patching the master_resume.docx in-place.

Strategy: open the master DOCX, rewrite only the run-level text in specific
"anchor" paragraphs (bullets, skill lines, interests, education bullets).
Paragraph-level formatting (bullet glyph via numPr, indentation, tab stops,
fonts) is preserved because we never touch pPr — only runs.

Bullet text from the LLM carries ** markers for inline emphasis. The parser
splits on ** and emits alternating bold/normal runs whose font inherits from
the master's Garamond 12pt body style.
"""
from __future__ import annotations

import copy
import logging
from io import BytesIO
from pathlib import Path

from docx import Document
from docx.shared import Pt
from docx.text.paragraph import Paragraph

from agents.agent_1 import TailoredAssets

logger = logging.getLogger(__name__)

# ── Anchor paragraph indices in data/master_resume.docx ──────────────────────
# Produced by scripts/inspect_master_docx.py. These are stable as long as the
# master resume structure doesn't change (same number of roles + bullets). If
# you add/remove a role or bullet in the master, update these ranges.
_ROLE_1_BULLETS = range(7, 13)      # P007–P012 (6 bullets, most-recent role)
_ROLE_2_BULLETS = range(15, 18)     # P015–P017 (3 bullets, older role)
_EDUCATION_BULLETS = range(22, 24)  # P022–P023
_SKILL_LINE_INDICES = {
    "Languages & Backend": 26,
    "Frontend & Architecture": 27,
    "Cloud & DevOps": 28,
    "Testing & Design": 29,
}
_INTERESTS_LINE = 30

# Body font used throughout — matches master's run-level formatting.
_BODY_FONT_NAME = "Garamond"
_BODY_FONT_SIZE_PT = 12


def _parse_bold_segments(text: str) -> list[tuple[str, bool]]:
    """Split LLM bullet text on ** markers into (text, is_bold) segments.

    "**Lead-in:** body with **bold tech**." →
        [("Lead-in:", True), (" body with ", False), ("bold tech", True), (".", False)]

    Unbalanced ** markers are treated as literal text so we never drop content.
    """
    if "**" not in text:
        return [(text, False)]

    parts = text.split("**")
    # Odd count means balanced pairs: "" before first **, then alternating.
    # Even count means an unclosed ** — bail out and treat the whole string as plain.
    if len(parts) % 2 == 0:
        logger.warning(f"Unbalanced ** in bullet, rendering as plain text: {text[:60]!r}")
        return [(text, False)]

    segments: list[tuple[str, bool]] = []
    for i, part in enumerate(parts):
        if not part:
            continue
        segments.append((part, i % 2 == 1))
    return segments


def _clear_paragraph_runs(paragraph: Paragraph) -> None:
    """Remove all <w:r> children from the paragraph while keeping pPr intact.

    We keep paragraph properties (numPr for bullet glyph, tab stops, indent,
    alignment) untouched — only text-bearing runs go.
    """
    p_elem = paragraph._p
    for r in list(p_elem.findall(".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}r")):
        p_elem.remove(r)


def _apply_body_font(run) -> None:
    """Set Garamond 12pt on a run (idempotent, matches master body style)."""
    run.font.name = _BODY_FONT_NAME
    run.font.size = Pt(_BODY_FONT_SIZE_PT)


def _rewrite_paragraph(paragraph: Paragraph, text: str) -> None:
    """Clear a paragraph's runs and emit new ones from ** bold markers in text.

    Paragraph-level formatting (bullet, indent, alignment) survives untouched
    because we only mutate runs, never pPr.
    """
    _clear_paragraph_runs(paragraph)
    for segment_text, is_bold in _parse_bold_segments(text):
        run = paragraph.add_run(segment_text)
        run.bold = is_bold
        _apply_body_font(run)


def _rewrite_skill_line(paragraph: Paragraph, category: str, skills: list[str]) -> None:
    """Emit '<b>Category:</b> skill1, skill2, ...' preserving paragraph formatting."""
    _clear_paragraph_runs(paragraph)

    label = paragraph.add_run(f"{category}:")
    label.bold = True
    _apply_body_font(label)

    body = paragraph.add_run(f" {', '.join(skills)}.")
    body.bold = False
    _apply_body_font(body)


def _rewrite_interests_line(paragraph: Paragraph, interests: str) -> None:
    """Emit '<b>Interests:</b> ...' preserving paragraph formatting."""
    _clear_paragraph_runs(paragraph)

    label = paragraph.add_run("Interests:")
    label.bold = True
    _apply_body_font(label)

    # Interests may or may not already end with a period — normalize.
    body_text = interests.strip()
    if not body_text.endswith("."):
        body_text += "."
    body = paragraph.add_run(f" {body_text}")
    body.bold = False
    _apply_body_font(body)


def _patch_bullet_range(
    paragraphs: list[Paragraph],
    anchor_range: range,
    tailored_bullets: list[str],
    label: str,
) -> None:
    """Patch N master paragraphs with up to N tailored bullets.

    If fewer bullets are provided than the master has, surplus master paragraphs
    are emptied (their bullet glyph still renders but the line is blank — which
    collapses visually under Word/LibreOffice list behavior). If more bullets
    are provided than master slots, the extras are dropped with a warning.
    """
    master_slots = len(anchor_range)
    given = len(tailored_bullets)

    if given > master_slots:
        logger.warning(
            f"{label}: LLM produced {given} bullets but master has {master_slots} slots — "
            f"dropping the last {given - master_slots}"
        )
        tailored_bullets = tailored_bullets[:master_slots]
    elif given < master_slots:
        logger.warning(
            f"{label}: LLM produced only {given} bullets vs master's {master_slots} slots — "
            f"emptying surplus master paragraphs"
        )

    for slot_idx, p_idx in enumerate(anchor_range):
        paragraph = paragraphs[p_idx]
        if slot_idx < len(tailored_bullets):
            _rewrite_paragraph(paragraph, tailored_bullets[slot_idx])
        else:
            # Empty the paragraph entirely — keeps layout stable, blanks content.
            _clear_paragraph_runs(paragraph)


def render_tailored_docx(assets: TailoredAssets, master_path: Path) -> bytes:
    """Return tailored DOCX bytes by patching the master's anchor paragraphs.

    The master DOCX is loaded fresh each call so mutations never leak across
    per-job renders.
    """
    doc = Document(master_path)
    paragraphs = doc.paragraphs

    # Sanity: the anchor map is tied to a specific master structure.
    if len(paragraphs) < _INTERESTS_LINE + 1:
        raise RuntimeError(
            f"Master DOCX has only {len(paragraphs)} paragraphs — anchor map expects "
            f"at least {_INTERESTS_LINE + 1}. Re-run scripts/inspect_master_docx.py "
            f"after any master edit and update agents/docx_renderer.py anchors."
        )

    # ── Experience bullets ────────────────────────────────────────────────
    if len(assets.experience) >= 1:
        _patch_bullet_range(
            paragraphs, _ROLE_1_BULLETS,
            assets.experience[0].bullets,
            label=f"role 1 ({assets.experience[0].company})",
        )
    if len(assets.experience) >= 2:
        _patch_bullet_range(
            paragraphs, _ROLE_2_BULLETS,
            assets.experience[1].bullets,
            label=f"role 2 ({assets.experience[1].company})",
        )

    # ── Education bullets ─────────────────────────────────────────────────
    _patch_bullet_range(
        paragraphs, _EDUCATION_BULLETS,
        assets.education.bullets,
        label="education",
    )

    # ── Skill lines ───────────────────────────────────────────────────────
    for category, p_idx in _SKILL_LINE_INDICES.items():
        skills = assets.skills.get(category, [])
        if not skills:
            logger.warning(f"Skill category {category!r} is empty — skipping patch")
            continue
        _rewrite_skill_line(paragraphs[p_idx], category, skills)

    # ── Interests line ────────────────────────────────────────────────────
    if assets.interests:
        _rewrite_interests_line(paragraphs[_INTERESTS_LINE], assets.interests)

    buf = BytesIO()
    doc.save(buf)
    return buf.getvalue()
