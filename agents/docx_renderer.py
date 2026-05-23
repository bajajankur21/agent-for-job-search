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
_ROLE_1_BULLETS = range(7, 12)       # P007–P011 (5 bullets, most-recent role)
_ROLE_2_BULLETS = range(14, 17)      # P014–P016 (3 bullets, intern role)
_EDUCATION_BULLETS = range(28, 30)   # P028–P029
_SKILL_LINE_INDICES = {
    "Languages & Backend": 32,
    "Frontend & Architecture": 33,
    "Cloud & DevOps": 34,
    "Testing & Design": 35,
}
_INTERESTS_LINE = 36

# Paragraphs whose `<w:spacing w:after=…>` we zero out at render time to keep
# the tailored resume on a single page. The master DOCX has 12pt of trailing
# space on the last intern bullet (pushes Projects + Education + Skills down)
# and on the Interests line (wasted space on the last line of the doc). Both
# combined push the Interests line onto page 2.
_PARAGRAPHS_TO_TIGHTEN = (16, 36)

# Paragraphs whose paragraph-mark rPr uses Noto Sans Symbols without an
# explicit <w:sz>, which makes Word render the bullet glyph at the default
# (larger) size and inflates the line height — visible as an extra gap above
# those bullets. Normalize them to Garamond 12pt to match every other bullet.
_PARAGRAPHS_TO_NORMALIZE_MARK = (16, 36)

# Index of the trailing empty paragraph Word forces into the document. With
# default line height it can overflow onto a phantom page 2 when the layout
# is tight. Render it with a 1pt font + no line spacing to collapse it.
_TRAILING_EMPTY_PARAGRAPH = 37

# Body font used throughout — matches master's run-level formatting.
_BODY_FONT_NAME = "Garamond"
_BODY_FONT_SIZE_PT = 12

_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _qn(tag: str) -> str:
    """Qualify a w:-prefixed tag name with the wordprocessingml namespace."""
    return _W_NS + tag


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


def _normalize_paragraph_mark_size(paragraph: Paragraph) -> None:
    """Set the paragraph-mark size to 12pt without touching its font.

    Word renders the list bullet glyph using the paragraph mark's font + size
    (the rPr inside pPr). Two bullets in the master (P016 last intern bullet
    and P036 Interests line) declare a font (Noto Sans Symbols) but NO size,
    so the glyph inherits a larger size from style defaults — which inflates
    line height and shows up as a gap above those rows. Pinning only the size
    to 12pt (matches every other body bullet) removes the gap. We deliberately
    leave rFonts alone — the master's bullet character renders correctly in
    Noto Sans Symbols and falls back to a smaller glyph in Garamond.
    """
    ppr = paragraph._p.find(_qn("pPr"))
    if ppr is None:
        return
    rpr = ppr.find(_qn("rPr"))
    if rpr is None:
        from lxml import etree as _et
        rpr = _et.SubElement(ppr, _qn("rPr"))

    half_pt = str(_BODY_FONT_SIZE_PT * 2)
    sz = rpr.find(_qn("sz"))
    if sz is None:
        from lxml import etree as _et
        sz = _et.SubElement(rpr, _qn("sz"))
    sz.set(_qn("val"), half_pt)

    sz_cs = rpr.find(_qn("szCs"))
    if sz_cs is None:
        from lxml import etree as _et
        sz_cs = _et.SubElement(rpr, _qn("szCs"))
    sz_cs.set(_qn("val"), half_pt)


def _collapse_trailing_paragraph(paragraph: Paragraph) -> None:
    """Shrink the bare trailing paragraph to ~zero height.

    Word refuses to remove the final paragraph in a section, but if the
    body just barely overflows the page that paragraph alone is enough
    to spawn a blank page 2. Setting its line spacing to an exact tiny
    value + its paragraph-mark font to 1pt collapses it visually without
    deleting it.
    """
    from lxml import etree as _et
    p = paragraph._p
    ppr = p.find(_qn("pPr"))
    if ppr is None:
        ppr = _et.SubElement(p, _qn("pPr"))
        p.insert(0, ppr)

    # spacing: exact 20 twips (≈1pt) line, no before/after
    spacing = ppr.find(_qn("spacing"))
    if spacing is None:
        spacing = _et.SubElement(ppr, _qn("spacing"))
    for attr in ("before", "after"):
        if _qn(attr) in spacing.attrib:
            del spacing.attrib[_qn(attr)]
    spacing.set(_qn("line"), "20")
    spacing.set(_qn("lineRule"), "exact")

    # paragraph-mark rPr → 1pt
    rpr = ppr.find(_qn("rPr"))
    if rpr is None:
        rpr = _et.SubElement(ppr, _qn("rPr"))
    sz = rpr.find(_qn("sz"))
    if sz is None:
        sz = _et.SubElement(rpr, _qn("sz"))
    sz.set(_qn("val"), "2")  # 1pt = 2 half-points
    sz_cs = rpr.find(_qn("szCs"))
    if sz_cs is None:
        sz_cs = _et.SubElement(rpr, _qn("szCs"))
    sz_cs.set(_qn("val"), "2")


def _tighten_paragraph_spacing(paragraph: Paragraph) -> None:
    """Drop the `w:after` attribute from <w:spacing> on a paragraph.

    Leaves all other spacing attrs (line, lineRule, before) untouched so the
    paragraph's vertical rhythm is preserved — we only remove the trailing
    gap below it. Idempotent; safe if no spacing element exists.
    """
    ppr = paragraph._p.find(_W_NS + "pPr")
    if ppr is None:
        return
    spacing = ppr.find(_W_NS + "spacing")
    if spacing is None:
        return
    after_attr = _W_NS + "after"
    if after_attr in spacing.attrib:
        del spacing.attrib[after_attr]


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

    # ── Spacing fix: remove trailing 12pt below the last intern bullet and
    #     the Interests line so the tailored resume stays on a single page.
    for idx in _PARAGRAPHS_TO_TIGHTEN:
        if idx < len(paragraphs):
            _tighten_paragraph_spacing(paragraphs[idx])

    # ── Bullet-glyph fix: normalize paragraph-mark font on the two bullets
    #     whose master rPr uses Noto Sans Symbols without an explicit size,
    #     which renders their bullet glyph oversize and inflates line height.
    for idx in _PARAGRAPHS_TO_NORMALIZE_MARK:
        if idx < len(paragraphs):
            _normalize_paragraph_mark_size(paragraphs[idx])

    # ── Page-2 fix: collapse the bare trailing paragraph to ~1pt so it
    #     never overflows onto a phantom second page.
    if _TRAILING_EMPTY_PARAGRAPH < len(paragraphs):
        _collapse_trailing_paragraph(paragraphs[_TRAILING_EMPTY_PARAGRAPH])

    buf = BytesIO()
    doc.save(buf)
    return buf.getvalue()
