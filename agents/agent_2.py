import os
import json
import logging
import boto3
from datetime import datetime
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from dotenv import load_dotenv
from agents.agent_0a_profiler import CandidateProfile
from agents.agent_0b_scraper import JobListing
from agents.agent_1 import TailoredAssets

load_dotenv()
logger = logging.getLogger(__name__)

# ── PDF colour palette ────────────────────────────────────────────────────────
_DARK = colors.HexColor("#1a1a1a")
_MID  = colors.HexColor("#555555")
_RULE = colors.HexColor("#cccccc")


def _sanitize_filename(text: str) -> str:
    """Converts 'GE HealthCare' → 'GEHealthCare' for safe S3 keys."""
    return "".join(c for c in text if c.isalnum() or c in "-_")


def _escape(text: str) -> str:
    """Escape ReportLab special characters."""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;"))


def _build_resume_pdf(assets: TailoredAssets, profile: CandidateProfile) -> bytes:
    """
    Renders a complete, ATS-friendly resume PDF from structured TailoredAssets.
    Sections: Header, Summary, Experience, Skills, Projects, Education.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
    )

    # ── Styles ────────────────────────────────────────────────────────────────
    name_style = ParagraphStyle(
        "Name",
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=_DARK,
        alignment=TA_CENTER,
        spaceAfter=2,
    )
    contact_style = ParagraphStyle(
        "Contact",
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=_MID,
        alignment=TA_CENTER,
        spaceAfter=4,
    )
    section_style = ParagraphStyle(
        "Section",
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=_DARK,
        spaceBefore=10,
        spaceAfter=3,
    )
    role_title_style = ParagraphStyle(
        "RoleTitle",
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=_DARK,
        spaceBefore=6,
        spaceAfter=1,
    )
    role_meta_style = ParagraphStyle(
        "RoleMeta",
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=12,
        textColor=_MID,
        spaceAfter=2,
    )
    body_style = ParagraphStyle(
        "Body",
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=_DARK,
        alignment=TA_JUSTIFY,
        spaceAfter=4,
    )
    bullet_style = ParagraphStyle(
        "Bullet",
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=_DARK,
        leftIndent=12,
        spaceAfter=2,
    )
    skill_style = ParagraphStyle(
        "Skill",
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=_DARK,
        spaceAfter=3,
    )

    story = []

    # ── Header ────────────────────────────────────────────────────────────────
    story.append(Paragraph(_escape(profile.full_name), name_style))

    contact_parts = []
    if profile.location_city:
        contact_parts.append(_escape(profile.location_city))
    if profile.email:
        contact_parts.append(_escape(profile.email))
    if profile.phone:
        contact_parts.append(_escape(profile.phone))
    if profile.linkedin:
        contact_parts.append(_escape(profile.linkedin))
    if profile.github:
        contact_parts.append(_escape(profile.github))

    if contact_parts:
        story.append(Paragraph("  |  ".join(contact_parts), contact_style))

    story.append(HRFlowable(width="100%", thickness=1, color=_RULE, spaceAfter=6))

    # ── Summary ───────────────────────────────────────────────────────────────
    if assets.summary:
        story.append(Paragraph("SUMMARY", section_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=_RULE, spaceAfter=4))
        story.append(Paragraph(_escape(assets.summary), body_style))

    # ── Experience ────────────────────────────────────────────────────────────
    if assets.experience:
        story.append(Paragraph("EXPERIENCE", section_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=_RULE, spaceAfter=4))

        for exp in assets.experience:
            # "Software Engineer  |  Razorpay"
            header_text = f"<b>{_escape(exp.title)}</b>  |  {_escape(exp.company)}"
            story.append(Paragraph(header_text, role_title_style))

            # "Jan 2023 – Present  |  Bengaluru, India"
            meta_parts = [_escape(exp.dates)]
            if exp.location:
                meta_parts.append(_escape(exp.location))
            story.append(Paragraph("  |  ".join(meta_parts), role_meta_style))

            for bullet in exp.bullets:
                story.append(Paragraph(f"• {_escape(bullet)}", bullet_style))

    # ── Skills ────────────────────────────────────────────────────────────────
    if assets.skills:
        story.append(Paragraph("SKILLS", section_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=_RULE, spaceAfter=4))

        for category, skill_list in assets.skills.items():
            line = f"<b>{_escape(category)}:</b>  {_escape(', '.join(skill_list))}"
            story.append(Paragraph(line, skill_style))

    # ── Projects ──────────────────────────────────────────────────────────────
    if assets.projects:
        story.append(Paragraph("PROJECTS", section_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=_RULE, spaceAfter=4))

        for proj in assets.projects:
            header_parts = [f"<b>{_escape(proj.name)}</b>"]
            if proj.tech_stack:
                header_parts.append(f"<i>{_escape(proj.tech_stack)}</i>")
            story.append(Paragraph("  |  ".join(header_parts), role_title_style))

            for bullet in proj.bullets:
                story.append(Paragraph(f"• {_escape(bullet)}", bullet_style))

    # ── Education ─────────────────────────────────────────────────────────────
    if assets.education:
        story.append(Paragraph("EDUCATION", section_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=_RULE, spaceAfter=4))
        story.append(Paragraph(_escape(assets.education), body_style))

    doc.build(story)
    return buffer.getvalue()


def _build_cover_letter_pdf(assets: TailoredAssets, profile: CandidateProfile) -> bytes:
    """Generates a professional cover letter PDF."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=25 * mm,
        leftMargin=25 * mm,
        topMargin=25 * mm,
        bottomMargin=25 * mm,
    )

    styles = getSampleStyleSheet()
    header_style = ParagraphStyle(
        "Header", parent=styles["Normal"],
        fontSize=11, spaceAfter=16,
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=10.5, leading=16, spaceAfter=12,
        alignment=TA_JUSTIFY,
    )

    today = datetime.now().strftime("%B %d, %Y")
    story = []
    story.append(Paragraph(f"{_escape(profile.full_name)}<br/>{today}", header_style))
    story.append(Paragraph(
        f"Hiring Manager<br/>{_escape(assets.company_name_used)}", header_style
    ))
    story.append(Spacer(1, 4 * mm))

    for para in assets.cover_letter.split("\n\n"):
        if para.strip():
            story.append(Paragraph(_escape(para.strip()), body_style))

    doc.build(story)
    return buffer.getvalue()


def publish_to_s3(
    job: JobListing,
    assets: TailoredAssets,
    profile: CandidateProfile,
    score: int = 0,
) -> dict[str, str]:
    """
    Uploads 4 files to S3. Returns dict of {asset_type: s3_key}.
    S3 key structure: YYYY-MM-DD/CompanyName_RoleTitle/filename
    """
    bucket_name = os.getenv("AWS_S3_BUCKET")
    if not bucket_name:
        raise EnvironmentError("AWS_S3_BUCKET not set")

    s3 = boto3.client("s3")
    date_prefix = datetime.now().strftime("%Y-%m-%d")
    company_clean = _sanitize_filename(job.company)
    title_clean = _sanitize_filename(job.title)
    folder = f"{date_prefix}/{company_clean}_{title_clean}"

    uploads = {}

    # 1. Resume PDF
    resume_bytes = _build_resume_pdf(assets, profile)
    resume_key = f"{folder}/resume.pdf"
    s3.put_object(
        Bucket=bucket_name,
        Key=resume_key,
        Body=resume_bytes,
        ContentType="application/pdf",
    )
    uploads["resume"] = resume_key
    logger.info(f"Uploaded resume: {resume_key}")

    # 2. Cover letter PDF
    cover_bytes = _build_cover_letter_pdf(assets, profile)
    cover_key = f"{folder}/cover_letter.pdf"
    s3.put_object(
        Bucket=bucket_name,
        Key=cover_key,
        Body=cover_bytes,
        ContentType="application/pdf",
    )
    uploads["cover_letter"] = cover_key
    logger.info(f"Uploaded cover letter: {cover_key}")

    # 3. Form answers JSON
    form_key = f"{folder}/form_answers.json"
    s3.put_object(
        Bucket=bucket_name,
        Key=form_key,
        Body=json.dumps(assets.form_answers, indent=2),
        ContentType="application/json",
    )
    uploads["form_answers"] = form_key
    logger.info(f"Uploaded form answers: {form_key}")

    # 4. Job metadata — id, apply link, score for easy tracking
    job_info = {
        "job_id": job.job_id,
        "title": job.title,
        "company": job.company,
        "location": job.location,
        "score": score,
        "apply_url": job.job_url,
        "date_posted": job.date_posted,
        "date_processed": datetime.now().strftime("%Y-%m-%d"),
    }
    info_key = f"{folder}/job_info.json"
    s3.put_object(
        Bucket=bucket_name,
        Key=info_key,
        Body=json.dumps(job_info, indent=2),
        ContentType="application/json",
    )
    uploads["job_info"] = info_key
    logger.info(f"Uploaded job info: {info_key}")

    return uploads
