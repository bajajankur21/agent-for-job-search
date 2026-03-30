import os
import json
import logging
import boto3
from datetime import datetime
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY
from dotenv import load_dotenv
from agents.agent_0b_scraper import JobListing
from agents.agent_2 import TailoredAssets

load_dotenv()
logger = logging.getLogger(__name__)


def _sanitize_filename(text: str) -> str:
    """Converts 'GE HealthCare' → 'GEHealthCare' for safe S3 keys."""
    return "".join(c for c in text if c.isalnum() or c in "-_")


def _build_resume_pdf(assets: TailoredAssets, profile_name: str) -> bytes:
    """Generates a clean resume PDF with tailored bullet points."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=20*mm,
        leftMargin=20*mm,
        topMargin=20*mm,
        bottomMargin=20*mm
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title", parent=styles["Heading1"],
        fontSize=16, spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        "Subtitle", parent=styles["Normal"],
        fontSize=11, spaceAfter=12, textColor=(0.4, 0.4, 0.4)
    )
    bullet_style = ParagraphStyle(
        "Bullet", parent=styles["Normal"],
        fontSize=10.5, spaceAfter=6, leading=14,
        leftIndent=12, alignment=TA_LEFT
    )

    story = []
    story.append(Paragraph(profile_name, title_style))
    story.append(Paragraph(
        f"Tailored for: {assets.job_title_used} at {assets.company_name_used}",
        subtitle_style
    ))
    story.append(Spacer(1, 6*mm))
    story.append(Paragraph("Relevant Experience Highlights", styles["Heading2"]))
    story.append(Spacer(1, 3*mm))

    for bullet in assets.resume_bullets:
        story.append(Paragraph(f"• {bullet}", bullet_style))

    doc.build(story)
    return buffer.getvalue()


def _build_cover_letter_pdf(assets: TailoredAssets, profile_name: str) -> bytes:
    """Generates a professional cover letter PDF."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=25*mm,
        leftMargin=25*mm,
        topMargin=25*mm,
        bottomMargin=25*mm
    )

    styles = getSampleStyleSheet()
    header_style = ParagraphStyle(
        "Header", parent=styles["Normal"],
        fontSize=11, spaceAfter=16
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=10.5, leading=16, spaceAfter=12,
        alignment=TA_JUSTIFY
    )

    today = datetime.now().strftime("%B %d, %Y")
    story = []
    story.append(Paragraph(f"{profile_name}<br/>{today}", header_style))
    story.append(Paragraph(
        f"Hiring Manager<br/>{assets.company_name_used}", header_style
    ))
    story.append(Spacer(1, 4*mm))

    for para in assets.cover_letter.split("\n\n"):
        if para.strip():
            story.append(Paragraph(para.strip(), body_style))

    doc.build(story)
    return buffer.getvalue()


def publish_to_s3(
    job: JobListing,
    assets: TailoredAssets,
    profile_name: str
) -> dict[str, str]:
    """
    Uploads 3 files to S3. Returns dict of {asset_type: s3_url}.
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
    resume_bytes = _build_resume_pdf(assets, profile_name)
    resume_key = f"{folder}/resume.pdf"
    s3.put_object(
        Bucket=bucket_name,
        Key=resume_key,
        Body=resume_bytes,
        ContentType="application/pdf"
    )
    uploads["resume"] = f"s3://{bucket_name}/{resume_key}"
    logger.info(f"Uploaded resume: {resume_key}")

    # 2. Cover letter PDF
    cover_bytes = _build_cover_letter_pdf(assets, profile_name)
    cover_key = f"{folder}/cover_letter.pdf"
    s3.put_object(
        Bucket=bucket_name,
        Key=cover_key,
        Body=cover_bytes,
        ContentType="application/pdf"
    )
    uploads["cover_letter"] = f"s3://{bucket_name}/{cover_key}"
    logger.info(f"Uploaded cover letter: {cover_key}")

    # 3. Form answers JSON
    form_key = f"{folder}/form_answers.json"
    s3.put_object(
        Bucket=bucket_name,
        Key=form_key,
        Body=json.dumps(assets.form_answers, indent=2),
        ContentType="application/json"
    )
    uploads["form_answers"] = f"s3://{bucket_name}/{form_key}"
    logger.info(f"Uploaded form answers: {form_key}")

    return uploads