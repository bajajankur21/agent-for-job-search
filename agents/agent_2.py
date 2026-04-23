import os
import json
import logging
from pathlib import Path
from datetime import datetime

import boto3
from dotenv import load_dotenv

from agents.agent_0a_profiler import CandidateProfile
from agents.agent_0b_scraper import JobListing
from agents.agent_1 import TailoredAssets
from agents.docx_renderer import render_tailored_docx
from agents.pdf_converter import convert_docx_to_pdf

load_dotenv()
logger = logging.getLogger(__name__)

_MASTER_DOCX_PATH = Path("data/master_resume.docx")


def _sanitize_filename(text: str) -> str:
    """Converts 'GE HealthCare' → 'GEHealthCare' for safe S3 keys."""
    return "".join(c for c in text if c.isalnum() or c in "-_")


def publish_to_s3(
    job: JobListing,
    assets: TailoredAssets,
    profile: CandidateProfile,
    score: int = 0,
) -> dict[str, str]:
    """
    Uploads 3 files to S3. Returns dict of {asset_type: s3_key}.
    S3 key structure: YYYY-MM-DD/CompanyName_RoleTitle/filename

    Resume pipeline: patch master DOCX → LibreOffice/docx2pdf → upload PDF.
    The CandidateProfile arg is retained for API compatibility with main.py
    even though the renderer reads identity info directly from the master DOCX.
    """
    del profile  # unused — master DOCX already carries the header/contact info

    bucket_name = os.getenv("AWS_S3_BUCKET")
    if not bucket_name:
        raise EnvironmentError("AWS_S3_BUCKET not set")
    if not _MASTER_DOCX_PATH.exists():
        raise FileNotFoundError(
            f"Master DOCX not found at {_MASTER_DOCX_PATH} — required for rendering"
        )

    s3 = boto3.client("s3")
    date_prefix = datetime.now().strftime("%Y-%m-%d")
    company_clean = _sanitize_filename(job.company)
    title_clean = _sanitize_filename(job.title)
    folder = f"{date_prefix}/{company_clean}_{title_clean}"

    uploads = {}

    # 1. Resume PDF — patch master.docx in-memory, then convert to PDF.
    docx_bytes = render_tailored_docx(assets, _MASTER_DOCX_PATH)
    resume_bytes = convert_docx_to_pdf(docx_bytes)
    resume_key = f"{folder}/resume.pdf"
    s3.put_object(
        Bucket=bucket_name,
        Key=resume_key,
        Body=resume_bytes,
        ContentType="application/pdf",
    )
    uploads["resume"] = resume_key
    logger.info(f"Uploaded resume: {resume_key}")

    # 2. Form answers JSON
    form_key = f"{folder}/form_answers.json"
    s3.put_object(
        Bucket=bucket_name,
        Key=form_key,
        Body=json.dumps(assets.form_answers, indent=2),
        ContentType="application/json",
    )
    uploads["form_answers"] = form_key
    logger.info(f"Uploaded form answers: {form_key}")

    # 3. Job metadata — id, apply link, score for easy tracking
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
