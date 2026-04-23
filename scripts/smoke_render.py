"""Smoke test: render a hand-built TailoredAssets through docx renderer + PDF converter.

Produces scripts/smoke_resume.docx and scripts/smoke_resume.pdf so we can
eyeball the output before touching S3.

Inline assets (no tests.fixtures import) so this runs without pandas/jobspy
which are only needed by the scraper path.

Run: python scripts/smoke_render.py
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")

from agents.agent_1 import TailoredAssets, ExperienceEntry, EducationEntry
from agents.docx_renderer import render_tailored_docx
from agents.pdf_converter import convert_docx_to_pdf

MASTER = Path("data/master_resume.docx")
OUT_DOCX = Path("scripts/smoke_resume.docx")
OUT_PDF = Path("scripts/smoke_resume.pdf")

assets = TailoredAssets(
    experience=[
        ExperienceEntry(
            company="Philips",
            title="Software Development Engineer",
            dates="Aug. 2023 – Present",
            location="Bangalore, India",
            bullets=[
                "**Frontend Platform:** Directed a 4-member team to architect an enterprise launchpad using **React**, **TypeScript**, and **AWS Serverless**, increasing deployment frequency by **35%** for **10k+** daily users.",
                "**MFE Architecture:** Engineered MFE solutions using **Module Federation** and **Vite**, slashing session-related errors by **40%** while improving page load speeds by **1.2s**.",
                "**UI Design Systems:** Designed dynamic forms and workflow-based UI components; improved component reusability by **60%**, saving **150+** engineering hours per quarter.",
                "**Cloud Infrastructure:** Managed **Java Spring Boot** microservices on **AWS EKS** for IoT medical device connectivity, maintaining **99.9%** uptime and supporting **50k+** concurrent connections.",
                "**Data Integration:** Developed a **Python**-based serverless data lake backend and a **React** analytics dashboard that reduced manual data entry by **80%**.",
                "**System Architecture:** Implemented **Domain-Driven Design (DDD)** and **Design Patterns (Strategy, Proxy, Facade)**, resulting in a **25%** reduction in technical debt.",
            ],
        ),
        ExperienceEntry(
            company="Philips",
            title="SDE Intern",
            dates="Feb. 2023 – July 2023",
            location="Bangalore, India",
            bullets=[
                "**Cloud Automation:** Built a secure **AWS Serverless** application using **Python (Boto3)**, **API Gateway**, and **PostgreSQL/DynamoDB**, reducing processing time from **3 days** to **15 minutes**.",
                "**Performance Tuning:** Optimized **PostgreSQL** queries and indexing, improving retrieval performance by **55%** and eliminating timeouts during peak hours.",
                "**Agile Collaboration:** Participated in an **Agile SDLC (Scrum)** environment, delivering **100%** of sprint commitments while refining technical requirements.",
            ],
        ),
    ],
    skills={
        "Languages & Backend": ["TypeScript", "Java", "Python", "SQL", "Spring Boot", "REST APIs", "Microservices"],
        "Frontend & Architecture": ["React.js", "Micro Frontends (MFE)", "Module Federation", "Vite", "Next.js"],
        "Cloud & DevOps": ["AWS (Lambda, S3, DynamoDB, RDS)", "EKS", "CI/CD", "Infrastructure as Code (IaC)"],
        "Testing & Design": ["Design Patterns", "Domain-Driven Design (DDD)", "Karate", "Jest", "Unit Testing"],
    },
    interests="Badminton (University Team), Weightlifting, Adventure Sports, Reading",
    education=EducationEntry(
        institution="Chandigarh Group of Colleges",
        degree="B.Tech in Computer Science & Engineering",
        date="July 2023",
        bullets=[
            "**Leadership:** Captain of the University Badminton Team (led to multiple tournament wins).",
            "**Management:** Finance head for flagship festival (**500+** delegates); Lead Anchor for university events.",
        ],
    ),
    form_answers={
        "describe_last_role": "SDE at Philips.",
        "describe_second_last_role": "SDE Intern at Philips.",
        "why_this_company": "Strong engineering culture.",
        "biggest_achievement": "Cut errors 40% via MFE.",
        "notice_period": "30 days",
        "expected_ctc": "Open",
    },
    job_title_used="Senior Software Engineer — Frontend Platform",
    company_name_used="Swiggy",
)

print(f"Rendering tailored DOCX for: {assets.job_title_used} @ {assets.company_name_used}")
docx_bytes = render_tailored_docx(assets, MASTER)
OUT_DOCX.write_bytes(docx_bytes)
print(f"  wrote {OUT_DOCX} ({len(docx_bytes):,} bytes)")

print("Converting to PDF...")
try:
    pdf_bytes = convert_docx_to_pdf(docx_bytes)
    OUT_PDF.write_bytes(pdf_bytes)
    print(f"  wrote {OUT_PDF} ({len(pdf_bytes):,} bytes)")
except Exception as e:
    print(f"  PDF conversion FAILED: {e}")
    print(f"  DOCX is still at {OUT_DOCX} — open it in Word to verify content.")
    sys.exit(1)

print("\nSmoke test PASSED — open the DOCX and PDF to eyeball formatting.")
