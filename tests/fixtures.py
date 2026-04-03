"""
Mock outputs for every agent in the pipeline.
Each fixture returns the same Pydantic model the real agent would.
Update these to match your actual resume / expected test data.
"""

from agents.agent_0a_profiler import CandidateProfile
from agents.agent_0b_scraper import JobListing
from agents.agent_1 import TailoredAssets


def mock_agent_0a() -> CandidateProfile:
    """Mock profiler output — edit to match your resume."""
    return CandidateProfile(
        full_name="Ankur Bajaj",
        current_title="Software Engineer",
        total_yoe=2.0,
        core_skills=["Python", "React", "Node.js", "AWS", "TypeScript"],
        frameworks=["FastAPI", "Next.js", "Django", "Express"],
        domains=["HealthTech", "IoT"],
        seniority="junior",
        companies=[],
        education="B.Tech Computer Science",
        preferred_locations=["Bengaluru", "Remote"],
        company_type_preference="product",
        max_yoe_applying_for=3,
        search_keywords=[
            "Software Engineer",
            "Backend Developer",
            "Full Stack Developer",
            "React Developer",
            "Python Developer",
            "SDE",
        ],
        raw_summary=(
            "Junior software engineer with 2 years of experience building "
            "full-stack applications using Python, React, and AWS. "
            "Comfortable with backend services, REST APIs, and cloud deployments."
        ),
    )


def mock_agent_0b() -> list[JobListing]:
    """Mock scraper output — a handful of realistic jobs for testing downstream agents."""
    return [
        JobListing(
            job_id="abc123",
            title="Software Engineer",
            company="Razorpay",
            location="Bengaluru, Karnataka, India",
            description=(
                "We are looking for a Software Engineer with experience in "
                "Python, React, and AWS. You will build scalable microservices "
                "and contribute to our payments platform. 1-3 years experience required."
            ),
            date_posted="2 days ago",
            job_url="https://example.com/job1",
        ),
        JobListing(
            job_id="def456",
            title="Full Stack Developer",
            company="Zerodha",
            location="Bengaluru, Karnataka, India",
            description=(
                "Join our engineering team to build trading platforms using "
                "React, TypeScript, Node.js and PostgreSQL. Experience with "
                "REST APIs and cloud infrastructure preferred. 0-2 years."
            ),
            date_posted="1 day ago",
            job_url="https://example.com/job2",
        ),
        JobListing(
            job_id="ghi789",
            title="Backend Developer",
            company="Postman",
            location="Bengaluru, Karnataka, India (Hybrid)",
            description=(
                "Build and maintain backend services using Python, FastAPI, "
                "and AWS. Strong understanding of distributed systems and "
                "API design. 1-3 years of experience."
            ),
            date_posted="3 days ago",
            job_url="https://example.com/job3",
        ),
    ]


def mock_agent_0c(jobs: list[JobListing]) -> list[tuple[JobListing, int]]:
    """Mock ranker — passes all jobs through with score 75."""
    return [(job, 75) for job in jobs]


def mock_agent_1(job: JobListing) -> TailoredAssets:
    """Mock tailor — returns plausible placeholder assets."""
    return TailoredAssets(
        resume_bullets=[
            "Built scalable REST APIs serving 10K+ requests/day using Python and FastAPI",
            "Developed responsive frontend dashboards with React and TypeScript",
            "Deployed microservices on AWS ECS with CI/CD pipelines",
            "Reduced API latency by 40% through query optimization and caching",
            "Implemented authentication and authorization using OAuth2 and JWT",
            "Collaborated with cross-functional teams to ship features biweekly",
        ],
        cover_letter=(
            "I am excited to apply for this role at your company. "
            "My experience building full-stack applications with Python and React "
            "aligns well with your requirements.\n\n"
            "In my current role, I have built scalable backend services and "
            "responsive frontends. I am comfortable with AWS, CI/CD, and "
            "agile development practices.\n\n"
            "I look forward to discussing how I can contribute to your team."
        ),
        form_answers={
            "describe_last_role": "Built backend services and React frontends.",
            "why_this_company": "Strong engineering culture and impactful product.",
            "biggest_achievement": "Reduced API latency by 40% through caching.",
            "notice_period": "Immediate to 30 days",
            "expected_ctc": "Open to discussion",
        },
        job_title_used=job.title,
        company_name_used=job.company,
    )


def mock_agent_2(job: JobListing) -> dict[str, str]:
    """Mock publisher — returns fake S3 URLs."""
    return {
        "resume": f"s3://mock-bucket/2026-01-01/{job.company}_{job.title}/resume.pdf",
        "cover_letter": f"s3://mock-bucket/2026-01-01/{job.company}_{job.title}/cover_letter.pdf",
        "form_answers": f"s3://mock-bucket/2026-01-01/{job.company}_{job.title}/form_answers.json",
    }
