"""
Mock outputs for every agent in the pipeline.
Each fixture returns the same Pydantic model the real agent would.
Update these to match your actual resume / expected test data.
"""

from agents.agent_0a_profiler import CandidateProfile
from agents.agent_0b_scraper import JobListing
from agents.agent_1 import TailoredAssets, ExperienceEntry, ProjectEntry


def mock_agent_0a() -> CandidateProfile:
    """Mock profiler output — edit to match your resume."""
    return CandidateProfile(
        full_name="Rahul Bajaj",
        current_title="Software Engineer",
        total_yoe=1.5,
        core_skills=["Python", "React", "Node.js", "AWS", "TypeScript"],
        frameworks=["FastAPI", "Next.js", "Django", "Express"],
        domains=["FinTech", "IoT"],
        seniority="junior",
        companies=["Infosys BPM", "Freelance"],
        education="B.Tech Computer Science | BITS Pilani | 2023",
        email="rahul.bajaj@email.com",
        phone="+91 98765 43210",
        linkedin="linkedin.com/in/rahulbajaj",
        github="github.com/bajajankur21",
        location_city="Bengaluru, India",
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
            "Junior software engineer with 1.5 years of experience building "
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
    """Mock tailor — returns plausible placeholder assets matching the full resume structure."""
    return TailoredAssets(
        summary=(
            f"Junior software engineer with 1.5 years of experience building backend services "
            f"and React frontends, applying for {job.title} at {job.company}. "
            "Proven track record delivering scalable REST APIs and cloud-deployed microservices."
        ),
        experience=[
            ExperienceEntry(
                company="Infosys BPM",
                title="Software Engineer",
                dates="Jul 2023 – Present",
                location="Bengaluru, India",
                bullets=[
                    "Built scalable REST APIs serving 10K+ requests/day using Python and FastAPI",
                    "Developed responsive frontend dashboards with React and TypeScript",
                    "Deployed microservices on AWS ECS with CI/CD pipelines reducing release time by 30%",
                    "Reduced API latency by 40% through query optimisation and Redis caching",
                ],
            ),
            ExperienceEntry(
                company="Freelance",
                title="Full Stack Developer",
                dates="Jan 2023 – Jun 2023",
                location="Remote",
                bullets=[
                    "Delivered 3 client web apps using React and Node.js within tight deadlines",
                    "Implemented OAuth2 authentication and role-based access control",
                ],
            ),
        ],
        skills={
            "Languages": ["Python", "TypeScript", "Java", "SQL"],
            "Frameworks": ["React", "FastAPI", "Node.js", "Express"],
            "Tools & Cloud": ["AWS (ECS, S3, Lambda)", "Docker", "Git", "PostgreSQL"],
        },
        projects=[
            ProjectEntry(
                name="Job Application Automation Agent",
                tech_stack="Python, Claude AI, Gemini, JobSpy, AWS S3",
                bullets=[
                    "Built a multi-agent pipeline that scrapes, ranks, and tailors job applications autonomously",
                    "Reduced manual application time from 2 hours to zero per day",
                ],
            ),
        ],
        education="B.Tech Computer Science | BITS Pilani | 2023",
        form_answers={
            "describe_last_role": "Built backend APIs and React frontends at Infosys BPM.",
            "describe_second_last_role": "Delivered full-stack client projects as a freelance developer.",
            "why_this_company": f"Strong engineering culture and impactful product at {job.company}.",
            "biggest_achievement": "Reduced API latency by 40% through Redis caching and query optimisation.",
            "notice_period": "Immediate to 30 days",
            "expected_ctc": "Open to discussion",
        },
        job_title_used=job.title,
        company_name_used=job.company,
    )


def mock_agent_2(job: JobListing) -> dict[str, str]:
    """Mock publisher — returns fake S3 keys."""
    return {
        "resume": f"2026-01-01/{job.company}_{job.title}/resume.pdf",
        "form_answers": f"2026-01-01/{job.company}_{job.title}/form_answers.json",
        "job_info": f"2026-01-01/{job.company}_{job.title}/job_info.json",
    }
