"""
Mock outputs for every agent in the pipeline.
Each fixture returns the same Pydantic model the real agent would.
Update these to match your actual resume / expected test data.
"""

from agents.agent_0a_profiler import CandidateProfile
from agents.agent_0b_scraper import JobListing
from agents.agent_1 import TailoredAssets, ExperienceEntry, EducationEntry


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
    """Mock tailor — returns plausible placeholder assets matching the master resume structure."""
    return TailoredAssets(
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
                    "**Cloud Automation:** Built a secure **AWS Serverless** application using **Python (Boto3)**, **API Gateway**, and **PostgreSQL/DynamoDB**, reducing manual processing time from **3 days** to **15 minutes**.",
                    "**Performance Tuning:** Optimized **PostgreSQL** queries and indexing, improving retrieval performance by **55%** and eliminating timeouts during peak hours.",
                    "**Agile Collaboration:** Participated in an **Agile SDLC (Scrum)** environment, delivering **100%** of sprint commitments while refining technical requirements.",
                ],
            ),
        ],
        skills={
            "Languages & Backend": ["Java", "Python", "SQL", "TypeScript", "Spring Boot", "REST APIs", "Microservices"],
            "Frontend & Architecture": ["React.js", "Micro Frontends (MFE)", "Module Federation", "Vite", "Next.js"],
            "Cloud & DevOps": ["AWS (Lambda, S3, DynamoDB)", "EKS", "CI/CD", "Infrastructure as Code (IaC)"],
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
            "describe_last_role": "Software Development Engineer at Philips focused on frontend platform architecture using Module Federation and AWS Serverless.",
            "describe_second_last_role": "SDE Intern at Philips building AWS Serverless automation and optimizing PostgreSQL performance.",
            "why_this_company": f"Strong engineering culture and impactful product at {job.company}.",
            "biggest_achievement": "Reduced session-related errors by 40% via Micro Frontend architecture using Module Federation.",
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
