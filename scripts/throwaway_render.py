import json
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
from markupsafe import Markup, escape
import re

# Mimic md_bold filter
_BOLD = re.compile(r"\*\*(.+?)\*\*")
def md_bold(text: str) -> Markup:
    safe = str(escape(text))
    if safe.count("**") % 2 != 0:
        return Markup(safe)
    return Markup(_BOLD.sub(r"<strong>\1</strong>", safe))

def run_throwaway_render():
    templates_dir = Path("templates")
    env = Environment(loader=FileSystemLoader(templates_dir), autoescape=True)
    env.filters["md_bold"] = md_bold

    # Hardcoded View Model based on master_resume.pdf (approximate)
    view_model = {
        "name": "Rahul Bajaj",
        "contact": ["rahul@example.com", "+91 98765 43210", "Bengaluru, India"],
        "separator": "❖",
        "section_order": ["experience", "projects", "education", "skills"],
        "experience": [
            {
                "company": "Tech Corp",
                "dates": "Jan 2022 - Present",
                "title": "Software Engineer",
                "location": "Bengaluru",
                "bullets": [
                    "**Agentic Pipeline:** Developed a multi-agent system for automated PR reviews.",
                    "**Performance:** Reduced build times by **40%** through caching optimizations."
                ]
            },
            {
                "company": "StartUp Inc",
                "dates": "Jun 2020 - Dec 2021",
                "title": "Junior Developer",
                "location": "Remote",
                "bullets": [
                    "**Feature Delivery:** Implemented the core payment gateway integration.",
                    "**Collaboration:** Worked with UX teams to refine the onboarding flow."
                ]
            }
        ],
        "projects": [
            {
                "name": "Spring Code Forger",
                "bullets": [
                    "**Agentic Codegen:** Built a multi-agent system that ingests **Swagger/OpenAPI** specs and auto-generates production-ready **Spring Boot** CRUD scaffolding.",
                    "**LLM Orchestration:** Designed an agent-per-layer pipeline using **LangChain + Claude**, cutting boilerplate authoring time by **~70%**."
                ]
            },
            {
                "name": "TestForge",
                "bullets": [
                    "**Test Automation:** Built a multi-agent workflow that generates **Selenium/Playwright** automation tests from plain-English user stories.",
                    "**Agentic Pipeline:** Coordinated planner → writer → validator agents for self-healing test generation."
                ]
            }
        ],
        "education": {
            "institution": "Top University",
            "degree": "B.Tech in Computer Science",
            "date": "2020",
            "bullets": ["GPA: 3.8/4.0", "Awarded Best Capstone Project 2020"]
        },
        "skills": {
            "Languages": ["Python", "Java", "TypeScript", "Go"],
            "Frameworks": ["React", "Spring Boot", "FastAPI", "PyTorch"],
            "Tools": ["Docker", "Kubernetes", "AWS", "Git"],
            "Concepts": ["Distributed Systems", "LLMs", "CI/CD"]
        },
        "interests": "Open Source, Agentic Workflows, Chess"
    }

    html_content = env.get_template("resume.html.j2").render(**view_model)
    
    # We need to make sure we can find the CSS and fonts.
    # WeasyPrint uses base_url for relative paths.
    HTML(string=html_content, base_url=str(templates_dir)).write_pdf("templates/throwaway_render.pdf")
    print("Rendered to templates/throwaway_render.pdf")

if __name__ == "__main__":
    run_throwaway_render()
