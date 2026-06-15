import logging
import asyncio
from pathlib import Path
from io import BytesIO
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape
from playwright.async_api import async_playwright

from agents.agent_0a_profiler import CandidateProfile
from agents.agent_1 import TailoredAssets

logger = logging.getLogger(__name__)

# ── Density Profiles ─────────────────────────────────────────────────────────
# Adjusts the visual density of the resume based on the candidate's experience.
DENSITY_PROFILES = {
    "compact": {
        "line_height": "1.2",
        "margin_top": "30px",
        "margin_bottom": "30px",
        "margin_side": "40px",
        "font_size_base": "10pt",
    },
    "balanced": {
        "line_height": "1.4",
        "margin_top": "40px",
        "margin_bottom": "40px",
        "margin_side": "50px",
        "font_size_base": "11pt",
    },
    "expansive": {
        "line_height": "1.6",
        "margin_top": "50px",
        "margin_bottom": "50px",
        "margin_side": "60px",
        "font_size_base": "12pt",
    },
}

def _get_density_profile(yoe: float) -> str:
    if yoe < 5:
        return "compact"
    elif yoe < 10:
        return "balanced"
    else:
        return "expansive"


class PDFEngine:
    def __init__(self, template_path: str = "agents/resume_template.html"):
        self.template_path = template_path
        # Setup Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(Path(template_path).parent),
            autoescape=select_autoescape(['html', 'xml']),
        )

    async def render(self, assets: TailoredAssets, profile: CandidateProfile) -> bytes:
        """Renders TailoredAssets and CandidateProfile into a professional PDF."""
        template_name = Path(self.template_path).name
        template = self.env.get_template(template_name)
        
        # Determine density profile based on YOE
        density_key = _get_density_profile(profile.total_yoe)
        density = DENSITY_PROFILES[density_key]

        # Render HTML with data and density variables
        html_content = template.render(
            assets=assets,
            profile=profile,
            density=density
        )

        # Use Playwright to print HTML to PDF
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            # Set content and wait for network idle to ensure fonts/styles load
            await page.set_content(html_content, wait_until="networkidle")
            
            # PDF generation with professional settings
            pdf_bytes = await page.pdf(
                format="A4",
                print_background=True,
                margin={"top": "0px", "right": "0px", "bottom": "0px", "left": "0px"}, # Margins handled by CSS
            )
            
            await browser.close()
            return pdf_bytes

def render_tailored_pdf(assets: TailoredAssets, profile: CandidateProfile) -> bytes:
    """Synchronous wrapper for the async PDF engine."""
    engine = PDFEngine()
    return asyncio.run(engine.render(assets, profile))
