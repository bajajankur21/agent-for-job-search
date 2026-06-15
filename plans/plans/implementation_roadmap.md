# Implementation Roadmap: Commercial Job Hunt Agent

This document outlines the strict priority sequence for the commercialization of the platform. Each phase must be validated before moving to the next.

## Priority 1: Core Agent Agnosticism & Dynamics (The Intelligence)
**Goal:** Remove all hard-coded assumptions about roles, locations, and skills.
- **Agent 0A (Profiler):** Make it support any resume format. Generate search terms dynamically via LLM.
- **Agent 0B/0C (Scraper/Ranker):** Remove `NON_SDE_TITLE_KEYWORDS` and hard-coded location filters. Implement the "Composite Profile" logic (Resume + User Preferences).
- **Requirement:** The pipeline must produce high-quality results for a non-SDE role (e.g., Product Manager) using only a resume and minimal preferences.

## Priority 2: Resume Tailoring Agnosticism (The Fidelity)
**Goal:** Professional PDF output via a Dynamic HTML-to-PDF Engine.
- **HTML/CSS Template System:** Implement a high-fidelity rendering pipeline using Jinja2 and Playwright (Headless Chromium) to generate pixel-perfect PDFs from designer-approved templates.
- **Dynamic Density Logic:** Implement a "Density Profile" system that adjusts line-heights, margins, and pagination rules based on the candidate's YOE (e.g., Compact for < 5yrs, Expansive for 10+ yrs) to optimize page count.
- **Requirement:** Successful render of a tailored resume using a dynamic HTML template that automatically scales layout and pagination based on candidate experience without manual tagging.

## Priority 3: Orchestration Layer (The API & Queue)
**Goal:** Transition from a linear script to a scalable service.
- **FastAPI Wrapper:** Create endpoints for resume upload, preference setting, and pipeline triggering.
- **Celery/Redis Queue:** Implement asynchronous task processing to avoid HTTP timeouts.
- **State Management:** Move from S3 JSON to a PostgreSQL database.

## Priority 4: User Interface & Experience (The Frontend)
**Goal:** A professional way for users to interact with the AI.
- **Onboarding Flow:** Resume upload $ightarrow$ Profile Review $ightarrow$ Preference Setting.
- **Pipeline Tracker:** Real-time status updates for the background tasks.
- **Result Delivery:** A clean dashboard to view and download tailored assets.

## Priority 5: Delivery & Distribution
**Goal:** Make the results easy to use and access.
- **Notification System:** Email or Webhook notifications when the pipeline is complete.
- **S3 Delivery:** Secure, temporary signed URLs for PDF downloads.

## Priority 6: DevOps & DevSecOps (The Stability)
**Goal:** Secure, free, and stable deployment.
- **Zero-Budget Deploy:** Host on Hugging Face Spaces (Backend) and GitHub Pages (Frontend).
- **Multi-tenant Isolation:** Final audit of the `user_id` boundaries in DB and S3.
- **BYOK Implementation:** Secure handling of user-provided AI Studio keys.
