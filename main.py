import os
from agent_1 import run_gatekeeper
from agent_2 import run_tailor

def main():
    # Placeholder for the next task: Scraper
    job_desc = "Junior React Developer at a startup. 1 year experience required."
    my_resume = "Full Stack Engineer at Philips, 2.5 years experience..."

    print("🤖 Starting the AI Agent Pipeline...")
    
    # Run Agent 1
    validation = run_gatekeeper(job_desc)
    
    if validation.is_fit:
        # Run Agent 2
        tailored_text = run_tailor(job_desc, my_resume)
        print("✅ Tailoring Complete!")
        print(tailored_text)
    else:
        print(f"🛑 Job rejected: {validation.reasoning}")

if __name__ == "__main__":
    main()