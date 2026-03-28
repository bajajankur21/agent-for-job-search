import os
import sys
import google.generativeai as genai
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# 1. Load environment variables for local testing
load_dotenv()

# 2. Configure the Gatekeeper
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class JobValidation(BaseModel):
    years_required: int = Field(description="The minimum years of experience required.")
    is_fit: bool = Field(description="True if years_required <= 3, else False.")
    reasoning: str = Field(description="Short explanation of the YOE found.")

def run_gatekeeper(job_description: str):
    """
    Agent 1: Uses Gemini 1.5 Flash to extract YOE.
    Flash is extremely fast and free/cheap for this extraction task.
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    Analyze this job description. Extract the minimum years of experience required.
    If no YOE is mentioned, assume 0.
    
    Job Description: {job_description}
    
    Return the data in strict JSON format.
    """

    # We use Flash's ability to generate structured output
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            response_mime_type="application/json",
            response_schema=JobValidation
        )
    )

    # Parse the result
    result = JobValidation.model_validate_json(response.text)
    
    print(f"--- Gatekeeper Analysis ---")
    print(f"Detected YOE: {result.years_required}")
    print(f"Reasoning: {result.reasoning}")

    if not result.is_fit:
        print("CRITICAL: Experience requirement too high. Executing Kill Switch.")
        sys.exit(0) # Exit gracefully so GitHub Action stops here
    
    print("PROCEED: Job meets criteria. Handing off to Agent 2.")
    return result

if __name__ == "__main__":
    # Test sample
    sample_job = "Looking for a React dev with 5 years of Spring Boot experience."
    run_gatekeeper(sample_job)