import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

# Initialize the client
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def run_tailor(job_description: str, master_resume: str):
    """
    Agent 2: Tailors the resume using Claude 3.5 Sonnet.
    Uses Prompt Caching for the Master Resume to save 90% on costs.
    """
    
    # We use the Beta header for Prompt Caching
    response = client.beta.prompt_caching.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        temperature=0.4,
        system=[
            {
                "type": "text", 
                "text": "You are an expert technical recruiter. Use the following Master Resume to tailor a response.",
                "cache_control": {"type": "ephemeral"} # This starts the cache
            },
            {
                "type": "text",
                "text": f"MASTER RESUME: {master_resume}",
                "cache_control": {"type": "ephemeral"} # This caches the heavy data
            }
        ],
        messages=[
            {
                "role": "user",
                "content": f"Tailor my experience for this job: {job_description}. Output a 150-word cover letter and 3 specific bullet points."
            }
        ]
    )
    
    # Check if we hit the cache (to verify savings)
    cache_status = getattr(response.usage, 'cache_read_input_tokens', 0)
    print(f"Tailor finished. Tokens saved via cache: {cache_status}")
    
    return response.content[0].text

if __name__ == "__main__":
    # Test Data
    my_resume = "Software Engineer at Philips, 2.5 years exp, Java, Spring Boot, React..."
    job_info = "Seeking a Full Stack dev for a HealthTech startup using React/Spring."
    
    result = run_tailor(job_info, my_resume)
    print("\n--- TAILORED CONTENT ---")
    print(result)