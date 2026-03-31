"""
Lists all Gemini models available for your API key that support generateContent.
Run this to find the correct model ID to use.

Usage: python list_models.py
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise EnvironmentError("GEMINI_API_KEY not set")

genai.configure(api_key=api_key)

print(f"\n{'Model Name':<50} {'Supported Methods'}")
print("-" * 80)

for model in genai.list_models():
    if "generateContent" in model.supported_generation_methods:
        print(f"{model.name:<50} {model.supported_generation_methods}")
