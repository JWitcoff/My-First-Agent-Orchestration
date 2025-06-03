# app/agents/utils/landmarks.py
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_landmark_from_preferences(preference_text: str) -> str:
    """Uses GPT to extract the hotel location landmark from user preferences."""
    prompt = f"""
Extract ONLY the location-related landmark or area from the following hotel preferences:

"{preference_text}"

If none found, return an empty string.

Examples:
- "Pool, near Shibuya" → "Shibuya"
- "Close to Eiffel Tower, gym" → "Eiffel Tower"
- "Walking distance to downtown" → "downtown"
- "Spa and gym" → ""
- "No preference" → ""

Landmark:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ Failed to extract landmark: {e}")
        return ""
