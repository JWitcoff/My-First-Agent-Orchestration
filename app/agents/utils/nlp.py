import os
import openai
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def extract_landmark_hint(preferences: str, destination: str) -> str:
    """
    Use OpenAI to extract a landmark from hotel preferences like:
    "hotel with a pool, near Shibuya" → "Shibuya"
    """
    prompt = f"""
Given this hotel preference string: "{preferences}" and destination city: "{destination}",
extract a single landmark or neighborhood name that refers to a location in the destination city.

Return ONLY the name of the landmark or neighborhood, e.g.:
- "Shibuya"
- "Golden Gate Bridge"
- "Central Park"
- "the Eiffel Tower"

If no such landmark or location is mentioned, respond with "none".
NEVER return full sentences. Output should be a single name or "none".
"""

    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt}
            ],
            temperature=0.3,
        )
        result = response.choices[0].message.content.strip()
        return result if result.lower() != "none" else ""
    except Exception as e:
        print(f"❌ Landmark extraction failed: {e}")
        return ""
