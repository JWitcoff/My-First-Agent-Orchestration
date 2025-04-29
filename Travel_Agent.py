from dotenv import load_dotenv
import os
from agents import Agent, Runner, function_tool
from pydantic import BaseModel, Field
import asyncio
import requests

load_dotenv()

model = os.getenv("MODEL_CHOICE", "gpt-4o-mini")

class TravelPlan(BaseModel):
    destination: str = Field(description="The user's preferred destination")
    duration_days: int = Field(description="The length of the trip")
    budget: float = Field(description="The user's budget for the trip")
    activities: list[str] = Field(description="A list of activities to do in the destination")
    notes: str = Field(description="Any additional information about the trip")

@function_tool
def get_current_weather(city: str) -> str:
    """Get the current weather for a specific city using OpenWeatherMap API"""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return f"Weather information is currently unavailable. Please check a weather service for current conditions in {city}."
    
    url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return f"The current weather in {city} is {data['weather'][0]['description']} with a temperature of {data['main']['temp']}°C (feels like {data['main']['feels_like']}°C). Note: this weather report is for the current day."
    except Exception as e:
        return f"Weather information is currently unavailable. Please check a weather service for current conditions in {city}."

# Main Travel Agent
travel_agent = Agent(
    name="Travel Agent",
    instructions="""
    You are a comprehensive travel agent that helps users plan their perfect trip.
    Follow these steps to create a travel plan:
    1. Extract the key information from the user's request (destination, duration, budget, interests, accommodation preferences)
    2. Check the current weather at the destination (optional)
    3. Create a realistic budget breakdown considering:
       - Accommodation costs (if available)
       - Transportation costs (if available)
       - Food and dining costs (if available)
       - Activities and attractions
    4. Suggest specific activities based on the user's interests
    5. Provide practical travel tips, local insights, or safety notes about the destination.
    Always ensure the plan stays within the user's budget and duration constraints.
    If the budget seems unrealistic for the destination, suggest adjustments or alternatives.
    """,
    model=model,
    output_type=TravelPlan,
    tools=[get_current_weather]
)

async def main():
    queries = [
        "I want to go to Tokyo for 5 days on a budget of $5000. I love history and the outdoors. I want to stay in a hotel with a pool.",
        "I'm going to Paris for 3 days. I like to eat good food and explore local culture. I want to stay a hotel near the Eiffel Tower.",
        "I'm going to San Francisco for 4 days. I like to hike and explore nature. Ideally near the Golden Gate Bridge."
    ]

    for query in queries:
        print("\n" + "="*50)
        print(f"Query: {query}\n")

        result = await Runner.run(travel_agent, query)

        print("\nFinal Response:")
        plan = result.final_output
        print(f"Destination: {plan.destination}")
        print(f"Duration: {plan.duration_days} days")
        print(f"Budget: ${plan.budget}")
        print("Activities:")
        for activity in plan.activities:
            print(f"  - {activity}")
        print(f"Notes: {plan.notes}")

if __name__ == "__main__":
    asyncio.run(main())
