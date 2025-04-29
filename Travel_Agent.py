from dotenv import load_dotenv
import os
from agents import Agent, Runner
from pydantic import BaseModel, Field
import asyncio

load_dotenv()

model = os.getenv("MODEL_CHOICE", "gpt-4o-mini")

class TravelPlan(BaseModel):
    destination: str = Field(description="The user's preferred destination")
    duration_days: int = Field(description="The length of the trip")
    budget: float = Field(description="The user's budget for the trip")
    activities: list[str] = Field(description="A list of activities to do in the destination")
    notes: str = Field(description="Any additional information about the trip")

# Main Travel Agent
travel_agent = Agent(
    name="Travel Agent",
    instructions="""
    You are a comprehensive travel agent that helps users plan their perfect trip.
    You can create personalized travel itineraries based on the user's interests and preferences.
    Always be helpful, informative, and enthusiastic about travel.
    Provide specific recommendations based on the user's preferences and interests.
    When creating travel plans, always consider the following:
    - Destination: The user's preferred destination
    - Duration: The length of the trip
    - Budget: The user's budget for the trip
    - Interests: The user's interests and preferences
    - Accommodation: The user's preferred type of accommodation as well as 
    local attractions and activities at that destination.
    """,
    model=model,
    output_type=TravelPlan
)

async def main():
    # Example Queries
    queries = [
        "I want to go to Tokyo for 5 days on a budget of $1000. I like history and art. I want to stay in a hotel with a pool.",
        "I'm going to Paris for 3 days. I like to eat good food and explore local culture. I want to stay in a hotel with a pool.",
        "I'm going to San Francisco for 4 days. I like to hike and explore nature. I want to stay in a hotel with a pool."
    ]

    for query in queries:
        print("\n" + "="*50)
        print(f"Query: {query}\n")
    
        result = await Runner.run(travel_agent, query)

        print("\nFinal Response:")
        print(result.final_output)

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
