# ─── Standard Library Imports ───────────────────────────────────────────────
import os
import re
import json
import asyncio
import inspect

# ─── Third-Party Library Imports ────────────────────────────────────────
import openai
from openai import AsyncOpenAI
import requests
from typing import List, Optional, Any
from pydantic import BaseModel, Field

# ─── Custom or Project-Specific Imports ──────────────────────────────
from amadeus import Client, ResponseError

# -- Agent Imports --
from app.agents.tools.hotel_search import hotel_finder_tool
from app.agents.tools.flight_search import flight_finder_tool
from app.agents.utils.dates import parse_date_range_fuzzy
from app.agents.utils.nlp import extract_landmark_hint
from app.config import AMA_CLIENT, OPENAI_API_KEY, MODEL_CHOICE



# -- Amadeus API Setup --
client_id = os.getenv("AMADEUS_API_KEY")
client_secret = os.getenv("AMADEUS_API_SECRET")
if not client_id or not client_secret:
    raise ValueError("Missing Amadeus API credentials in environment variables.")

amadeus = Client(client_id=client_id, client_secret=client_secret)

# -- Helper: Extract JSON from Text --
def extract_json_from_text(text):
    """Extract the first JSON object found inside a text blob."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match: 
        return match.group(0)
    else:
        return None

# -- Helper: Test Amadeus API Connection --
def test_amadeus_connection():
    """Test the Amadeus API connection with a simple request"""
    try:
        response = amadeus.reference_data.locations.get(
            keyword="Paris",
            subType="CITY"
        )
        print("✅ Amadeus connection successful!")
        print(f"Sample response: {response.data[0] if response.data else 'No data'}")
        return True
    except ResponseError as e:
        print(f"❌ Amadeus API Error: {str(e)}")
        print(f"Error details: {e.response.body}")
        return False
    except Exception as e:
        print(f"❌ Unexpected Amadeus error: {str(e)}")
        return False

# -- Run API Tests --
print("\n=== Testing Amadeus API Connection ===")
api_working = test_amadeus_connection()
print("=======================================\n")


# -- Model Selection --
model = os.getenv("MODEL_CHOICE", "gpt-4o")

# ─── Models and Agent Classes (Schemas, Not Instances) ──────────────────────

class Agent:
    def __init__(self, name, instructions, model, output_type=None, tools=None, handoffs=None):
        self.name = name
        self.instructions = instructions
        self.model = model
        self.output_type = output_type
        self.tools = tools or []
        self.handoffs = handoffs or []
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Async OpenAI client

    async def run(self, query, history=None):
        history = history or []

        # ✅ If query is already structured dict, use it directly
        if isinstance(query, dict):
            pre_extracted_inputs = query
        else:
            pre_extracted_inputs = None

        system_prompt = f"""You are {self.name}.
    {self.instructions}
    If tools are provided, you may call them when needed.
    Always return your output in strict JSON matching this schema:
    {json.dumps(self.output_type.model_json_schema(), indent=2) if self.output_type else "Return plain text."}
    """

        # 🔵 Special pre-processing for FlightAgent and HotelAgent ONLY IF needed
        if pre_extracted_inputs is None and self.name in ["Flight Agent", "Hotel Agent"]:
            analysis_prompt = f"""Extract the following information from the user's travel request:

- Origin city (default to "Los Angeles" if not provided)
- Destination city (required)
- Travel dates (list of strings, rough time periods are OK)
- Trip duration in days (default 5 if missing)
- Budget (optional)

Always respond ONLY with pure JSON:
{{
"origin": "...",
"destination": "...",
"dates": ["..."],
"duration_days": 5,
"budget": 3000.0
}}

If anything is missing, make reasonable assumptions. DO NOT ask clarifying questions. Assume defaults.

User request: "{query}"
"""
            extraction_response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": analysis_prompt}
                ],
                temperature=0.2
            )

            extraction_content = extraction_response.choices[0].message.content
            try:
                extracted_json = extract_json_from_text(extraction_content)
                pre_extracted_inputs = json.loads(extracted_json)
            except Exception as e:
                print(f"Failed to extract inputs: {e}")
                pre_extracted_inputs = None
        # 🛠️ Regular agent message flow
        if isinstance(query, dict):
            query_text = json.dumps(query, indent=2)
        else:
            query_text = query

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query_text}
        ]

        functions = []
        function_name_map = {}

        for tool in self.tools:
            functions.append(tool.openai_schema)
            function_name_map[tool.openai_schema["name"]] = tool

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            functions=functions if functions else None,
            function_call="auto" if functions else None,
            temperature=0.3
        )

        choice = response.choices[0]
        content = choice.message.content  # ✅ ensures content is always defined

        # 🛠️ If the model called a tool
        if choice.finish_reason == "function_call":
            function_call = choice.message.function_call
            func_name = function_call.name
            func_args = json.loads(function_call.arguments)
            print(f"Tool call: {func_name} with {func_args}")

            if func_name in function_name_map:
                tool = function_name_map[func_name]
                tool_args = func_args or pre_extracted_inputs or {}

                # 🛠 Clean tool_args based on the tool's function signature
                tool_signature = inspect.signature(tool)
                expected_params = set(tool_signature.parameters.keys())
                cleaned_args = {k: v for k, v in tool_args.items() if k in expected_params}

                tool_result = tool(**cleaned_args)

                if inspect.iscoroutinefunction(tool):
                    tool_result = await tool_result

                # Continue the conversation after tool use
                messages.append({
                    "role": "assistant",
                    "function_call": {
                        "name": func_name,
                        "arguments": json.dumps(func_args)
                    }
                })
                messages.append({
                    "role": "function",
                    "name": func_name,
                    "content": json.dumps(tool_result)
                })

                # Re-call model with tool result
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3
                )

                choice = response.choices[0]
                content = choice.message.content

        # Guardrail: if model asks clarifying questions, raise an error
        if content and any(phrase in content.lower() for phrase in ["could you", "can you", "please provide", "may i know"]):
            raise ValueError(f"Model asked a clarifying question instead of using defaults: {content}")

        if self.output_type:
            try:
                json_blob = extract_json_from_text(content)
                if not json_blob:
                    raise ValueError(f"Could not find JSON in response: {content}")

                data = json.loads(json_blob)
                return self.output_type.model_validate(data)
            except Exception as e:
                print(f"Error parsing output: {e}")
                print(f"Raw output: {content}")
                raise e
        else:
            return content

# -- Pydantic Models for Structured Outputs ──────────────────────────────
    
class TravelPlan(BaseModel):
    destination: str = Field(description="The user's preferred destination")
    duration_days: int = Field(description="The length of the trip")
    dates: list[str] = Field(description="The rough time period of the trip, such as ['late June', 'early July', 'Summer', 'Winter', or specific dates like '2024-07-01 to 2024-07-10']")
    budget: float = Field(description="The user's total budget for the trip")
    amenities: Optional[List[str]] = Field(default_factory=list, description="Requested hotel amenities like gym, pool, spa, etc.")
    location_preferences: Optional[str] = Field(default="", description="Location-related hotel preferences, e.g., 'near Shibuya' or 'walkable to downtown'")
    activities: list[str] = Field(description="A list of specific, location-based activities to do in the destination")
    notes: Optional[str] = Field(default=None, description="Any additional information about the trip")


class FlightRecommendation(BaseModel):
    origin: str = Field(description="The origin city")
    airline: str = Field(description="The airline to fly with")
    destination: str = Field(description="The destination city")
    departure_date: str = Field(description="The departure date")
    return_date: str = Field(description="The return date")
    price: float = Field(description="The price of the flight")
    direct_flight: bool = Field(description="Whether the flight is a direct flight")
    recommendation_reason: str = Field(description="The reason for the recommendation")

class HotelRecommendation(BaseModel):
    name: str = Field(description="The name of the hotel")
    destination: str = Field(description="The destination city")
    checkin_date: str = Field(description="The check-in date")
    checkout_date: str = Field(description="The check-out date")
    price_per_night: float = Field(description="The price per night for the hotel")
    amenities: list[str] = Field(description="The amenities available at the hotel, i.e. pool, gym, etc.")
    recommendation_reason: str = Field(description="The reason for the recommendation")

# -- Helper Function to Convert Functions to OpenAI-Compatible Tools -- 
def function_tool(func):
    """Wraps a function and attaches an OpenAI schema to it."""
    func.openai_schema = {
        "name": func.__name__,
        "description": func.__doc__ or "",
        "parameters": {}  # Can fill later if needed
    }
    return func

# # ─── Tools (Functions Available to Agents) ──────────────────────

# -- Hotel Agent Object --
hotel_agent = Agent( 
    name="Hotel Agent",
    instructions="""
You are a hotel booking specialist responsible for finding the best accommodations based on the user's preferences, including amenities, location, and price.

YOU MUST ALWAYS RETURN A VALID JSON OBJECT MATCHING THE HotelRecommendation SCHEMA.
Even if no hotel is needed (for example, if check-in and check-out dates are the same), 
you must still return a HotelRecommendation JSON with empty or default fields, 
and set the "recommendation_reason" to explain why no hotel is needed.

NEVER write casual sentences. NEVER explain outside the JSON. NEVER break structure.

=========================================
EXAMPLES:

✅ If a hotel is found:
{
  "name": "Hotel Tokyo Palace",
  "destination": "Tokyo",
  "checkin_date": "2025-06-25",
  "checkout_date": "2025-06-30",
  "price_per_night": 150.0,
  "amenities": ["WiFi", "Pool", "Breakfast"],
  "recommendation_reason": "Found a hotel matching your preferences."
}

✅ If it is a day trip (no hotel needed):
{
  "name": "",
  "destination": "Tokyo",
  "checkin_date": "2025-06-25",
  "checkout_date": "2025-06-25",
  "price_per_night": 0.0,
  "amenities": [],
  "recommendation_reason": "No hotel needed for a same-day trip."
}

elif subagent.name == "Hotel Agent":
    tasks.append(subagent.run({
        "destination": primary_output.destination,
        "checkin_date": dep_date,
        "checkout_date": ret_date,
        "budget": primary_output.budget * 0.4,
        "preferences": combined_prefs,
        "landmark_hint": landmark_hint  # ✅ NEW
    }, history))


=========================================

CRITICAL RULES:
- Always use the 'HotelRecommendation' structure.
- Even if no hotel is needed, you must still populate all fields.
- 'recommendation_reason' must clearly explain the situation in 2-3 sentences.

""",
    model=model,
    tools=[hotel_finder_tool],
    output_type=HotelRecommendation
)

# -- Flight Agent --
flight_agent = Agent(
    name="Flight Agent",
    instructions="""
You are a flight agent that helps users find the best flight for their trip.
Use the `flight_finder` tool to find the best flight for the user's trip.

CRITICAL: You MUST return ONLY valid JSON matching the FlightRecommendation schema.
NEVER return conversational text. NEVER ask questions. NEVER explain.

IMPORTANT RULES:
- If any critical information is missing, ALWAYS assume defaults
- Default departure city is 'Los Angeles' unless otherwise specified  
- If no dates provided, assume travel ~1 month from today for 5 days
- Always return valid JSON following FlightRecommendation schema
- Never ask follow-up questions - proceed with reasonable assumptions

EXAMPLE RESPONSE (always return JSON like this):
{
  "origin": "Los Angeles",
  "airline": "American Airlines", 
  "destination": "Tokyo",
  "departure_date": "2025-06-25",
  "return_date": "2025-06-30", 
  "price": 1245.50,
  "direct_flight": true,
  "recommendation_reason": "Found flight based on your search criteria."
}

Steps:
1. Extract origin, destination, dates from user request
2. Call flight_finder tool with extracted information
3. Return the result as valid JSON only - no extra text
""",
    model=model,
    output_type=FlightRecommendation,
    tools=[flight_finder_tool]
)

# -- Travel Agent Object (And Overall Orchestrator) --

travel_agent = Agent(
    name="Travel Agent",
    instructions="""
You are a comprehensive travel agent responsible for extracting and organizing travel information from user requests.

You MUST respond with a valid JSON object matching the TravelPlan schema.
No extra text, no greetings, no markdown formatting.
Only pure, valid JSON output.

=========================================
YOUR EXACT WORKFLOW:

1. ANALYZE USER REQUEST AND EXTRACT ALL DATA:
   - Extract destination, dates, budget, duration, activities
   - CRITICAL: Always extract hotel amenities and location preferences:

   FOR AMENITIES: Look for words like "pool", "gym", "spa", "breakfast", "wifi"
   FOR LOCATION: Look for phrases like "near [landmark]", "by [location]", "close to [place]"

   Examples of what to extract:
   • "hotel with a pool near Shibuya" → amenities: ["pool"], location_preferences: "near Shibuya"
   • "stay by the Eiffel Tower" → amenities: [], location_preferences: "by the Eiffel Tower"
   • "near Golden Gate Bridge with gym" → amenities: ["gym"], location_preferences: "near Golden Gate Bridge"

   YOU MUST populate the amenities and location_preferences fields in your JSON response.

2. BUILD FINAL TRAVEL PLAN:
    Return a JSON object with EXACTLY these fields:
    {
    "destination": string,           // The city name (e.g., "Tokyo")
    "duration_days": integer,        // Trip length in days
    "dates": array of strings,       // List of dates or date range
    "budget": float,                 // Total trip budget
    "amenities": array of strings,   // Hotel amenities like ["pool", "gym"]
    "location_preferences": string,  // Location like "near Shibuya Crossing"
    "activities": array of strings,  // Plain-text list of activities that are specific to the destination (i.e. "Visit historical temples like Senso-ji", "Explore outdoor gardens like Ueno Park", "Visit museums like the Tokyo National Museum")
    "notes": string (optional)       // Notes about any assumptions
}

CRITICAL DATE RULE:
- NEVER generate dates in the past
- If user says "Jan 1 to Jan 8" without year, assume NEXT occurrence (2026-01-01 to 2026-01-08)
- BUT if a user says "Winter 2025" or "Summer 2025", assume the current year (2025)
- All dates must be in the future (after 2025-06-04)

CRITICAL BUDGET RULE:
- If no budget is provided, estimate a reasonable budget based on:
  * Domestic trips: $1000-2000 
  * International trips: $2000-5000
- NEVER use null for budget - always provide a number
- If an estimated budget is needed, mention this in the "notes" field of the final plan.

When you are ready to respond: Immediately output the JSON object. No explanations.

=========================================
EXAMPLE RESPONSE:
{
  "destination": "Tokyo",
  "duration_days": 5,
  "dates": ["2025-06-25", "2025-06-30"],
  "budget": 5000.0, (or estimated budget if no budget is provided)
  "amenities": ["pool"],
  "location_preferences": "near Shibuya Crossing",
  "activities": ["Visit historical temples like Senso-ji", "Explore outdoor gardens like Ueno Park", "Visit museums like the Tokyo National Museum"],
  "notes": "Dates estimated based on late June request. Budget estimated based on domestic trip."
}
""",
    model=model,
    output_type=TravelPlan,
    handoffs=[flight_agent, hotel_agent],
)

# ─── Runner and Result Classes ───────────────────────────────────────

class Runner:
    @staticmethod
    async def run(agent: Agent, input_text: str):
        """Run agents in parallel instead of sequential"""
        outputs = []
        history = []
        
        # Step 1: Run primary agent
        primary_output = await agent.run(input_text, history)
        outputs.append(primary_output)
        
        # Step 2: Run handoff agents IN PARALLEL
        if agent.handoffs and isinstance(primary_output, TravelPlan):
            tasks = []

            print(f"🐛 primary_output.dates: {primary_output.dates}")
            print(f"🐛 primary_output.duration_days: {primary_output.duration_days}")

            dep_date, ret_date = parse_date_range_fuzzy(
                primary_output.dates, primary_output.duration_days
            )

            print(f"🐛 Parsed result: {dep_date} to {ret_date}")


            # Combine preferences from amenities and location
            amenities = getattr(primary_output, "amenities", [])
            location = getattr(primary_output, "location_preferences", "")
            combined_prefs = ", ".join(filter(None, amenities + [location.strip()]))
            
            # Landmark Debugging
            print(f"🐛 amenities: {amenities}")
            print(f"🐛 location_preferences: '{location}'")  
            print(f"🐛 combined_prefs: '{combined_prefs}'")

            # Optional: extract a landmark string from natural language if needed
            landmark_hint = await extract_landmark_hint(combined_prefs, primary_output.destination)
            print(f"🧭 Landmark Hint Extracted: {landmark_hint}")


            for subagent in agent.handoffs:
                if subagent.name == "Flight Agent":
                    tasks.append(subagent.run({
                        "origin": "Los Angeles",  # or extract from primary_output
                        "destination": primary_output.destination,
                        "departure_date": dep_date,
                        "return_date": ret_date,
                        "budget": primary_output.budget * 0.6,
                        "dates": [dep_date, ret_date]
                    }, history))

                elif subagent.name == "Hotel Agent":
                    tasks.append(subagent.run({
                        "destination": primary_output.destination,
                        "checkin_date": dep_date,
                        "checkout_date": ret_date,
                        "budget": primary_output.budget * 0.4,
                        "preferences": combined_prefs,
                        "landmark_hint": landmark_hint
                    }, history))

            if tasks:
                sub_results = await asyncio.gather(*tasks, return_exceptions=True)
                outputs.extend([r for r in sub_results if not isinstance(r, Exception)])

        return SimpleResult(outputs)


class SimpleResult:
    def __init__(self, outputs: List[Any]):
        self.outputs = outputs

    @property
    def final_output(self):
        """
        Pick the 'main' final output: prioritize TravelPlan if present.
        """
        for output in self.outputs:
            if isinstance(output, TravelPlan):
                return output
        return self.outputs[0] if self.outputs else None

# -- Demo Queries for Testing the Travel Agent System ──────────────────

async def main():
    queries = [
        "I want to go to Tokyo for 5 days in late June on a budget of $5000. I love history and the outdoors. I want to stay in a hotel with a pool and be near the Shibuya Crossing.",
        "I'm going to Paris for 3 days during the winter. I like to eat good food and explore local culture. I want to stay at a hotel near the Eiffel Tower.",
        "I'm going to San Francisco from Jan 1 to Jan 8. I like to hike and explore nature. Ideally near the Golden Gate Bridge.",
        "I'm thinking of going to Rome for a week in late Summer. I want to eat amazing food, see ruins, and stay somewhere walkable, maybe near the Colosseum. Budget is $3000.",
        "I'm going to New York today for the day. I have a budget of $500. I want to eat great steak and see a Broadway show. Can I stay by the Empire State Building?"
    ]

    for query in queries:
        print("\n" + "="*50)
        print(f"Query: {query}\n")

        result = await Runner.run(travel_agent, query)

        flight = None
        hotel = None
        travel_plan = None

        for output in result.outputs:
            if isinstance(output, FlightRecommendation):
                flight = output
            elif isinstance(output, HotelRecommendation):
                hotel = output
            elif isinstance(output, TravelPlan):
                travel_plan = output

        if not travel_plan:
            print("\n⚠️ No full travel plan created.")
            continue

        print("\nFinal Travel Plan:")
        print(f"Destination: {travel_plan.destination}")
        print(f"Duration: {travel_plan.duration_days} days")
        print(f"Dates: {', '.join(travel_plan.dates)}")
        print(f"Budget: ${travel_plan.budget}")
        print("Activities:")
        for activity in travel_plan.activities:
            print(f"  - {activity}")
        print(f"Notes: {travel_plan.notes or 'No additional notes.'}")

        if flight:
            print("\nFlight Recommendation:")
            print(f"  - Airline: {flight.airline}")
            print(f"  - From {flight.origin} to {flight.destination}")
            print(f"  - Departure: {flight.departure_date}")
            print(f"  - Return: {flight.return_date}")
            print(f"  - Price: ${flight.price}")
            print(f"  - Direct: {'Yes' if flight.direct_flight else 'No'}")
            print(f"  - Reason: {flight.recommendation_reason}")
        else:
            print("\n⚠️ No flight recommendation found.")

        if hotel:
            print("\nHotel Recommendation:")
            print(f"  - Hotel: {hotel.name}")
            print(f"  - Check-In: {hotel.checkin_date}")
            print(f"  - Check-Out: {hotel.checkout_date}")
            print(f"  - Price per Night: ${hotel.price_per_night}")
            print(f"  - Amenities: {', '.join(hotel.amenities) if hotel.amenities else 'No amenities listed.'}")
            print(f"  - Reason: {hotel.recommendation_reason}")
        else:
            print("\n⚠️ No hotel recommendation found.")

# -- Run the main function --
if __name__ == "__main__":
    asyncio.run(main())
