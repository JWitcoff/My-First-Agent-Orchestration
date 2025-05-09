# ─── Standard Library Imports ─────────────────────────────────────
import os
import re
import json
import asyncio
import random
from datetime import datetime, timedelta
import inspect

# ─── Third-Party Library Imports ─────────────────────────────────
import openai
from openai import AsyncOpenAI
from dotenv import load_dotenv
import requests
from typing import List, Optional, Any
from pydantic import BaseModel, Field

# ─── Custom or Project-Specific Imports ──────────────────────────
from amadeus import Client, ResponseError

import streamlit as st

# -- For Streamlit -- 
openweather_key = st.secrets["OPENWEATHER_API_KEY"]
amadeus_key = st.secrets["AMADEUS_API_KEY"]
amadeus_secret = st.secrets["AMADEUS_API_SECRET"]
openai_key = st.secrets["OPENAI_API_KEY"]

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
        # Simple request to test authentication
        response = amadeus.reference_data.locations.get(
            keyword="Paris",
            subType="CITY"
        )
        print("✅ Amadeus connection successful!")
        print(f"Sample response: {response.data[0] if response.data else 'No data'}")
        return True
    except ResponseError as e:
        print(f"❌ Amadeus API Error: {str(e)}")
        # Print more detailed error information
        print(f"Error details: {e.response.body}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return False

# Call the test function early
print("\n=== Testing Amadeus API Connection ===")
api_working = test_amadeus_connection()
print("=======================================\n")

# -- Helper: Convert City Name to IATA Code --
CITY_CODE_CACHE = {
    "tokyo": "TYO",
    "paris": "PAR",
    "rome": "ROM",
    "san francisco": "SFO",
    "new york": "NYC",
    "los angeles": "LAX",
}

def get_city_code(city_name: str) -> Optional[str]:
    """Find the IATA city code for a given city name using Amadeus API."""
    city_name_lower = city_name.strip().lower()
    if city_name_lower in CITY_CODE_CACHE:
        return CITY_CODE_CACHE[city_name_lower]

    try:
        response = amadeus.reference_data.locations.get(
            keyword=city_name,
            subType="CITY"
        )
        results = response.data
        if results:
            return results[0].get("iataCode")
        else:
            return None
    except Exception as e:
        print(f"Error getting city code for {city_name}: {e}")
        return None

    
# -- Model Selection--

model = os.getenv("MODEL_CHOICE", "gpt-4o")

# Helper Fuzzy Date Parsing -- 

def parse_date_range_fuzzy(dates: list[str], duration_days: int = 5):
    """
    Parse dates from various formats including specific dates, relative terms, months, and seasons.
    Returns tuple of (departure_date, return_date)
    """
    today = datetime.today()
    max_future_date = today + timedelta(days=365)  # Limit to 1 year in future
    
    # Handle empty input
    if not dates or len(dates) == 0:
        default_dep = min((today + timedelta(days=30)).date(), max_future_date.date())
        default_ret = min(default_dep + timedelta(days=duration_days), max_future_date.date())
        return str(default_dep), str(default_ret)
    
    # Try to parse exact dates if there are two entries
    try:
        if len(dates) == 2:
            # Try multiple date formats
            for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y", "%B %d, %Y", "%b %d, %Y"]:
                try:
                    dep = datetime.strptime(dates[0], fmt).date()
                    ret = datetime.strptime(dates[1], fmt).date()
                    # Ensure dates are within valid range
                    dep = min(max(dep, today.date()), max_future_date.date())
                    ret = min(max(ret, dep), max_future_date.date())
                    return str(dep), str(ret)
                except ValueError:
                    continue
    except Exception:
        pass
    
    # If we have at least one entry, try to interpret it
    if len(dates) >= 1:
        term = dates[0].strip().lower()
        term = re.sub(r"[^a-z\s0-9]", "", term)  # Keep letters, spaces, and numbers
        
        # Handle "today"
        if "today" in term:
            dep = today.date()
            ret = min(dep + timedelta(days=duration_days), max_future_date.date())
            return str(dep), str(ret)
        
        # Handle "tomorrow"
        if "tomorrow" in term:
            dep = min((today + timedelta(days=1)).date(), max_future_date.date())
            ret = min(dep + timedelta(days=duration_days), max_future_date.date())
            return str(dep), str(ret)
        
        # Handle relative dates
        if "next week" in term:
            next_monday = today + timedelta(days=(7 - today.weekday()))
            dep = min(next_monday.date(), max_future_date.date())
            ret = min(dep + timedelta(days=duration_days), max_future_date.date())
            return str(dep), str(ret)
            
        if "next month" in term:
            # First day of next month
            if today.month == 12:
                dep = min(datetime(today.year + 1, 1, 15).date(), max_future_date.date())
            else:
                dep = min(datetime(today.year, today.month + 1, 15).date(), max_future_date.date())
            ret = min(dep + timedelta(days=duration_days), max_future_date.date())
            return str(dep), str(ret)
        
        # Month names
        month_lookup = {
            "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, 
            "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7, 
            "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, 
            "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12
        }
        
        # Try to extract a month
        for word, month in month_lookup.items():
            if word in term:
                # Determine which part of the month
                base_day = 15  # Default to mid-month
                if "early" in term:
                    base_day = 5
                elif "mid" in term:
                    base_day = 15
                elif "late" in term or "end" in term:
                    base_day = 25
                
                # Determine year
                year = today.year if today.month <= month else today.year + 1
                if year > max_future_date.year:
                    year = max_future_date.year
                
                # Create dates
                dep = min(datetime(year, month, base_day).date(), max_future_date.date())
                ret = min(dep + timedelta(days=duration_days), max_future_date.date())
                return str(dep), str(ret)
        
        # Seasons
        seasonal = {
            "winter": ("january", 15),
            "spring": ("april", 15),
            "summer": ("july", 15),
            "fall": ("october", 15),
            "autumn": ("october", 15)
        }
        
        for season, (month_word, day) in seasonal.items():
            if season in term:
                month = month_lookup.get(month_word)
                year = today.year if today.month <= month else today.year + 1
                if year > max_future_date.year:
                    year = max_future_date.year
                dep = min(datetime(year, month, day).date(), max_future_date.date())
                ret = min(dep + timedelta(days=duration_days), max_future_date.date())
                return str(dep), str(ret)
    
    # Default to near future if we couldn't parse anything
    dep = min((today + timedelta(days=30)).date(), max_future_date.date())
    ret = min(dep + timedelta(days=duration_days), max_future_date.date())
    return str(dep), str(ret)

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
        # Safely stringify if query is structured input (dict)
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
    budget: float = Field(description="The user's budget for the trip")
    activities: list[str] = Field(description="A list of activities to do in the destination")
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
    return func  # <-- No need to create `.func` or anything fancy

# # ─── Tools (Functions Available to Agents) ──────────────────────

@function_tool
def get_current_weather(city: Optional[str] = None) -> str:
    """Get the current weather for a specific city using OpenWeatherMap API"""
    if not city:
        return "Weather information is currently unavailable because no city was provided."

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

        if "weather" not in data or "main" not in data:
            return f"Weather information is currently unavailable for {city}."
        
        return f"The current weather in {city} is {data['weather'][0]['description']} with a temperature of {data['main']['temp']}°C (feels like {data['main']['feels_like']}°C). Note: this weather report is for the current day."
    
    except Exception as e:
        print(f"Error getting current weather for {city}: {e}")
        return f"Weather information is currently unavailable. Please check a weather service for current conditions in {city}."
    
# -- Flight Finder --
# First, add this mock data helper function
def get_mock_flight_data(origin, destination, dep_date, ret_date):
    """Return simulated flight data when the API fails"""
    # Estimate flight prices based on distance
    domestic_destinations = ["NYC", "LAX", "SFO", "LAS", "MIA", "CHI", "BOS", "DFW"]
    short_haul = ["CUN", "YVR", "YYZ", "MEX", "HAV", "SJO"]
    
    origin_code = origin[:3].upper() if len(origin) > 3 else origin.upper()
    dest_code = destination[:3].upper() if len(destination) > 3 else destination.upper()
    
    # Estimate price based on flight type
    price = 350  # Default domestic price
    if dest_code in domestic_destinations and origin_code in domestic_destinations:
        price = 350
    elif dest_code in short_haul or origin_code in short_haul:
        price = 750
    else:
        price = 1200  # International long-haul
    
    # Add some random variation
    import random
    price = price + random.randint(-50, 150)
    
    # Mock airline selection
    airlines = ["Delta", "United", "American", "JetBlue", "Southwest"]
    airline = random.choice(airlines)
    
    return {
        "origin": origin,
        "destination": destination,
        "departure_date": dep_date,
        "return_date": ret_date,
        "airline": airline,
        "price": price,
        "direct_flight": random.choice([True, False]),
        "recommendation_reason": "NOTE: This is estimated flight data due to API unavailability."
    }

@function_tool
def flight_finder(origin: str, destination: str, dates: list[str]) -> dict:
    """Return a structured flight recommendation with estimated cost and flight details."""
    # Default values - explicitly handle "Current location"
    if not origin or origin.lower() == "current location":
        origin = "Los Angeles"  # Default to Los Angeles if origin is empty or "Current location"
    
    destination = destination or ""
    
    # Prepare response template for errors
    error_response = {
        "origin": origin,
        "destination": destination,
        "departure_date": "",
        "return_date": "",
        "airline": "",
        "price": 0.0,
        "direct_flight": False,
        "recommendation_reason": ""
    }
    
    # Validate inputs
    if not destination:
        error_response["recommendation_reason"] = "No destination provided."
        return error_response
    
    # Parse dates
    try:
        dep_date, ret_date = parse_date_range_fuzzy(dates)
        if not dep_date or not ret_date:
            error_response["recommendation_reason"] = f"Unable to interpret provided dates: {dates}"
            return error_response
            
        # Validate dates are not too far in the future (max 1 year)
        today = datetime.today()
        max_future_date = today + timedelta(days=365)
        
        dep_dt = datetime.strptime(dep_date, "%Y-%m-%d")
        ret_dt = datetime.strptime(ret_date, "%Y-%m-%d")
        
        if dep_dt > max_future_date:
            dep_date = max_future_date.strftime("%Y-%m-%d")
        if ret_dt > max_future_date:
            ret_date = max_future_date.strftime("%Y-%m-%d")
            
        # Update error response template with parsed dates
        error_response["departure_date"] = dep_date
        error_response["return_date"] = ret_date
    except Exception as e:
        error_response["recommendation_reason"] = f"Date parsing error: {str(e)}"
        return error_response
    
    # Try the API call first, then fall back to mock data
    try:
        # Print debug information
        print(f"Searching flights: {origin} to {destination}, {dep_date} to {ret_date}")
        
        origin_code = get_city_code(origin) if len(origin) != 3 else origin.upper()
        destination_code = get_city_code(destination) if len(destination) != 3 else destination.upper()
        
        if not origin_code:
            print(f"Could not find city code for origin: {origin}")
            return get_mock_flight_data(origin, destination, dep_date, ret_date)
            
        if not destination_code:
            print(f"Could not find city code for destination: {destination}")
            return get_mock_flight_data(origin, destination, dep_date, ret_date)
            
        print(f"Using city codes: {origin_code} to {destination_code}")
        
        # Try API call with simpler parameters first
        try:
            response = amadeus.shopping.flight_offers_search.get(
                originLocationCode=origin_code,
                destinationLocationCode=destination_code,
                departureDate=dep_date,
                returnDate=ret_date,
                adults=1
            )
            
            flights = response.data
            if not flights:
                print("No flights found from API")
                return get_mock_flight_data(origin, destination, dep_date, ret_date)
                
            # Process the first flight offer
            flight = flights[0]
            itineraries = flight.get("itineraries", [])
            
            if len(itineraries) < 1:
                print("No itineraries in flight data")
                return get_mock_flight_data(origin, destination, dep_date, ret_date)
            
            # Handle one-way trips vs round trips
            outbound_segments = itineraries[0].get("segments", [])
            if not outbound_segments:
                print("No segments in itinerary")
                return get_mock_flight_data(origin, destination, dep_date, ret_date)
                
            outbound_seg = outbound_segments[0]
            departure_date = outbound_seg.get("departure", {}).get("at", dep_date)[:10]
            
            # Get inbound data if it exists
            return_date = ret_date  # Default to provided return date
            if len(itineraries) > 1:
                inbound_segments = itineraries[1].get("segments", [])
                if inbound_segments:
                    inbound_seg = inbound_segments[0]
                    return_date = inbound_seg.get("departure", {}).get("at", ret_date)[:10]
            
            # Get price and airline
            price = float(flight.get("price", {}).get("total", 0.0))
            airline = outbound_seg.get("carrierCode", "Unknown Airline")
            
            # Count stops
            num_stops = 0
            for itinerary in itineraries:
                num_stops += len(itinerary.get("segments", [])) - 1
            
            return {
                "origin": origin,
                "destination": destination,
                "departure_date": departure_date,
                "return_date": return_date,
                "airline": airline,
                "price": price,
                "direct_flight": num_stops == 0,
                "recommendation_reason": "Found flight based on your search criteria."
            }
            
        except ResponseError as e:
            print(f"Amadeus API error: {str(e)}")
            print(f"Response body: {e.response.body if hasattr(e, 'response') else 'No response body'}")
            # Fall back to mock data
            return get_mock_flight_data(origin, destination, dep_date, ret_date)
            
    except Exception as e:
        print(f"Unexpected error in flight_finder: {str(e)}")
        # Fall back to mock data on any error
        return get_mock_flight_data(origin, destination, dep_date, ret_date)

# -- Hotel Finder ───────────────────────────────────────────────────────

# - Mock Data for Hotel Finder
def get_mock_hotel_data(destination, checkin_date, checkout_date, budget):
    """Return simulated hotel data when the API fails"""
    # Calculate nights of stay
    try:
        checkin_dt = datetime.strptime(checkin_date, "%Y-%m-%d")
        checkout_dt = datetime.strptime(checkout_date, "%Y-%m-%d")
        nights = (checkout_dt - checkin_dt).days
    except:
        nights = 3  # Default to 3 nights if date parsing fails
    
    # Estimate hotel prices based on destination type
    luxury_destinations = ["PAR", "TYO", "NYC", "LON", "ROM", "SYD"]
    mid_tier_destinations = ["SFO", "CHI", "MIA", "BOS", "SEA", "BER"]
    
    dest_code = destination[:3].upper() if len(destination) > 3 else destination.upper()
    
    # Estimate price based on destination tier
    if dest_code in luxury_destinations:
        price_per_night = 250
    elif dest_code in mid_tier_destinations:
        price_per_night = 180
    else:
        price_per_night = 120
    
    # Add some random variation
    price_per_night = price_per_night + random.randint(-20, 40)
    
    # Generate a hotel name
    hotel_prefixes = ["Grand", "Royal", "Plaza", "Majestic", "Luxury", "Embassy", "Comfort", "Downtown"]
    hotel_suffixes = ["Hotel", "Inn", "Suites", "Resort", "Place", "Lodge", "Towers"]
    hotel_name = f"{random.choice(hotel_prefixes)} {destination} {random.choice(hotel_suffixes)}"
    
    # Generate amenities
    all_amenities = ["WiFi", "Pool", "Fitness Center", "Restaurant", "Bar", "Room Service", 
                      "Business Center", "Spa", "Parking", "Airport Shuttle", "Concierge"]
    num_amenities = random.randint(3, 7)
    amenities = random.sample(all_amenities, num_amenities)
    
    return {
        "name": hotel_name,
        "destination": destination,
        "checkin_date": checkin_date,
        "checkout_date": checkout_date,
        "price_per_night": price_per_night,
        "amenities": amenities,
        "recommendation_reason": "NOTE: This is estimated hotel data due to API unavailability."
    }

# -- Hotel Finder Logic --
def actual_hotel_finder_logic(destination: str, checkin_date: str, checkout_date: str, budget: float, preferences: str = "") -> dict:
    """Return a structured hotel recommendation with estimated cost and hotel details."""
    if not destination:
        return {
            "name": "",
            "destination": "",
            "checkin_date": checkin_date,
            "checkout_date": checkout_date,
            "price_per_night": 0.0,
            "amenities": [],
            "recommendation_reason": "ERROR: No destination provided."
        }

    if checkin_date == checkout_date:
        return {
            "name": "",
            "destination": destination,
            "checkin_date": checkin_date,
            "checkout_date": checkout_date,
            "price_per_night": 0.0,
            "amenities": [],
            "recommendation_reason": "No hotel needed for a same-day trip."
        }

    try:
        checkin_dt = datetime.strptime(checkin_date, "%Y-%m-%d")
        checkout_dt = datetime.strptime(checkout_date, "%Y-%m-%d")
        nights = (checkout_dt - checkin_dt).days
        per_night_budget = budget / max(nights, 1)
    except:
        nights = 1
        per_night_budget = budget

    # Get coordinates via Amadeus reference search
    try:
        response = amadeus.reference_data.locations.get(
            keyword=destination,
            subType="CITY"
        )
        if not response.data:
            print(f"No city data found for {destination}")
            return get_mock_hotel_data(destination, checkin_date, checkout_date, budget)

        geo = response.data[0]["geoCode"]
        lat = geo["latitude"]
        lon = geo["longitude"]

    except Exception as e:
        print(f"Error getting geoCode for {destination}: {e}")
        return get_mock_hotel_data(destination, checkin_date, checkout_date, budget)

    print(f"Searching hotels near {destination} ({lat}, {lon}) for dates {checkin_date} to {checkout_date}")

    try:
        city_code = get_city_code(destination)
        if not city_code:
            print(f"No city code found for {destination}")
            return get_mock_hotel_data(destination, checkin_date, checkout_date, budget)

        hotel_response = amadeus.shopping.hotel_offers_search.get(
            cityCode=city_code,
            checkInDate=checkin_date,
            checkOutDate=checkout_date,
            adults=1,
            roomQuantity=1,
            currency="USD",
            radius=5,
            radiusUnit="KM"
        )

        hotels = hotel_response.data

        if not hotels:
            print(f"No hotels found via Amadeus, falling back to mock data.")
            return get_mock_hotel_data(destination, checkin_date, checkout_date, budget)

        candidates = []
        for hotel_data in hotels:
            hotel_info = hotel_data.get("hotel", {})
            offers = hotel_data.get("offers", [])
            if not offers:
                continue

            offer = offers[0]
            total_price = float(offer.get("price", {}).get("total", 0.0))
            price_per_night = total_price / max(nights, 1)

            if price_per_night <= per_night_budget:
                candidates.append((hotel_info, offer, price_per_night))

        if not candidates:
            print(f"No hotels under budget. Falling back to cheapest available.")
            for hotel_data in hotels:
                hotel_info = hotel_data.get("hotel", {})
                offers = hotel_data.get("offers", [])
                if offers:
                    offer = offers[0]
                    total_price = float(offer.get("price", {}).get("total", 0.0))
                    price_per_night = total_price / max(nights, 1)
                    candidates.append((hotel_info, offer, price_per_night))

        if not candidates:
            return get_mock_hotel_data(destination, checkin_date, checkout_date, budget)

        hotel_info, offer, price = sorted(candidates, key=lambda x: x[2])[0]

        amenities = hotel_info.get("amenities", [])
        if not amenities:
            amenities = ["WiFi", "Fitness Center"]

        return {
            "name": hotel_info.get("name", "Unknown Hotel"),
            "destination": destination,
            "checkin_date": checkin_date,
            "checkout_date": checkout_date,
            "price_per_night": price,
            "amenities": amenities,
            "recommendation_reason": "Found a hotel matching your preferences and budget."
        }

    except ResponseError as e:
        print(f"Amadeus API error during hotel search: {e}")
        return get_mock_hotel_data(destination, checkin_date, checkout_date, budget)

    except Exception as e:
        print(f"Unexpected error during hotel search: {e}")
        return get_mock_hotel_data(destination, checkin_date, checkout_date, budget)

# -- Tool-wrapped hotel finder --
@function_tool
def hotel_finder_tool(destination: str, checkin_date: str, checkout_date: str, budget: float, preferences: str = "") -> dict:
    """Tool-wrapped hotel finder."""
    return actual_hotel_finder_logic(destination, checkin_date, checkout_date, budget, preferences)


# -- Agent Instances (Objects) ───────────────────────────────────────────────────────

# -- Hotel Agent Object --
hotel_agent = Agent( 
    name="Hotel Agent",
    instructions="""
You are a hotel booking specialist responsible for finding the best accommodations.

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

=========================================

CRITICAL RULES:
- Always use the 'HotelRecommendation' structure.
- Even if no hotel is needed, you must still populate all fields.
- 'recommendation_reason' must clearly explain the situation.

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

    IMPORTANT:
   - You MUST NEVER ask the user for missing information.
    - If any critical information (like origin city or dates) is missing, ALWAYS assume a default.
    - The default departure city is 'Los Angeles' unless otherwise specified.
    - If no dates are provided, assume travel will occur approximately one month from today for a standard duration of 5 days.
    - Your response MUST always be in valid JSON following the FlightRecommendation schema, even if you have to make assumptions.
    - Never ask follow-up questions. Proceed based on reasonable defaults.

    
    Follow these steps to find the best flight for the user's trip:
    1. Extract the key information from the user's request (origin, destination, dates, budget, flight type)
    2. Use the `flight_finder` tool to find the best flight for the user's trip.
        - The best flight is defined as the flight that has the lowest price and the most direct route.
    3. If the user does not mention any dates (fuzzy or otherwise), assume the trip is approximately 1 month from today, with a duration of 5 days.
    4. If the user does not mention a budget, find the cheapest flight with the most direct route.
    5. If the user does not mention a preference for direct vs connecting flights, assume they prefer a direct flight. Prioritize recommending non-stop flights whenever possible. If no direct flights are available, or if connecting flights are significantly cheaper, clearly explain the tradeoff to the user.

        Structure your flight recommendation response with:
        - Airline Name
        - Flight Route (Origin to Destination)
        - Departure and Return Dates
        - Price
        - Whether it is a Direct Flight
        - Short Reasoning for Recommendation
    """,
    model=model,
    output_type=FlightRecommendation,
    tools=[flight_finder]
)

# -- Travel Agent Object (And Overall Orchestrator) --

travel_agent = Agent(
    name="Travel Agent",
    instructions="""
You are a comprehensive travel agent responsible for planning end-to-end trips based on user requests.

You MUST respond with a valid JSON object matching the TravelPlan schema.  
No extra text, no greetings, no markdown formatting, no bullet points outside of the JSON structure.  
Only pure, valid JSON output.

=========================================
YOUR EXACT WORKFLOW:

1. ANALYZE USER REQUEST:
   - Extract origin city (default to "Los Angeles" if not provided)
   - Extract destination city (required)
   - Extract travel dates or general timeframe (e.g., "late June")
   - Extract trip duration in days
   - Extract total trip budget
   - Extract interests and preferred activities
   - Extract hotel preferences (e.g., location, amenities like pool)
   - Determine if this is a day trip (no hotel needed)

If any required field is missing, make a reasonable assumption and mention this in the "notes" field of the final plan.

2. CALL FLIGHT AGENT:
   - Call flight_agent with extracted information.
   - For day trips, set one_way=True if supported.
   - Use specific dates if provided, or estimate if necessary.
   - Always provide origin, destination, and intended travel dates.

3. ANALYZE FLIGHT RESULTS:
   - If flight was found successfully, extract its price and dates.
   - If the flight_agent reports an API error, use estimated prices:
     * Domestic flights: $300–500
     * International short-haul: $600–1000
     * International long-haul: $1000–1800
   - Use provided or estimated departure_date and return_date.

4. CALL HOTEL AGENT:
   - If it's not a day trip, call hotel_agent using flight dates.
   - Pass adjusted hotel budget after accounting for flight cost.
   - Include special requests (e.g., "hotel with pool").
   - If hotel search fails, still proceed with best estimates.

5. BUILD FINAL TRAVEL PLAN:
Return a JSON object with EXACTLY these fields:

{
  "destination": string,           // The city name (e.g., "Tokyo")
  "duration_days": integer,         // Trip length in days
  "dates": array of strings,        // List of dates or date range
  "budget": float,                  // Total trip budget
  "activities": array of strings,   // Plain-text list of activities
  "notes": string (optional)        // Notes about any assumptions
}

IMPORTANT:
- The "activities" array must be simple text descriptions (not objects).
- If assumptions are made (like guessing dates or budget estimates), mention them in "notes."
- You MUST use destination, dates, and price estimates derived from earlier steps.
- Always generate a complete TravelPlan even if some data is missing.

=========================================
ABSOLUTE RULES FOR OUTPUT:
- Output must be a **pure JSON object**.
- No introductions, no headers, no prose, no markdown.
- No text before or after the JSON.
- No bullet points outside the JSON fields.

When you are ready to respond:  
Immediately output the JSON object. No explanations.

=========================================
EXAMPLE RESPONSE:

{
  "destination": "Tokyo",
  "duration_days": 5,
  "dates": ["2024-06-25", "2024-06-30"],
  "budget": 5000.0,
  "activities": ["Visit historical temples", "Explore outdoor gardens", "Visit museums"],
  "notes": "Dates estimated based on late June request. Flight price estimated due to API timeout."
}
""",
    model=model,
    output_type=TravelPlan,
    handoffs=[flight_agent, hotel_agent],
    tools=[get_current_weather]
)

# ─── Runner and Result Classes ───────────────────────────────────────

class Runner:
    @staticmethod
    async def run(agent: Agent, input_text: str):
        outputs = []
        history = []

        # Step 1: Run the initial agent
        primary_output = await agent.run(input_text, history)
        outputs.append(primary_output)

        # Step 2: If agent has handoffs (i.e. other agents), run them too
        if agent.handoffs:
            for subagent in agent.handoffs:
                if isinstance(primary_output, TravelPlan) and subagent.name == "Hotel Agent":
                    # Build structured input for Hotel Agent
                    dates = primary_output.dates

                    # Try to parse fuzzy dates into real ISO format
                    if dates and isinstance(dates[0], str) and not re.match(r"\d{4}-\d{2}-\d{2}", dates[0]):
                        dep_date, ret_date = parse_date_range_fuzzy(dates, duration_days=primary_output.duration_days)
                    else:
                        dep_date = dates[0]
                        ret_date = dates[-1]

                    hotel_inputs = {
                        "destination": primary_output.destination,
                        "checkin_date": dep_date,
                        "checkout_date": ret_date,
                        "budget": primary_output.budget,
                        "preferences": ""
                    }

                    sub_output = await subagent.run(hotel_inputs, history)
                else:
                    sub_output = await subagent.run(input_text, history)

                outputs.append(sub_output)

        # Step 3: Return all outputs
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
        "I want to go to Tokyo for 5 days in late June on a budget of $5000. I love history and the outdoors. I want to stay in a hotel with a pool.",
        "I'm going to Paris for 3 days during the winter. I like to eat good food and explore local culture. I want to stay at a hotel near the Eiffel Tower.",
        "I'm going to San Francisco from Jan 1 to Jan 8. I like to hike and explore nature. Ideally near the Golden Gate Bridge.",
        "I'm thinking of going to Rome for a week in late Summer. I want to eat amazing food, see ruins, and stay somewhere walkable. Budget is $3000.",
        "I'm going to New York today for the day. I have a budget of $500. I want to eat great steak and see a Broadway show."
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
