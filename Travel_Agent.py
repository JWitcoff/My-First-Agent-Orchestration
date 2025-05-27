# ─── Standard Library Imports ─────────────────────────────────────
import os
import re
import json
import asyncio
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

# -- Load Environment Variables --
load_dotenv()

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
    max_future_date = today + timedelta(days=365)

    def enforce_valid_date_range(dep: datetime.date, ret: datetime.date):
        dep = max(dep, today.date())
        ret = max(ret, dep)
        dep = min(dep, max_future_date.date())
        ret = min(ret, max_future_date.date())
        return dep, ret

    # Handle empty input
    if not dates or len(dates) == 0:
        dep, ret = enforce_valid_date_range(today.date() + timedelta(days=30),
                                            today.date() + timedelta(days=30 + duration_days))
        return str(dep), str(ret)

    # Try to parse exact dates if there are two entries
    try:
        if len(dates) == 2:
            for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y", "%B %d, %Y", "%b %d, %Y"]:
                try:
                    dep = datetime.strptime(dates[0], fmt).date()
                    ret = datetime.strptime(dates[1], fmt).date()
                    dep, ret = enforce_valid_date_range(dep, ret)
                    return str(dep), str(ret)
                except ValueError:
                    continue
    except Exception:
        pass

    # Process single fuzzy term
    if len(dates) >= 1:
        term = dates[0].strip().lower()
        term = re.sub(r"[^a-z\s0-9]", "", term)

        # ✅ FIXED: Handle "last week of [month]" patterns
        if "last week of" in term:
            month_lookup = {
                "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
                "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
                "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
                "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12
            }
            
            for month_name, month_num in month_lookup.items():
                if month_name in term:
                    # Calculate last week of the month
                    year = today.year if today.month <= month_num else today.year + 1
                    
                    # Find last day of month
                    if month_num == 12:
                        last_day = datetime(year + 1, 1, 1) - timedelta(days=1)
                    else:
                        last_day = datetime(year, month_num + 1, 1) - timedelta(days=1)
                    
                    # Find the Monday of the last week
                    days_to_monday = (last_day.weekday() + 1) % 7
                    last_monday = last_day - timedelta(days=days_to_monday + 6)  # Previous Monday
                    
                    dep_raw = last_monday.date()
                    ret_raw = dep_raw + timedelta(days=duration_days)
                    
                    dep, ret = enforce_valid_date_range(dep_raw, ret_raw)
                    return str(dep), str(ret)

        # Handle other patterns (keeping existing logic)
        if "today" in term:
            dep, ret = enforce_valid_date_range(today.date(), today.date() + timedelta(days=duration_days))
            return str(dep), str(ret)

        if "tomorrow" in term:
            dep, ret = enforce_valid_date_range(today.date() + timedelta(days=1),
                                                today.date() + timedelta(days=1 + duration_days))
            return str(dep), str(ret)

        if "next week" in term:
            next_monday = today + timedelta(days=(7 - today.weekday()))
            dep, ret = enforce_valid_date_range(next_monday.date(), next_monday.date() + timedelta(days=duration_days))
            return str(dep), str(ret)

        if "next month" in term:
            if today.month == 12:
                dep_raw = datetime(today.year + 1, 1, 15).date()
            else:
                dep_raw = datetime(today.year, today.month + 1, 15).date()
            dep, ret = enforce_valid_date_range(dep_raw, dep_raw + timedelta(days=duration_days))
            return str(dep), str(ret)

        # Month detection (existing logic)
        month_lookup = {
            "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
            "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
            "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
            "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12
        }

        for word, month in month_lookup.items():
            if word in term:
                base_day = 15
                if "early" in term:
                    base_day = 5
                elif "mid" in term:
                    base_day = 15
                elif "late" in term or "end" in term:
                    base_day = 25

                year = today.year if today.month <= month else today.year + 1
                dep_raw = datetime(year, month, base_day).date()
                dep, ret = enforce_valid_date_range(dep_raw, dep_raw + timedelta(days=duration_days))
                return str(dep), str(ret)

        # Seasonal patterns (existing logic)
        seasonal = {
            "winter": ("january", 15),
            "spring": ("april", 15), 
            "summer": ("july", 15),
            "fall": ("october", 15),
            "autumn": ("october", 15)
        }

        for season, (month_word, day) in seasonal.items():
            if season in term:
                month = month_lookup[month_word]
                year = today.year if today.month <= month else today.year + 1
                dep_raw = datetime(year, month, day).date()
                dep, ret = enforce_valid_date_range(dep_raw, dep_raw + timedelta(days=duration_days))
                return str(dep), str(ret)

    # Final fallback
    dep, ret = enforce_valid_date_range(today.date() + timedelta(days=30),
                                        today.date() + timedelta(days=30 + duration_days))
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

        if not destination_code:
            print(f"Could not find city code for destination: {destination}")
         
            
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
                
            # Process the first flight offer
            flight = flights[0]
            itineraries = flight.get("itineraries", [])
            
            if len(itineraries) < 1:
                print("No itineraries in flight data")
            
            # Handle one-way trips vs round trips
            outbound_segments = itineraries[0].get("segments", [])
            if not outbound_segments:
                print("No segments in itinerary")
                
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
            print(f"❌ Amadeus API error during flight search: {e}")

    except Exception as e:
        print(f"❌ Unexpected error during flight search: {e}")


# -- Hotel Finder ───────────────────────────────────────────────────────
# ✅ Hotel Finder with Currency Detection and Conversion

# Currency conversion rates (approximate - you could use a real API for live rates)
CURRENCY_TO_USD = {
    "USD": 1.0,
    "JPY": 0.0067,  # 1 JPY = ~0.0067 USD  
    "EUR": 1.08,    # 1 EUR = ~1.08 USD
    "GBP": 1.25,    # 1 GBP = ~1.25 USD
    "CAD": 0.74,    # 1 CAD = ~0.74 USD
    "AUD": 0.65,    # 1 AUD = ~0.65 USD
    "CNY": 0.14,    # 1 CNY = ~0.14 USD
    "KRW": 0.00075, # 1 KRW = ~0.00075 USD
}

def detect_and_convert_currency(price: float, destination: str, price_currency: str = None) -> tuple[float, str]:
    """
    Detect likely currency based on price and destination, then convert to USD
    Returns: (usd_price, detected_currency)
    """
    
    # If currency is explicitly provided in the response
    if price_currency and price_currency in CURRENCY_TO_USD:
        usd_price = price * CURRENCY_TO_USD[price_currency]
        return usd_price, price_currency
    
    # Heuristic detection based on price range and destination
    destination_lower = destination.lower()
    
    # Japan - prices typically in Yen (high numbers)
    if any(city in destination_lower for city in ["tokyo", "osaka", "kyoto", "japan"]):
        if price > 5000:  # Likely Yen
            usd_price = price * CURRENCY_TO_USD["JPY"]
            return usd_price, "JPY"
    
    # Europe - prices typically in Euros  
    elif any(city in destination_lower for city in ["paris", "rome", "madrid", "berlin", "amsterdam"]):
        if 50 < price < 2000:  # Reasonable Euro range
            usd_price = price * CURRENCY_TO_USD["EUR"]
            return usd_price, "EUR"
    
    # UK - prices typically in Pounds
    elif any(city in destination_lower for city in ["london", "manchester", "edinburgh"]):
        if 50 < price < 2000:  # Reasonable GBP range
            usd_price = price * CURRENCY_TO_USD["GBP"] 
            return usd_price, "GBP"
    
    # Korea - prices typically in Won (very high numbers)
    elif any(city in destination_lower for city in ["seoul", "korea"]):
        if price > 50000:  # Likely Won
            usd_price = price * CURRENCY_TO_USD["KRW"]
            return usd_price, "KRW"
    
    # China - prices typically in Yuan
    elif any(city in destination_lower for city in ["beijing", "shanghai", "china"]):
        if 200 < price < 5000:  # Reasonable Yuan range
            usd_price = price * CURRENCY_TO_USD["CNY"]
            return usd_price, "CNY"
    
    # If price seems reasonable for USD (50-1000 range), assume it's already USD
    if 50 <= price <= 1000:
        return price, "USD"
    
    # If price is very high, guess it might be Yen
    elif price > 5000:
        usd_price = price * CURRENCY_TO_USD["JPY"] 
        return usd_price, "JPY (estimated)"
    
    # Default to USD if unsure
    return price, "USD (assumed)"

@function_tool  
def hotel_finder_tool(destination: str, checkin_date: str, checkout_date: str, budget: float, preferences: str = "") -> dict:
    """Enhanced hotel finder with better location filtering"""
    try:
        # Calculate nights and budget per night
        checkin_dt = datetime.strptime(checkin_date, "%Y-%m-%d")
        checkout_dt = datetime.strptime(checkout_date, "%Y-%m-%d")
        nights = max((checkout_dt - checkin_dt).days, 1)
        budget_per_night = budget / nights

        # Get city code for hotels
        city_code = get_city_code(destination)
        if not city_code:
            raise ValueError(f"Could not resolve destination '{destination}' to city code")

        print(f"🏨 Searching hotels in {destination} (code: {city_code}) from {checkin_date} to {checkout_date}")
        print(f"💰 Budget: ${budget_per_night:.2f} per night (USD)")
        if preferences:
            print(f"📍 Location preference: {preferences}")

        # Get Hotel List by City
        hotel_list_response = amadeus.reference_data.locations.hotels.by_city.get(
            cityCode=city_code
        )
        
        hotels_list = hotel_list_response.data
        if not hotels_list:
            raise ValueError(f"No hotels found for city {city_code}")

        print(f"✅ Found {len(hotels_list)} hotels in {destination}")
        
        # ✅ IMPROVED: Filter hotels by location preference if specified
        if preferences.strip():
            location_keywords = [word.strip().lower() for word in preferences.split() if len(word.strip()) > 2]
            filtered_hotels = []
            
            for hotel in hotels_list:
                hotel_name = hotel.get('name', '').lower()
                hotel_location = hotel.get('address', {}).get('cityName', '').lower()
                searchable_text = f"{hotel_name} {hotel_location}"
                
                # Check if any location keyword matches
                location_match = any(keyword in searchable_text for keyword in location_keywords)
                if location_match:
                    filtered_hotels.append(hotel)
            
            if filtered_hotels:
                print(f"🎯 Filtered to {len(filtered_hotels)} hotels matching location preference")
                hotels_list = filtered_hotels[:15]  # Take top 15 matching hotels
            else:
                print(f"⚠️ No hotels found matching '{preferences}', using all hotels")
                hotels_list = hotels_list[:20]  # Fallback to first 20
        else:
            hotels_list = hotels_list[:20]
        
        hotel_ids = [hotel['hotelId'] for hotel in hotels_list]
        
        if not hotel_ids:
            raise ValueError("No hotel IDs found")

        # Search Hotel Offers
        successful_searches = 0
        valid_hotels = []
        preference_keywords = [p.strip().lower() for p in preferences.split(",") if p.strip()]

        print(f"🔍 Checking {len(hotel_ids)} hotels for availability...")

        for i, hotel_id in enumerate(hotel_ids):
            try:
                # Try to get offers for this hotel
                offers_response = amadeus.shopping.hotel_offers_search.get(
                    hotelIds=hotel_id,
                    checkInDate=checkin_date,
                    checkOutDate=checkout_date,
                    adults=1,
                    roomQuantity=1,
                    currency="USD"
                )
                
                hotel_offers = offers_response.data
                if not hotel_offers:
                    continue
                
                successful_searches += 1
                
                # Process hotel data
                hotel_data = hotel_offers[0]
                hotel_info = hotel_data.get("hotel", {})
                name = hotel_info.get("name", "Unknown Hotel")
                offers = hotel_data.get("offers", [])
                
                if not offers:
                    continue
                
                # Get the cheapest offer
                best_offer = min(offers, key=lambda x: float(x.get("price", {}).get("total", float('inf'))))
                price_info = best_offer.get("price", {})
                raw_price = float(price_info.get("total", 0))
                response_currency = price_info.get("currency", None)
                
                # Currency detection and conversion
                usd_price_total, detected_currency = detect_and_convert_currency(
                    raw_price, destination, response_currency
                )
                usd_price_per_night = usd_price_total / nights
                
                print(f"  💱 {name}: {raw_price:.0f} {detected_currency} = ${usd_price_per_night:.2f} USD/night")
                
                # Skip if unreasonably expensive
                if usd_price_per_night > budget_per_night * 5:
                    print(f"    ❌ Too expensive after conversion")
                    continue
                
                # Extract amenities and check preferences
                amenities = hotel_info.get("amenities", [])
                description = hotel_info.get("description", {}).get("text", "").lower()
                searchable_text = f"{description} {name.lower()} {' '.join(amenities).lower()}"
                
                # Enhanced location scoring
                location_score = 0
                if preferences.strip():
                    location_keywords = [word.strip().lower() for word in preferences.split() if len(word.strip()) > 2]
                    for keyword in location_keywords:
                        if keyword in name.lower():
                            location_score += 50  # High score for name match
                        elif keyword in searchable_text:
                            location_score += 20  # Lower score for description match

                hotel_result = {
                    "name": name,
                    "destination": destination,
                    "checkin_date": checkin_date,
                    "checkout_date": checkout_date,
                    "price_per_night": round(usd_price_per_night, 2),
                    "amenities": amenities if isinstance(amenities, list) else [],
                    "recommendation_reason": f"Found via Amadeus API (converted from {detected_currency})",
                    "budget_friendly": usd_price_per_night <= budget_per_night,
                    "location_score": location_score,
                    "original_price": f"{raw_price:.0f} {detected_currency}"
                }
                
                valid_hotels.append(hotel_result)
                budget_status = "within budget" if usd_price_per_night <= budget_per_night else "over budget"
                location_status = f"(location score: {location_score})" if preferences else ""
                print(f"    ✅ {budget_status} {location_status}")
                
            except ResponseError as e:
                error_msg = str(e)
                if "NO ROOMS AVAILABLE" in error_msg or "INVALID" in error_msg:
                    pass  # Common errors, don't spam
                else:
                    print(f"  ❌ API error for {hotel_id}: {e}")
                continue
            except Exception as e:
                print(f"  ❌ Error processing hotel {hotel_id}: {e}")
                continue

        print(f"✅ Successfully checked {successful_searches} hotels, found {len(valid_hotels)} with availability")

        if not valid_hotels:
            return {
                "name": "",
                "destination": destination,
                "checkin_date": checkin_date,
                "checkout_date": checkout_date,
                "price_per_night": 0.0,
                "amenities": [],
                "recommendation_reason": f"No hotels available for {checkin_date} to {checkout_date}. Try different dates."
            }

        # ✅ IMPROVED: Enhanced sorting with location preference
        def hotel_score(hotel):
            score = 0
            # Location preference gets highest priority
            score += hotel["location_score"]
            # Budget friendly gets points
            if hotel["budget_friendly"]:
                score += 50
                # Prefer cheaper hotels within budget
                score += (budget_per_night - hotel["price_per_night"]) / budget_per_night * 20
            return score

        valid_hotels.sort(key=hotel_score, reverse=True)
        best_hotel = valid_hotels[0]

        # Update recommendation reason based on selection criteria
        if best_hotel["location_score"] > 0 and best_hotel["budget_friendly"]:
            best_hotel["recommendation_reason"] = f"Best match: within budget and close to {preferences}!"
        elif best_hotel["location_score"] > 0:
            best_hotel["recommendation_reason"] = f"Good location match near {preferences}"
        elif best_hotel["budget_friendly"]:
            best_hotel["recommendation_reason"] = "Good match: within budget"
        else:
            best_hotel["recommendation_reason"] = "Best available option found"

        print(f"🏆 SELECTED: {best_hotel['name']} at ${best_hotel['price_per_night']:.2f}/night USD")
        
        # Clean up the result
        result = {k: v for k, v in best_hotel.items() if k not in ["budget_friendly", "location_score", "original_price"]}
        return result

    except Exception as e:
        print(f"❌ Hotel search error: {e}")
        return {
            "name": "",
            "destination": destination,
            "checkin_date": checkin_date,
            "checkout_date": checkout_date,
            "price_per_night": 0.0,
            "amenities": [],
            "recommendation_reason": f"Hotel search failed: {str(e)}"
        }
    
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
        flight = None

        # Step 1: Run the initial agent
        primary_output = await agent.run(input_text, history)
        outputs.append(primary_output)

        # Step 2: Run handoff agents if any
        if agent.handoffs:
            for subagent in agent.handoffs:
                if isinstance(primary_output, TravelPlan) and subagent.name == "Flight Agent":
                    # Handle Flight Agent
                    sub_output = await subagent.run(input_text, history)
                    if isinstance(sub_output, FlightRecommendation):
                        flight = sub_output
                        # ✅ UPDATE: Immediately update TravelPlan dates with flight dates
                        primary_output.dates = [flight.departure_date, flight.return_date]
                    outputs.append(sub_output)
                    
                elif isinstance(primary_output, TravelPlan) and subagent.name == "Hotel Agent":
                    # ✅ FIXED: Use flight dates if available, otherwise parse from TravelPlan
                    if flight:
                        # Use the actual flight dates
                        dep_date = flight.departure_date
                        ret_date = flight.return_date
                    else:
                        # Fallback: parse dates from TravelPlan
                        dates = primary_output.dates
                        if dates and isinstance(dates[0], str) and not re.match(r"\d{4}-\d{2}-\d{2}", dates[0]):
                            dep_date, ret_date = parse_date_range_fuzzy(dates, duration_days=primary_output.duration_days)
                        else:
                            dep_date = dates[0] if dates else None
                            ret_date = dates[-1] if dates and len(dates) > 1 else dep_date

                    # ✅ FIXED: Extract location preferences from input_text
                    location_prefs = ""
                    if isinstance(input_text, str):
                        # Extract location preferences from original input
                        import re
                        location_patterns = [
                            r"near\s+([^,\.]+)",
                            r"close to\s+([^,\.]+)", 
                            r"around\s+([^,\.]+)",
                            r"in\s+([^,\.]+)\s+area",
                            r"located\s+([^,\.]+)"
                        ]
                        for pattern in location_patterns:
                            match = re.search(pattern, input_text.lower())
                            if match:
                                location_prefs = match.group(1).strip()
                                break
                    elif isinstance(input_text, dict) and 'locationPrefs' in input_text:
                        location_prefs = input_text['locationPrefs']

                    # Build hotel inputs with proper dates and preferences
                    hotel_inputs = {
                        "destination": primary_output.destination,
                        "checkin_date": dep_date,
                        "checkout_date": ret_date,
                        "budget": primary_output.budget * 0.4,  # Allocate 40% of budget to hotels
                        "preferences": location_prefs  # ✅ FIXED: Pass location preferences
                    }

                    sub_output = await subagent.run(hotel_inputs, history)
                    outputs.append(sub_output)
                else:
                    sub_output = await subagent.run(input_text, history)
                    outputs.append(sub_output)

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