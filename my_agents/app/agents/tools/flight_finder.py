# flight_finder.py - Updated with Advanced Features

# ─── Standard Library Imports ─────────────────────────────────────
import os
import re
from datetime import datetime, timedelta
from typing import Optional

# ─── Third-Party Library Imports ─────────────────────────────────
from amadeus import Client, ResponseError
from dotenv import load_dotenv

# -- Load Environment Variables --
load_dotenv()

# -- Amadeus API Setup --
client_id = os.getenv("AMADEUS_API_KEY")
client_secret = os.getenv("AMADEUS_API_SECRET")
if not client_id or not client_secret:
    raise ValueError("Missing Amadeus API credentials in environment variables.")

amadeus = Client(client_id=client_id, client_secret=client_secret)

# -- Helper: City Code Cache and Conversion --
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
            code = results[0].get("iataCode")
            # Cache for future use
            CITY_CODE_CACHE[city_name_lower] = code
            return code
        else:
            return None
    except Exception as e:
        print(f"Error getting city code for {city_name}: {e}")
        return None

# -- Helper: Fuzzy Date Parsing --
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

        # Handle "last week of [month]" patterns
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
                    last_monday = last_day - timedelta(days=days_to_monday + 6)
                    
                    dep_raw = last_monday.date()
                    ret_raw = dep_raw + timedelta(days=duration_days)
                    
                    dep, ret = enforce_valid_date_range(dep_raw, ret_raw)
                    return str(dep), str(ret)

        # Handle other common patterns
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

        # Month detection
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

        # Seasonal patterns
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

# -- Helper Function to Convert Functions to OpenAI-Compatible Tools --
def function_tool(func):
    """Wraps a function and attaches an OpenAI schema to it."""
    func.openai_schema = {
        "name": func.__name__,
        "description": func.__doc__ or "",
        "parameters": {}
    }
    return func

# -- Main Flight Finder Function --
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
    
    # Try the API call
    try:
        # Print debug information
        print(f"Searching flights: {origin} to {destination}, {dep_date} to {ret_date}")
        
        origin_code = get_city_code(origin) if len(origin) != 3 else origin.upper()
        destination_code = get_city_code(destination) if len(destination) != 3 else destination.upper()
        
        if not origin_code:
            print(f"Could not find city code for origin: {origin}")
            error_response["recommendation_reason"] = f"Could not find airport code for origin: {origin}"
            return error_response

        if not destination_code:
            print(f"Could not find city code for destination: {destination}")
            error_response["recommendation_reason"] = f"Could not find airport code for destination: {destination}"
            return error_response
            
        print(f"Using city codes: {origin_code} to {destination_code}")
        
        # Try API call
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
                error_response["recommendation_reason"] = "No flights available for these dates and destinations."
                return error_response
                
            # Process the first flight offer
            flight = flights[0]
            itineraries = flight.get("itineraries", [])
            
            if len(itineraries) < 1:
                print("No itineraries in flight data")
                error_response["recommendation_reason"] = "No flight itineraries available."
                return error_response
            
            # Handle one-way trips vs round trips
            outbound_segments = itineraries[0].get("segments", [])
            if not outbound_segments:
                print("No segments in itinerary")
                error_response["recommendation_reason"] = "No flight segments available."
                return error_response
                
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
            error_response["recommendation_reason"] = "Flight search failed due to API error. Please try different dates or destinations."
            return error_response

    except Exception as e:
        print(f"❌ Unexpected error during flight search: {e}")
        error_response["recommendation_reason"] = f"Flight search failed: {str(e)}"
        return error_response

# -- Alternative Simple Function (for non-Agent usage) --
def find_flights(params: dict) -> dict:
    """
    Simple wrapper function for direct usage (non-Agent systems).
    
    Args:
        params (dict): Dictionary with keys:
            - origin (str): Origin city name or code
            - destination (str): Destination city name or code  
            - dates (list): List of date strings (can be fuzzy)
            - duration_days (int, optional): Trip duration, defaults to 5
    
    Returns:
        dict: Flight recommendation in same format as flight_finder
    """
    origin = params.get("origin", "Los Angeles")
    destination = params.get("destination", "")
    dates = params.get("dates", [])
    
    return flight_finder(origin, destination, dates)

# -- Test Function --
def test_flight_finder():
    """Test the flight finder with sample data."""
    print("🧪 Testing Flight Finder...")
    
    # Test 1: Basic search
    result1 = flight_finder("Los Angeles", "Tokyo", ["next month"])
    print(f"Test 1 - LA to Tokyo: {result1['recommendation_reason']}")
    
    # Test 2: Fuzzy dates
    result2 = flight_finder("New York", "Paris", ["last week of June"])
    print(f"Test 2 - NYC to Paris: {result2['recommendation_reason']}")
    
    # Test 3: Error handling
    result3 = flight_finder("", "London", [])
    print(f"Test 3 - Error case: {result3['recommendation_reason']}")

if __name__ == "__main__":
    test_flight_finder()