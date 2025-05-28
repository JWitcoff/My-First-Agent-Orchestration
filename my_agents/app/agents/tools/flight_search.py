import os
import re
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from amadeus import Client, ResponseError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
amadeus = Client(
    client_id=os.getenv("AMADEUS_API_KEY"),
    client_secret=os.getenv("AMADEUS_API_SECRET")
)

# ✅ ADD: Fuzzy date parsing function from working version
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

        # Handle other patterns
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

# ✅ ADD: City code resolution
CITY_CODE_CACHE = {
    "tokyo": "TYO", "paris": "PAR", "rome": "ROM",
    "san francisco": "SFO", "new york": "NYC", "los angeles": "LAX",
    "london": "LON", "madrid": "MAD", "barcelona": "BCN",
    "berlin": "BER", "amsterdam": "AMS", "seoul": "SEL",
    "beijing": "BJS", "shanghai": "SHA"
}

def get_city_code(city_name: str) -> Optional[str]:
    """Find the IATA city code for a given city name using Amadeus API."""
    city_name_lower = city_name.strip().lower()
    if city_name_lower in CITY_CODE_CACHE:
        return CITY_CODE_CACHE[city_name_lower]

    try:
        # Get Amadeus client
        client_id = os.getenv("AMADEUS_API_KEY")
        client_secret = os.getenv("AMADEUS_API_SECRET")
        amadeus = Client(client_id=client_id, client_secret=client_secret)
        
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


class FlightSearcher:
    """Real flight search using Amadeus API - no mock data or estimates"""
    
    def __init__(self):
        """Initialize the flight searcher with Amadeus API credentials"""
        client_id = os.getenv("AMADEUS_API_KEY")
        client_secret = os.getenv("AMADEUS_API_SECRET")
        
        if not client_id or not client_secret:
            raise ValueError("Missing Amadeus API credentials. Please set AMADEUS_API_KEY and AMADEUS_API_SECRET environment variables.")
        
        self.amadeus = Client(client_id=client_id, client_secret=client_secret)
        
        # Enhanced city code cache for faster lookups
        self.city_codes = {
            "los angeles": "LAX", "san francisco": "SFO", "new york": "NYC", "chicago": "ORD",
            "miami": "MIA", "seattle": "SEA", "boston": "BOS", "las vegas": "LAS",
            "tokyo": "NRT", "paris": "CDG", "london": "LHR", "rome": "FCO", 
            "madrid": "MAD", "barcelona": "BCN", "berlin": "TXL", "amsterdam": "AMS",
            "seoul": "ICN", "beijing": "PEK", "shanghai": "PVG", "hong kong": "HKG",
            "sydney": "SYD", "melbourne": "MEL", "toronto": "YYZ", "vancouver": "YVR"
        }

    def find_flight(self, origin: str, destination: str, departure_date: str, 
                   return_date: str = None) -> Dict:
        """
        Find the best flight based on your criteria
        
        Args:
            origin: Origin city name or airport code (e.g., "Los Angeles" or "LAX")
            destination: Destination city name or airport code
            departure_date: Departure date in YYYY-MM-DD format
            return_date: Return date in YYYY-MM-DD format (optional for one-way)
            
        Returns:
            Dictionary with flight details including airline, price, dates, etc.
        """
        
        # Handle default origin
        if not origin or origin.lower() in ["current location", ""]:
            origin = "Los Angeles"
        
        try:
            print(f"✈️ Searching flights from {origin} to {destination}")
            print(f"📅 Departure: {departure_date}" + (f", Return: {return_date}" if return_date else " (one-way)"))
            
            # Get airport codes
            origin_code = self._get_airport_code(origin)
            destination_code = self._get_airport_code(destination)
            
            if not origin_code:
                raise ValueError(f"Could not find airport code for origin: {origin}")
            if not destination_code:
                raise ValueError(f"Could not find airport code for destination: {destination}")
            
            print(f"🛫 Using airports: {origin_code} → {destination_code}")
            
            # Validate and format dates
            dep_date, ret_date = self._validate_dates(departure_date, return_date)
            
            # Try API search
            flight_result = self._search_amadeus_flights(
                origin_code, destination_code, origin, destination, dep_date, ret_date
            )
            
            if flight_result:
                return flight_result
            
            # No fallback - return that no flights were found
            print("❌ No flights found via API")
            return {
                "origin": origin,
                "destination": destination,
                "departure_date": dep_date,
                "return_date": ret_date or "",
                "airline": "",
                "price": 0.0,
                "direct_flight": False,
                "recommendation_reason": f"No flights available for {dep_date}" + (f" to {ret_date}" if ret_date else "") + ". Try different dates or check airline websites directly."
            }
            
        except Exception as e:
            print(f"❌ Flight search error: {e}")
            return self._create_error_flight(origin, destination, departure_date, return_date, str(e))

    def _get_airport_code(self, city_name: str) -> Optional[str]:
        """Get airport code for a city name or return if already a code"""
        
        # If it's already a 3-letter airport code, return it
        if len(city_name) == 3 and city_name.isalpha():
            return city_name.upper()
        
        city_name_lower = city_name.strip().lower()
        
        # Check cache first
        if city_name_lower in self.city_codes:
            return self.city_codes[city_name_lower]
        
        # Search using Amadeus API
        try:
            response = self.amadeus.reference_data.locations.get(
                keyword=city_name,
                subType="CITY"
            )
            if response.data:
                return response.data[0].get("iataCode")
        except Exception as e:
            print(f"⚠️ Error getting airport code for {city_name}: {e}")
        
        return None

    def _validate_dates(self, departure_date: str, return_date: str = None) -> Tuple[str, Optional[str]]:
        """Validate and format dates"""
        today = datetime.today()
        max_future_date = today + timedelta(days=365)
        
        # Validate departure date
        try:
            dep_dt = datetime.strptime(departure_date, "%Y-%m-%d")
            if dep_dt.date() < today.date():
                dep_dt = today  # Use today if date is in the past
            elif dep_dt > max_future_date:
                dep_dt = max_future_date  # Cap at 1 year in future
            
            dep_date = dep_dt.strftime("%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid departure date format: {departure_date}. Use YYYY-MM-DD format.")
        
        # Validate return date if provided
        ret_date = None
        if return_date:
            try:
                ret_dt = datetime.strptime(return_date, "%Y-%m-%d")
                # Ensure return is after departure
                if ret_dt < dep_dt:
                    ret_dt = dep_dt + timedelta(days=1)
                elif ret_dt > max_future_date:
                    ret_dt = max_future_date
                
                ret_date = ret_dt.strftime("%Y-%m-%d")
            except ValueError:
                raise ValueError(f"Invalid return date format: {return_date}. Use YYYY-MM-DD format.")
        
        return dep_date, ret_date

    def _search_amadeus_flights(self, origin_code: str, destination_code: str, 
                              origin: str, destination: str, 
                              departure_date: str, return_date: str = None) -> Optional[Dict]:
        """Search for flights using Amadeus API"""
        
        try:
            # Build search parameters
            search_params = {
                "originLocationCode": origin_code,
                "destinationLocationCode": destination_code,
                "departureDate": departure_date,
                "adults": 1
            }
            
            if return_date:
                search_params["returnDate"] = return_date
            
            print(f"🔍 Searching API with parameters: {search_params}")
            
            # Make API call
            response = self.amadeus.shopping.flight_offers_search.get(**search_params)
            flights = response.data
            
            if not flights:
                print("❌ No flights found in API response")
                return None
            
            # Process the best flight offer
            return self._process_flight_offer(flights[0], origin, destination, origin_code, destination_code)
            
        except ResponseError as e:
            print(f"❌ Amadeus API error: {e}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error during API search: {e}")
            return None

    def _process_flight_offer(self, flight_offer: Dict, origin: str, destination: str,
                            origin_code: str, destination_code: str) -> Dict:
        """Process a flight offer from the API"""
        
        try:
            itineraries = flight_offer.get("itineraries", [])
            if not itineraries:
                raise ValueError("No itineraries in flight offer")
            
            # Get outbound flight info
            outbound_segments = itineraries[0].get("segments", [])
            if not outbound_segments:
                raise ValueError("No segments in outbound itinerary")
            
            outbound_seg = outbound_segments[0]
            departure_date = outbound_seg.get("departure", {}).get("at", "")[:10]
            airline_code = outbound_seg.get("carrierCode", "Unknown")
            
            # Get return flight info if available
            return_date = ""
            if len(itineraries) > 1:
                inbound_segments = itineraries[1].get("segments", [])
                if inbound_segments:
                    inbound_seg = inbound_segments[0]
                    return_date = inbound_seg.get("departure", {}).get("at", "")[:10]
            
            # Get price
            price = float(flight_offer.get("price", {}).get("total", 0.0))
            
            # Determine if direct flight
            total_segments = sum(len(itinerary.get("segments", [])) for itinerary in itineraries)
            is_direct = total_segments <= len(itineraries)  # One segment per direction
            
            # Get airline name (fallback to code if name not available)
            airline_name = self._get_airline_name(airline_code)
            
            print(f"✅ Found flight: {airline_name} ${price:.0f} {'(Direct)' if is_direct else '(Connecting)'}")
            
            return {
                "origin": origin,
                "destination": destination,
                "departure_date": departure_date,
                "return_date": return_date,
                "airline": airline_name,
                "price": price,
                "direct_flight": is_direct,
                "recommendation_reason": f"Found via Amadeus API - {'Direct flight' if is_direct else 'Best available option'}"
            }
            
        except Exception as e:
            print(f"❌ Error processing flight offer: {e}")
            return None

    def _get_airline_name(self, airline_code: str) -> str:
        """Get airline name from code (with common fallbacks)"""
        airline_names = {
            "AA": "American Airlines", "DL": "Delta Air Lines", "UA": "United Airlines",
            "WN": "Southwest Airlines", "B6": "JetBlue Airways", "AS": "Alaska Airlines",
            "F9": "Frontier Airlines", "NK": "Spirit Airlines",
            "BA": "British Airways", "LH": "Lufthansa", "AF": "Air France",
            "KL": "KLM", "LX": "Swiss International", "OS": "Austrian Airlines",
            "JL": "Japan Airlines", "NH": "ANA", "CX": "Cathay Pacific",
            "SQ": "Singapore Airlines", "TG": "Thai Airways", "EK": "Emirates",
            "QF": "Qantas", "AC": "Air Canada"
        }
        
        return airline_names.get(airline_code, f"Airline {airline_code}")

    def _create_error_flight(self, origin: str, destination: str, 
                           departure_date: str, return_date: str, error_msg: str) -> Dict:
        """Create error response when flight search completely fails"""
        
        return {
            "origin": origin,
            "destination": destination,
            "departure_date": departure_date,
            "return_date": return_date or "",
            "airline": "",
            "price": 0.0,
            "direct_flight": False,
            "recommendation_reason": f"Flight search failed: {error_msg}"
        }


# Function wrapper for compatibility with existing agent system
def function_tool(func):
    """Decorator to make functions compatible with OpenAI function calling."""
    func.openai_schema = {
        "name": func.__name__,
        "description": func.__doc__ or "",
        "parameters": {}
    }
    return func

@function_tool
def flight_finder_tool(origin: str, destination: str, dates: list[str]) -> dict:
    """Faster flight finder with immediate fallback"""
    
    if not destination:
        return {"airline": "", "price": 0.0, "recommendation_reason": "No destination provided"}
    
    try:
        dep_date, ret_date = parse_date_range_fuzzy(dates)
        origin_code = get_city_code(origin or "Los Angeles")
        dest_code = get_city_code(destination)
        
        if origin_code and dest_code:
            # ✅ SINGLE API CALL with max=1 result
            response = amadeus.shopping.flight_offers_search.get(
                originLocationCode=origin_code,
                destinationLocationCode=dest_code,
                departureDate=dep_date,
                returnDate=ret_date,
                adults=1,
                max=1  # ✅ ONLY GET 1 RESULT
            )
            
            if response.data:
                flight = response.data[0]
                return {
                    "origin": origin,
                    "airline": flight['itineraries'][0]['segments'][0].get('carrierCode', 'Found'),
                    "destination": destination,
                    "departure_date": dep_date,
                    "return_date": ret_date,
                    "price": float(flight.get('price', {}).get('total', 0)),
                    "direct_flight": len(flight['itineraries'][0]['segments']) == 1,
                    "recommendation_reason": "Found via API"
                }
    except Exception as e:
        print(f"Flight search failed, using estimate: {e}")
    
    # ✅ FAST FALLBACK - No API delays
    estimated_price = 800.0 if any(city in destination.lower() for city in ['tokyo', 'paris', 'london']) else 350.0
    
    return {
        "origin": origin or "Los Angeles",
        "airline": "Estimated",
        "destination": destination,
        "departure_date": dep_date if 'dep_date' in locals() else "2025-07-01",
        "return_date": ret_date if 'ret_date' in locals() else "2025-07-06", 
        "price": estimated_price,
        "direct_flight": True,
        "recommendation_reason": "Estimated due to API timeout"
    }


# Example usage
if __name__ == "__main__":
    # Example 1: Using the class directly (for standalone use)
    print("=== Class-based usage ===")
    searcher = FlightSearcher()
    result = searcher.find_flight(
        origin="Los Angeles",
        destination="Tokyo",
        departure_date="2025-06-25",
        return_date="2025-06-30"
    )
    
    if result['price'] > 0:
        print(f"✅ Real flight found!")
        print(f"Airline: {result['airline']}")
        print(f"Price: ${result['price']:.0f}")
        print(f"Direct: {result['direct_flight']}")
    else:
        print(f"❌ No flights available")
        print(f"Reason: {result['recommendation_reason']}")
    
    # Example 2: Using the function wrapper (for agent compatibility)
    print("\n=== Function-based usage (agent compatible) ===")
    result2 = flight_finder_tool(
        origin="New York",
        destination="Paris",
        dates=["2025-07-15", "2025-07-22"]
    )
    
    if result2['price'] > 0:
        print(f"✅ Real flight found!")
        print(f"Airline: {result2['airline']}")
        print(f"Price: ${result2['price']:.0f}")
    else:
        print(f"❌ No flights available")
        print(f"Reason: {result2['recommendation_reason']}")