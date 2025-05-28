import os
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from amadeus import Client, ResponseError
from dotenv import load_dotenv
from dotenv import load_dotenv


load_dotenv()

class HotelSearcher:
    """Enhanced hotel search with location preferences and smart features"""
    
    def __init__(self):
        """Initialize the hotel searcher with Amadeus API credentials"""
        client_id = os.getenv("AMADEUS_API_KEY")
        client_secret = os.getenv("AMADEUS_API_SECRET")
        
        if not client_id or not client_secret:
            raise ValueError("Missing Amadeus API credentials. Please set AMADEUS_API_KEY and AMADEUS_API_SECRET environment variables.")
        
        self.amadeus = Client(client_id=client_id, client_secret=client_secret)
        
        # Enhanced currency conversion rates
        self.currency_rates = {
            "USD": 1.0,
            "JPY": 0.0067,   # 1 JPY = ~0.0067 USD  
            "EUR": 1.08,     # 1 EUR = ~1.08 USD
            "GBP": 1.25,     # 1 GBP = ~1.25 USD
            "CAD": 0.74,     # 1 CAD = ~0.74 USD
            "AUD": 0.65,     # 1 AUD = ~0.65 USD
            "CNY": 0.14,     # 1 CNY = ~0.14 USD
            "KRW": 0.00075,  # 1 KRW = ~0.00075 USD
            "CHF": 1.10,     # 1 CHF = ~1.10 USD
            "SEK": 0.096,    # 1 SEK = ~0.096 USD
            "NOK": 0.094,    # 1 NOK = ~0.094 USD
        }
        
        # City code cache for faster lookups
        self.city_codes = {
            "tokyo": "TYO", "paris": "PAR", "rome": "ROM",
            "san francisco": "SFO", "new york": "NYC", "los angeles": "LAX",
            "london": "LON", "madrid": "MAD", "barcelona": "BCN",
            "berlin": "BER", "amsterdam": "AMS", "seoul": "SEL",
            "beijing": "BJS", "shanghai": "SHA"
        }

    def find_hotel(self, destination: str, checkin_date: str, checkout_date: str, 
                   budget_per_night: float, preferences: str = "") -> Dict:
        """
        Find the best hotel based on your criteria
        
        Args:
            destination: City name (e.g., "Tokyo", "Paris")
            checkin_date: Check-in date in YYYY-MM-DD format
            checkout_date: Check-out date in YYYY-MM-DD format  
            budget_per_night: Maximum budget per night in USD
            preferences: Location/amenity preferences (e.g., "near shibuya, pool")
            
        Returns:
            Dictionary with hotel details including name, price, amenities, etc.
        """
        
        # Handle same-day trips (no hotel needed)
        if checkin_date == checkout_date:
            return {
                "name": "",
                "destination": destination,
                "checkin_date": checkin_date,
                "checkout_date": checkout_date,
                "price_per_night": 0.0,
                "amenities": [],
                "recommendation_reason": "No hotel needed for same-day trip."
            }
        
        try:
            print(f"🏨 Searching hotels in {destination} from {checkin_date} to {checkout_date}")
            print(f"💰 Budget: ${budget_per_night:.2f} per night (USD)")
            if preferences:
                print(f"📍 Preferences: {preferences}")
            
            # Get city code for the destination
            city_code = self._get_city_code(destination)
            if not city_code:
                raise ValueError(f"Could not find city code for '{destination}'")
            
            # Get list of hotels in the city
            hotels_list = self._get_hotels_in_city(city_code, destination)
            
            # Filter hotels by location preferences if specified
            if preferences.strip():
                hotels_list = self._filter_hotels_by_location(hotels_list, preferences)
            
            # Limit to 15 hotels to avoid rate limiting
            hotels_list = hotels_list[:15]
            hotel_ids = [hotel['hotelId'] for hotel in hotels_list]
            
            print(f"🔍 Checking {len(hotel_ids)} hotels for availability...")
            
            # Search for available hotels with rate limiting
            available_hotels = self._search_hotel_offers(
                hotel_ids, checkin_date, checkout_date, 
                destination, budget_per_night, preferences
            )
            
            if not available_hotels:
                return {
                    "name": "",
                    "destination": destination,
                    "checkin_date": checkin_date,
                    "checkout_date": checkout_date,
                    "price_per_night": 0.0,
                    "amenities": [],
                    "recommendation_reason": f"No hotels available for {checkin_date} to {checkout_date}. Try different dates."
                }
            
            # Find the best hotel based on our scoring system
            best_hotel = self._select_best_hotel(available_hotels, budget_per_night, preferences)
            
            print(f"🏆 SELECTED: {best_hotel['name']} at ${best_hotel['price_per_night']:.2f}/night USD")
            
            # Clean up and return the result
            return self._clean_hotel_result(best_hotel)
            
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

    def _get_city_code(self, city_name: str) -> Optional[str]:
        """Get IATA city code for a city name"""
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
            print(f"Error getting city code for {city_name}: {e}")
        
        return None

    def _get_hotels_in_city(self, city_code: str, destination: str) -> List[Dict]:
        """Get list of hotels in a city"""
        hotel_list_response = self.amadeus.reference_data.locations.hotels.by_city.get(
            cityCode=city_code
        )
        
        hotels_list = hotel_list_response.data
        if not hotels_list:
            raise ValueError(f"No hotels found for city {city_code}")
        
        print(f"✅ Found {len(hotels_list)} hotels in {destination}")
        return hotels_list

    def _filter_hotels_by_location(self, hotels_list: List[Dict], preferences: str) -> List[Dict]:
        """Filter hotels based on location preferences"""
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
            return filtered_hotels
        else:
            print(f"⚠️ No hotels found matching location '{preferences}', using all hotels")
            return hotels_list

    def _search_hotel_offers(self, hotel_ids: List[str], checkin_date: str, 
                           checkout_date: str, destination: str, 
                           budget_per_night: float, preferences: str) -> List[Dict]:
        """Search for hotel offers with rate limiting"""
        successful_searches = 0
        valid_hotels = []
        
        for i, hotel_id in enumerate(hotel_ids):
            try:
                # Rate limiting: Wait 150ms between requests
                if i > 0:
                    time.sleep(0.15)
                
                # Get hotel offers
                offers_response = self.amadeus.shopping.hotel_offers_search.get(
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
                
                # Get the best offer and process it
                hotel_result = self._process_hotel_offer(
                    hotel_info, offers, name, destination, 
                    checkin_date, checkout_date, 
                    budget_per_night, preferences
                )
                
                if hotel_result:
                    valid_hotels.append(hotel_result)
                    
                    # Stop early if we have enough good options
                    if len(valid_hotels) >= 5 and any(h["budget_friendly"] for h in valid_hotels):
                        print(f"    🎯 Found enough good options, stopping search early")
                        break
                
            except ResponseError as e:
                error_msg = str(e)
                if "429" in error_msg:
                    print(f"  ⏱️ Rate limit hit, waiting longer...")
                    time.sleep(0.5)
                elif "NO ROOMS AVAILABLE" in error_msg or "INVALID" in error_msg:
                    pass  # Common errors, don't spam
                else:
                    print(f"  ❌ API error for {hotel_id}: {e}")
                continue
            except Exception as e:
                print(f"  ❌ Error processing hotel {hotel_id}: {e}")
                continue
        
        print(f"✅ Successfully checked {successful_searches} hotels, found {len(valid_hotels)} with availability")
        return valid_hotels

    def _process_hotel_offer(self, hotel_info: Dict, offers: List[Dict], name: str,
                           destination: str, checkin_date: str, checkout_date: str,
                           budget_per_night: float, preferences: str) -> Optional[Dict]:
        """Process a single hotel offer"""
        
        # Get the cheapest offer
        best_offer = min(offers, key=lambda x: float(x.get("price", {}).get("total", float('inf'))))
        price_info = best_offer.get("price", {})
        raw_price = float(price_info.get("total", 0))
        response_currency = price_info.get("currency", None)
        
        # Calculate nights
        checkin_dt = datetime.strptime(checkin_date, "%Y-%m-%d")
        checkout_dt = datetime.strptime(checkout_date, "%Y-%m-%d")
        nights = max((checkout_dt - checkin_dt).days, 1)
        
        # Convert currency to USD
        usd_price_total, detected_currency = self._convert_currency(raw_price, destination, response_currency)
        usd_price_per_night = usd_price_total / nights
        
        print(f"  💱 {name}: {raw_price:.0f} {detected_currency} = ${usd_price_per_night:.2f} USD/night")
        
        # Skip if unreasonably expensive
        if usd_price_per_night > budget_per_night * 5:
            print(f"    ❌ Still too expensive after conversion")
            return None
        
        # Extract amenities and calculate scores
        amenities = self._extract_amenities(hotel_info, best_offer, name, destination)
        location_score = self._calculate_location_score(hotel_info, name, preferences)
        preferences_match = self._check_preferences_match(hotel_info, name, amenities, preferences)
        
        budget_status = "within budget" if usd_price_per_night <= budget_per_night else "over budget"
        location_status = f"(location score: {location_score})" if preferences else ""
        print(f"    ✅ {budget_status} {location_status}")
        
        return {
            "name": name,
            "destination": destination,
            "checkin_date": checkin_date,
            "checkout_date": checkout_date,
            "price_per_night": round(usd_price_per_night, 2),
            "amenities": amenities,
            "recommendation_reason": f"Found via Amadeus API (converted from {detected_currency})",
            "budget_friendly": usd_price_per_night <= budget_per_night,
            "preferences_match": preferences_match,
            "location_score": location_score,
            "original_price": f"{raw_price:.0f} {detected_currency}"
        }

    def _convert_currency(self, price: float, destination: str, price_currency: str = None) -> Tuple[float, str]:
        """Convert price to USD with smart currency detection"""
        
        # If currency is explicitly provided
        if price_currency and price_currency in self.currency_rates:
            usd_price = price * self.currency_rates[price_currency]
            return usd_price, price_currency
        
        # Smart detection based on destination and price range
        destination_lower = destination.lower()
        
        # Japan - prices typically in Yen (high numbers)
        if any(city in destination_lower for city in ["tokyo", "osaka", "kyoto", "japan", "hiroshima", "nagoya"]):
            if price > 5000:
                usd_price = price * self.currency_rates["JPY"]
                return usd_price, "JPY"
        
        # Europe - prices typically in Euros
        elif any(city in destination_lower for city in ["paris", "rome", "madrid", "berlin", "amsterdam", "barcelona", "milan", "vienna", "prague"]):
            if 50 < price < 2000:
                usd_price = price * self.currency_rates["EUR"]
                return usd_price, "EUR"
        
        # UK - prices typically in Pounds
        elif any(city in destination_lower for city in ["london", "manchester", "edinburgh", "glasgow", "birmingham"]):
            if 50 < price < 2000:
                usd_price = price * self.currency_rates["GBP"]
                return usd_price, "GBP"
        
        # Korea - prices typically in Won (very high numbers)
        elif any(city in destination_lower for city in ["seoul", "korea", "busan", "incheon"]):
            if price > 50000:
                usd_price = price * self.currency_rates["KRW"]
                return usd_price, "KRW"
        
        # China - prices typically in Yuan
        elif any(city in destination_lower for city in ["beijing", "shanghai", "china", "guangzhou", "shenzhen"]):
            if 200 < price < 5000:
                usd_price = price * self.currency_rates["CNY"]
                return usd_price, "CNY"
        
        # If price seems reasonable for USD, assume it's already USD
        if 50 <= price <= 1000:
            return price, "USD"
        
        # Default fallbacks based on price range
        elif price > 50000:
            usd_price = price * self.currency_rates["KRW"]
            return usd_price, "KRW (estimated)"
        elif price > 5000:
            usd_price = price * self.currency_rates["JPY"]
            return usd_price, "JPY (estimated)"
        
        return price, "USD (assumed)"

    def _extract_amenities(self, hotel_info: Dict, offer_info: Dict, hotel_name: str, destination: str) -> List[str]:
        """Extract and enhance hotel amenities"""
        amenities = []
        
        # Try multiple sources for amenities
        sources = [
            hotel_info.get("amenities", []),
            offer_info.get("room", {}).get("amenities", []),
        ]
        
        for source in sources:
            if isinstance(source, list):
                amenities.extend([str(item) for item in source if item])
        
        # Smart amenity guessing based on hotel name and destination
        name_lower = hotel_name.lower()
        dest_lower = destination.lower()
        
        # Brand-based amenities
        if any(brand in name_lower for brand in ["hilton", "marriott", "sheraton", "intercontinental", "hyatt"]):
            amenities.extend(["WiFi", "Fitness Center", "Business Center", "Concierge"])
        
        if any(term in name_lower for term in ["resort", "spa", "luxury"]):
            amenities.extend(["Spa", "Pool", "Room Service"])
        
        if any(term in name_lower for term in ["business", "conference", "convention"]):
            amenities.extend(["Business Center", "Meeting Rooms", "WiFi"])
        
        # Destination-specific amenities
        if any(city in dest_lower for city in ["tokyo", "japan"]):
            amenities.extend(["WiFi", "Air Conditioning"])
        elif any(city in dest_lower for city in ["paris", "rome", "london"]):
            amenities.extend(["WiFi", "Concierge"])
        elif any(city in dest_lower for city in ["miami", "california", "hawaii"]):
            amenities.extend(["Pool", "WiFi"])
        
        # Remove duplicates and limit
        return list(set(amenities))[:8]

    def _calculate_location_score(self, hotel_info: Dict, name: str, preferences: str) -> int:
        """Calculate location score based on preferences"""
        if not preferences.strip():
            return 0
        
        location_score = 0
        location_keywords = [word.strip().lower() for word in preferences.split() if len(word.strip()) > 2]
        
        description = hotel_info.get("description", {}).get("text", "").lower()
        searchable_text = f"{description} {name.lower()}"
        
        for keyword in location_keywords:
            if keyword in name.lower():
                location_score += 50  # High score for name match
            elif keyword in searchable_text:
                location_score += 20  # Lower score for description match
        
        return location_score

    def _check_preferences_match(self, hotel_info: Dict, name: str, amenities: List[str], preferences: str) -> bool:
        """Check if hotel matches general preferences"""
        if not preferences.strip():
            return True
        
        preference_keywords = [p.strip().lower() for p in preferences.split(",") if p.strip()]
        description = hotel_info.get("description", {}).get("text", "").lower()
        searchable_text = f"{description} {name.lower()} {' '.join(amenities).lower()}"
        
        return any(keyword in searchable_text for keyword in preference_keywords)

    def _select_best_hotel(self, hotels: List[Dict], budget_per_night: float, preferences: str) -> Dict:
        """Select the best hotel using our scoring system"""
        
        def hotel_score(hotel):
            score = 0
            # Location preference gets highest priority
            score += hotel["location_score"]
            # General preferences match
            if hotel["preferences_match"]:
                score += 100
            # Budget friendly gets points
            if hotel["budget_friendly"]:
                score += 50
                # Prefer cheaper hotels within budget
                score += (budget_per_night - hotel["price_per_night"]) / budget_per_night * 20
            # Bonus for having amenities
            score += len(hotel["amenities"]) * 5
            return score
        
        hotels.sort(key=hotel_score, reverse=True)
        best_hotel = hotels[0]
        
        # Update recommendation reason
        if best_hotel["location_score"] > 0 and best_hotel["preferences_match"] and best_hotel["budget_friendly"]:
            best_hotel["recommendation_reason"] = "Perfect match: within budget, matches preferences, and great location!"
        elif best_hotel["location_score"] > 0 and best_hotel["budget_friendly"]:
            first_pref = preferences.split()[0] if preferences else "preferred area"
            best_hotel["recommendation_reason"] = f"Excellent match: within budget and great location near {first_pref}!"
        elif best_hotel["location_score"] > 0:
            first_pref = preferences.split()[0] if preferences else "preferred area"
            best_hotel["recommendation_reason"] = f"Good location match near {first_pref}"
        elif best_hotel["preferences_match"] and best_hotel["budget_friendly"]:
            best_hotel["recommendation_reason"] = "Great match: within budget and matches preferences!"
        elif best_hotel["budget_friendly"]:
            best_hotel["recommendation_reason"] = "Good match: within budget"
        else:
            best_hotel["recommendation_reason"] = "Best available option found"
        
        return best_hotel

    def _clean_hotel_result(self, hotel: Dict) -> Dict:
        """Clean up hotel result for final output"""
        return {k: v for k, v in hotel.items() 
                if k not in ["budget_friendly", "preferences_match", "location_score", "original_price"]}


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
@function_tool
def hotel_finder_tool(destination: str, checkin_date: str, checkout_date: str, budget: float, preferences: str = "") -> dict:
    """Faster hotel finder with limits"""

    searcher = HotelSearcher()

    # Same-day trip check
    if checkin_date == checkout_date:
        return {
            "name": "", "destination": destination, "checkin_date": checkin_date,
            "checkout_date": checkout_date, "price_per_night": 0.0,
            "amenities": [], "recommendation_reason": "No hotel needed for same-day trip."
        }

    try:
        # ✅ Use class method
        city_code = searcher._get_city_code(destination)
        if not city_code:
            raise ValueError(f"Could not resolve destination '{destination}'")

        # ✅ Use the class's Amadeus instance
        hotel_list_response = searcher.amadeus.reference_data.locations.hotels.by_city.get(
            cityCode=city_code
        )
        hotels_list = hotel_list_response.data[:8]  # limit to 8 hotels

        if not hotels_list:
            raise ValueError("No hotels found")

        # ✅ Check only first 5 hotels
        for i, hotel in enumerate(hotels_list[:5]):
            try:
                offers_response = searcher.amadeus.shopping.hotel_offers_search.get(
                    hotelIds=hotel['hotelId'],
                    checkInDate=checkin_date,
                    checkOutDate=checkout_date,
                    adults=1,
                    roomQuantity=1,
                    currency="USD"
                )

                if offers_response.data and offers_response.data[0].get('offers'):
                    hotel_data = offers_response.data[0]
                    offer = hotel_data['offers'][0]

                    nights = max((datetime.strptime(checkout_date, "%Y-%m-%d") -
                                  datetime.strptime(checkin_date, "%Y-%m-%d")).days, 1)

                    price_total = float(offer.get('price', {}).get('total', 0))
                    price_per_night = price_total / nights

                    return {
                        "name": hotel_data.get('hotel', {}).get('name', 'Hotel Found'),
                        "destination": destination,
                        "checkin_date": checkin_date,
                        "checkout_date": checkout_date,
                        "price_per_night": round(price_per_night, 2),
                        "amenities": hotel_data.get('hotel', {}).get('amenities', [])[:3],
                        "recommendation_reason": "Found via fast search (first available)"
                    }

            except Exception:
                continue

        # Fallback
        estimated_price = min(budget * 0.4 / max((datetime.strptime(checkout_date, "%Y-%m-%d") -
                                                  datetime.strptime(checkin_date, "%Y-%m-%d")).days, 1), 150)

        return {
            "name": f"Estimated Hotel in {destination}",
            "destination": destination,
            "checkin_date": checkin_date,
            "checkout_date": checkout_date,
            "price_per_night": round(estimated_price, 2),
            "amenities": ["WiFi", "Air Conditioning"],
            "recommendation_reason": "API unavailable - estimated based on budget"
        }

    except Exception as e:
        return {
            "name": "", "destination": destination, "checkin_date": checkin_date,
            "checkout_date": checkout_date, "price_per_night": 0.0,
            "amenities": [], "recommendation_reason": f"Search failed: {str(e)}"
        }

# Example usage
if __name__ == "__main__":
    # Example 1: Using the class directly (for standalone use)
    print("=== Class-based usage ===")
    searcher = HotelSearcher()
    result = searcher.find_hotel(
        destination="Tokyo",
        checkin_date="2025-06-25",
        checkout_date="2025-06-30", 
        budget_per_night=200.0,
        preferences="near shibuya, pool"
    )
    print(f"Hotel: {result['name']}")
    print(f"Price: ${result['price_per_night']}/night")
    
    # Example 2: Using the function wrapper (for agent compatibility)
    print("\n=== Function-based usage (agent compatible) ===")
    result2 = hotel_finder_tool(
        destination="Tokyo",
        checkin_date="2025-06-25",
        checkout_date="2025-06-30",
        budget=1000.0,  # Total budget (will be divided by nights)
        preferences="near shibuya, pool"
    )
    print(f"Hotel: {result2['name']}")
    print(f"Price: ${result2['price_per_night']}/night")