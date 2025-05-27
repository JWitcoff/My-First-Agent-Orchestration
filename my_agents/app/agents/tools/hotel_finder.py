from datetime import datetime
from amadeus import Client, ResponseError
import os
from dotenv import load_dotenv

load_dotenv()

# Currency conversion rates (approximate)
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
    Detect likely currency based on price and destination, then convert to USD.
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

def get_amadeus_client():
    """Get configured Amadeus client."""
    client_id = os.getenv("AMADEUS_API_KEY")
    client_secret = os.getenv("AMADEUS_API_SECRET")
    
    if not client_id or not client_secret:
        raise ValueError("Missing Amadeus API credentials in environment variables.")
    
    return Client(client_id=client_id, client_secret=client_secret)

def get_city_code(city_name: str):
    """Find the IATA city code for a given city name using Amadeus API."""
    # City code cache for faster lookups
    CITY_CODE_CACHE = {
        "tokyo": "TYO",
        "paris": "PAR",
        "rome": "ROM",
        "san francisco": "SFO",
        "new york": "NYC",
        "los angeles": "LAX",
    }
    
    city_name_lower = city_name.strip().lower()
    if city_name_lower in CITY_CODE_CACHE:
        return CITY_CODE_CACHE[city_name_lower]

    try:
        amadeus = get_amadeus_client()
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

def function_tool(func):
    """Decorator to make functions compatible with OpenAI function calling."""
    func.openai_schema = {
        "name": func.__name__,
        "description": func.__doc__ or "",
        "parameters": {}
    }
    return func

@function_tool  
def hotel_finder_tool(destination: str, checkin_date: str, checkout_date: str, budget: float, preferences: str = "") -> dict:
    """Enhanced hotel finder with location preferences and currency conversion using Amadeus API."""
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

        # Get Hotel List by City using Amadeus API
        amadeus = get_amadeus_client()
        hotel_list_response = amadeus.reference_data.locations.hotels.by_city.get(
            cityCode=city_code
        )
        
        hotels_list = hotel_list_response.data
        if not hotels_list:
            raise ValueError(f"No hotels found for city {city_code}")

        print(f"✅ Found {len(hotels_list)} hotels in {destination}")
        
        # Filter hotels by location preference if specified
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

        # Search Hotel Offers using Amadeus API
        successful_searches = 0
        valid_hotels = []

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
                    currency="USD"  # Request USD but may not work for all locations
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
                
                # ✅ Currency detection and conversion
                usd_price_total, detected_currency = detect_and_convert_currency(
                    raw_price, destination, response_currency
                )
                usd_price_per_night = usd_price_total / nights
                
                print(f"  💱 {name}: {raw_price:.0f} {detected_currency} = ${usd_price_per_night:.2f} USD/night")
                
                # Skip if unreasonably expensive (likely conversion error)
                if usd_price_per_night > budget_per_night * 5:  # More than 5x budget
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

        # ✅ Enhanced sorting with location preference
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
        
        # Clean up the result (remove internal scoring fields)
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

# Alternative simpler function for direct API usage (non-OpenAI function calling)
def find_hotels(destination: str, checkin_date: str, checkout_date: str, budget: float, preferences: str = "") -> dict:
    """
    Simple hotel finder function that returns the same result as hotel_finder_tool
    but without OpenAI function calling decoration.
    """
    return hotel_finder_tool(destination, checkin_date, checkout_date, budget, preferences)