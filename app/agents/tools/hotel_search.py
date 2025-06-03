from datetime import datetime
from typing import List, Dict, Optional
from app.config import AMA_CLIENT as amadeus

class HotelSearcher:
    """Strict hotel search using Amadeus API only – no fallback estimates."""

    def __init__(self):
        if not amadeus:
            raise ValueError(
                "Missing Amadeus API credentials. "
                "Please set AMA_CLIENT in app.config via your .env."
            )
        # Reuse the central client
        self.amadeus = amadeus

        # ← THIS MUST BE INSIDE __init__ (4 spaces in from margin)
        self.city_codes = {
            "tokyo": "TYO", "paris": "PAR", "rome": "ROM",
            "san francisco": "SFO", "new york": "NYC", "los angeles": "LAX"
        }

    def find_hotel(self,
                   destination: str,
                   checkin_date: str,
                   checkout_date: str,
                   total_budget: float,
                   preferences: str = "",
                   landmark_hint: str = "") -> dict:

        if checkin_date == checkout_date:
            return self._empty_response(
                destination,
                checkin_date,
                checkout_date,
                "No hotel needed for same-day trip."
            )

        nights = max(
            (datetime.strptime(checkout_date, "%Y-%m-%d") -
             datetime.strptime(checkin_date, "%Y-%m-%d")).days,
            1
        )
        budget_per_night = total_budget / nights

        # Handle location-based search if landmark_hint is provided
        if landmark_hint:
            geo = self._geocode_landmark(landmark_hint, destination)
            if geo:
                hotel_list = self._get_hotels_by_geocode(
                    geo["latitude"], geo["longitude"]
                )
            else:
                city_code = self._get_city_code(destination)
                if not city_code:
                    return self._empty_response(
                        destination, checkin_date, checkout_date,
                        f"Could not resolve city code for '{destination}'."
                    )
                hotel_list = self._get_hotels(city_code)
        else:
            city_code = self._get_city_code(destination)
            if not city_code:
                return self._empty_response(
                    destination, checkin_date, checkout_date,
                    f"Could not resolve city code for '{destination}'."
                )
            hotel_list = self._get_hotels(city_code)

        if not hotel_list:
            return self._empty_response(
                destination, checkin_date, checkout_date,
                f"No hotels found for {destination}"
            )

        preference_keywords = [
            p.strip().lower() for p in preferences.split(",") if p.strip()
        ]
        successful_hotels = []

        # Iterate through up to 30 candidates
        for hotel in hotel_list[:30]:
            hotel_id = hotel.get("hotelId")
            if not hotel_id:
                continue
            try:
                offers_response = self.amadeus.shopping.hotel_offers_search.get(
                    hotelIds=hotel_id,
                    checkInDate=checkin_date,
                    checkOutDate=checkout_date,
                    adults=1,
                    roomQuantity=1,
                    currency="USD"
                ).data

                if not offers_response or not offers_response[0].get("offers"):
                    continue

                hotel_info = offers_response[0]["hotel"]
                lowest_offer = min(
                    offers_response[0]["offers"],
                    key=lambda x: float(x["price"]["total"])
                )
                total_price = float(lowest_offer["price"]["total"])
                currency = lowest_offer["price"]["currency"]
                price_per_night = self._convert_currency(
                    total_price / nights,
                    currency
                )

                name = hotel_info.get("name", "Unknown Hotel")
                amenities = hotel_info.get("amenities", [])
                description = hotel_info.get("description", {}).get(
                    "text", ""
                ) or ""

                searchable = " ".join([
                    name.lower(),
                    description.lower(),
                    " ".join(amenities).lower()
                ])
                preferences_match = (
                    any(k in searchable for k in preference_keywords)
                    if preference_keywords else True
                )

                successful_hotels.append({
                    "hotelId": hotel_id,
                    "name": name,
                    "destination": destination,
                    "checkin_date": checkin_date,
                    "checkout_date": checkout_date,
                    "price_per_night": round(price_per_night, 2),
                    "amenities": amenities,
                    "original_currency": currency,
                    "within_budget": price_per_night <= budget_per_night,
                    "preferences_match": preferences_match,
                    "recommendation_reason": f"Price: {total_price} {currency}",
                    "latitude": hotel.get("latitude"),
                    "longitude": hotel.get("longitude"),
                    "address": hotel.get("address", {})
                })

                if len(successful_hotels) >= 5:
                    break

            except Exception:
                continue

        if not successful_hotels:
            return self._empty_response(
                destination,
                checkin_date,
                checkout_date,
                "No hotels with available rooms found."
            )

        best = self._select_best(successful_hotels, budget_per_night)
        return self._clean(best)

    def _get_city_code(self, city: str) -> Optional[str]:
        key = city.lower().strip()
        if key in self.city_codes:
            return self.city_codes[key]
        res = self.amadeus.reference_data.locations.get(
            keyword=city, subType="CITY"
        )
        if res.data:
            code = res.data[0].get("iataCode")
            self.city_codes[key] = code
            return code
        return None

    def _geocode_landmark(self,
                          query: str,
                          city_hint: str = "") -> Optional[Dict[str, float]]:
        """Use Amadeus location API to geocode a landmark name."""
        try:
            full_query = f"{query} {city_hint}".strip()
            res = self.amadeus.reference_data.locations.get(
                keyword=full_query,
                subType="AIRPORT,CITY,POINT_OF_INTEREST"
            )
            if res.data:
                geo = res.data[0].get("geoCode")
                if geo:
                    return {
                        "latitude": geo["latitude"],
                        "longitude": geo["longitude"]
                    }
        except Exception as e:
            print(f"❌ Failed to geocode landmark '{query}': {e}")
        return None

    def _get_hotels_by_geocode(self,
                               lat: float,
                               lon: float,
                               radius_km: int = 8) -> List[Dict]:
        try:
            res = self.amadeus.reference_data.locations.hotels.by_geocode.get(
                latitude=lat,
                longitude=lon,
                radius=radius_km,
                radiusUnit="KM"
            )
            # Flatten each entry to the inner "hotel" dict
            return [entry["hotel"] for entry in res.data] if res.data else []
        except Exception as e:
            print(f"❌ Failed to fetch hotels by geocode: {e}")
            return []

    def _get_hotels(self, city_code: str) -> List[Dict]:
        res = self.amadeus.reference_data.locations.hotels.by_city.get(
            cityCode=city_code
        )
        return [entry["hotel"] for entry in res.data] if res.data else []

    def _convert_currency(self, price: float, currency: str) -> float:
        rates = {"JPY": 0.0067, "EUR": 1.08, "GBP": 1.25}
        return price * rates.get(currency, 1.0)

    def _select_best(self,
                     hotels: List[Dict],
                     budget_per_night: float) -> Dict:
        def score(h):
            s = 0
            if h["preferences_match"]:
                s += 100
            if h["within_budget"]:
                s += 50
                s += (budget_per_night - h["price_per_night"]) / budget_per_night * 20
            return s

        hotels.sort(key=score, reverse=True)
        best = hotels[0]
        reason = (
            "Perfect match"
            if best["preferences_match"] and best["within_budget"]
            else best["recommendation_reason"]
        )
        best["recommendation_reason"] = reason
        return best

    def _clean(self, h: Dict) -> Dict:
        return {
            "name": h["name"],
            "destination": h["destination"],
            "checkin_date": h["checkin_date"],
            "checkout_date": h["checkout_date"],
            "price_per_night": h["price_per_night"],
            "amenities": h["amenities"],
            "recommendation_reason": h["recommendation_reason"]
        }

    def _empty_response(self,
                        destination: str,
                        checkin: str,
                        checkout: str,
                        reason: str) -> Dict:
        return {
            "name": "",
            "destination": destination,
            "checkin_date": checkin,
            "checkout_date": checkout,
            "price_per_night": 0.0,
            "amenities": [],
            "recommendation_reason": reason
        }


# decorator to expose as OpenAI tool
def function_tool(func):
    func.openai_schema = {
        "name": func.__name__,
        "description": func.__doc__ or "",
        "parameters": {
            "type": "object",
            "properties": {
                "destination":   {"type": "string"},
                "checkin_date":  {"type": "string"},
                "checkout_date": {"type": "string"},
                "budget":        {"type": "number"},
                "preferences":   {"type": "string"},
                "landmark_hint": {"type": "string"}
            },
            "required": ["destination", "checkin_date", "checkout_date", "budget"]
        }
    }
    return func

@function_tool
def hotel_finder_tool(destination: str,
                      checkin_date: str,
                      checkout_date: str,
                      budget: float,
                      preferences: str = "",
                      landmark_hint: str = "") -> dict:
    """Find hotels using ONLY real Amadeus data."""
    return HotelSearcher().find_hotel(
        destination, checkin_date, checkout_date, budget, preferences, landmark_hint
    )
