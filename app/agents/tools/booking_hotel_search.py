import os
import requests
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

BOOKING_API_KEY = os.getenv("BOOKING_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not BOOKING_API_KEY:
    raise ValueError("Missing BOOKING_API_KEY in environment.")
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY in environment.")

RAPIDAPI_HOST = "booking-com15.p.rapidapi.com"

headers = {
    "X-RapidAPI-Key": BOOKING_API_KEY,
    "X-RapidAPI-Host": RAPIDAPI_HOST
}

class BookingHotelSearcher:
    def __init__(self):
        self.headers = headers

    def find_hotel(self,
                   destination: str,
                   checkin_date: str,
                   checkout_date: str,
                   total_budget: float,
                   preferences: str = "",
                   landmark_hint: str = "") -> Dict:

        lat, lon = None, None
        query = f"{landmark_hint} {destination}".strip()
        latlng = self._get_lat_lng_from_google(query)
        if latlng:
            lat, lon = latlng["lat"], latlng["lng"]
            print(f"📍 Using Google lat/lng for context: {lat}, {lon}")

        location_id = self._get_location_id(destination)
        if not location_id:
            return {"recommendation_reason": f"Could not resolve destination '{destination}'"}
        hotels = self._search_hotels_by_dest_id(location_id, checkin_date, checkout_date)


        if not hotels:
            return {"recommendation_reason": f"No hotels found in {destination}"}

        matches = []
        nights = max((datetime.strptime(checkout_date, "%Y-%m-%d") - datetime.strptime(checkin_date, "%Y-%m-%d")).days, 1)
        budget_per_night = total_budget / nights

        for hotel in hotels:
            price_info = hotel.get("compositePriceBreakdown", {}).get("grossAmount")
            if not price_info:
                continue

            price = float(price_info.get("value", 0))
            currency = price_info.get("currency")

            if price / nights > budget_per_night * 1.25:
                continue

            matches.append({
                "name": hotel.get("name"),
                "price_total": round(price, 2),
                "currency": currency,
                "review_score": hotel.get("reviewScore"),
                "address": hotel.get("location", {}).get("address", ""),
                "district": hotel.get("location", {}).get("district", ""),
                "latitude": hotel.get("location", {}).get("latitude"),
                "longitude": hotel.get("location", {}).get("longitude")
            })

            if len(matches) >= 3:
                break

        if not matches:
            return {"recommendation_reason": f"No hotels matched budget or preferences in {destination}"}

        return {"top_matches": matches}

    def _get_location_id(self, destination: str) -> Optional[str]:
        url = "https://booking-com15.p.rapidapi.com/api/v1/hotels/searchDestination"
        params = {"name": destination}
        try:
            response = requests.get(url, headers=self.headers, params=params)
            data = response.json()
            if data and isinstance(data, list):
                return data[0].get("dest_id")
        except Exception as e:
            print(f"❌ Location ID lookup failed: {e}")
        return None

    def _search_hotels_by_dest_id(self, location_id: str, checkin: str, checkout: str) -> List[Dict]:
        url = "https://booking-com15.p.rapidapi.com/api/v1/hotels/searchHotels"
        params = {
            "dest_id": location_id,
            "search_type": "CITY",
            "adults": 1,
            "room_qty": 1,
            "page_number": 1,
            "units": "metric",
            "temperature_unit": "c",
            "languagecode": "en-us",
            "currency_code": "USD",
            "arrival_date": checkin,
            "departure_date": checkout
        }
        try:
            response = requests.get(url, headers=self.headers, params=params)
            data = response.json()
            if not data.get("data"):
                print("⚠️ Booking city fallback response:", data)
            return data.get("data", [])
        except Exception as e:
            print(f"❌ Hotel search by dest_id failed: {e}")
        return []

    def _search_hotels_by_latlng(self, lat: float, lon: float, checkin: str, checkout: str) -> List[Dict]:
        url = "https://booking-com15.p.rapidapi.com/api/v1/hotels/searchHotels"
        params = {
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "search_type": "LATLONG",
            "adults": 1,
            "room_qty": 1,
            "page_number": 1,
            "units": "metric",
            "temperature_unit": "c",
            "languagecode": "en-us",
            "currency_code": "USD",
            "arrival_date": checkin,
            "departure_date": checkout
        }
        try:
            response = requests.get(url, headers=self.headers, params=params)
            data = response.json()
            if not data.get("data"):
                print("⚠️ Booking lat/lng response:", data)
            return data.get("data", [])
        except Exception as e:
            print(f"❌ Hotel search by lat/lng failed: {e}")
        return []

    def _get_lat_lng_from_google(self, query: str) -> Optional[Dict[str, float]]:
        url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        params = {"query": query, "key": GOOGLE_API_KEY}
        try:
            res = requests.get(url, params=params)
            data = res.json()
            if data.get("status") == "OK" and data.get("results"):
                loc = data["results"][0]["geometry"]["location"]
                return {"lat": loc["lat"], "lng": loc["lng"]}
        except Exception as e:
            print(f"❌ Google location resolution failed: {e}")
        return None

def booking_hotel_finder_tool(destination: str,
                               checkin_date: str,
                               checkout_date: str,
                               budget: float,
                               preferences: str = "",
                               landmark_hint: str = "") -> dict:
    return BookingHotelSearcher().find_hotel(
        destination, checkin_date, checkout_date, budget, preferences, landmark_hint
    )
