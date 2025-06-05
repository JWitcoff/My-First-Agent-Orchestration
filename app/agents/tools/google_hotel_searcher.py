import os
import requests
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY in environment variables.")

class GoogleHotelSearcher:
    def __init__(self):
        self.api_key = GOOGLE_API_KEY

    def find_hotel(self,
                   destination: str,
                   checkin_date: str,
                   checkout_date: str,
                   total_budget: float,
                   preferences: str = "",
                   landmark_hint: str = "") -> Dict:

        query = f"{landmark_hint} {destination}".strip()
        coords = self._get_lat_lng(query)
        if not coords:
            return {"recommendation_reason": f"Could not resolve landmark: '{query}'"}

        lat, lng = coords["lat"], coords["lng"]
        print(f"📍 Google resolved '{query}' to: ({lat}, {lng})")

        hotels = self._search_hotels_nearby(lat, lng)
        if not hotels:
            return {"recommendation_reason": f"No hotels found near {query}"}

        top = sorted(hotels, key=lambda h: h.get("rating", 0), reverse=True)[:3]

        results = []
        for h in top:
            results.append({
                "name": h.get("name"),
                "rating": h.get("rating"),
                "address": h.get("vicinity"),
                "latitude": h["geometry"]["location"]["lat"],
                "longitude": h["geometry"]["location"]["lng"]
            })

        return {"top_matches": results}

    def _get_lat_lng(self, query: str) -> Optional[Dict[str, float]]:
        url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        params = {"query": query, "key": self.api_key}
        try:
            res = requests.get(url, params=params)
            data = res.json()
            if data.get("status") == "OK" and data.get("results"):
                return data["results"][0]["geometry"]["location"]
            else:
                print(f"⚠️ Google location lookup failed: {data.get('status')}")
        except Exception as e:
            print(f"❌ Error in Google location lookup: {e}")
        return None

    def _search_hotels_nearby(self, lat: float, lng: float) -> List[Dict]:
        url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        params = {
            "location": f"{lat},{lng}",
            "radius": 2000,  # meters
            "type": "lodging",
            "key": self.api_key
        }
        try:
            res = requests.get(url, params=params)
            data = res.json()
            if data.get("status") == "OK" and data.get("results"):
                return data["results"]
            else:
                print(f"⚠️ Hotel nearby search failed: {data.get('status')}")
        except Exception as e:
            print(f"❌ Error in hotel nearby search: {e}")
        return []

def google_hotel_finder_tool(destination: str,
                             checkin_date: str,
                             checkout_date: str,
                             budget: float,
                             preferences: str = "",
                             landmark_hint: str = "") -> dict:
    return GoogleHotelSearcher().find_hotel(
        destination, checkin_date, checkout_date, budget, preferences, landmark_hint
    )
