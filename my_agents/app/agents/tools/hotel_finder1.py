# app/agents/tools/hotel_finder.py

import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("BOOKING_COM_API")
API_HOST = "booking-com.p.rapidapi.com"
HEADERS = {
    "X-RapidAPI-Key": API_KEY,
    "X-RapidAPI-Host": API_HOST
}

def find_hotels(params: dict) -> list:
    """
    params keys:
      - destination_code (e.g. "TYO")
      - start_date (YYYY-MM-DD)
      - end_date   (YYYY-MM-DD)
      - budget     (float total for entire stay)
    """
    dest_code = params.get("destination_code", params["destination"])
    checkin   = params["start_date"]
    checkout  = params["end_date"]
    nights    = (  # compute number of nights
        __import__("datetime").datetime.strptime(checkout, "%Y-%m-%d")
        - __import__("datetime").datetime.strptime(checkin, "%Y-%m-%d")
    ).days
    per_night = params["budget"] / max(nights, 1)

    url = f"https://{API_HOST}/v1/hotels/search"
    qs = {
        "destination_id":    dest_code,
        "checkin_date":      checkin,
        "checkout_date":     checkout,
        "adults_number":     "1",
        "order_by":          "popularity",
        "filter_by_currency":"USD",
        "room_number":       "1",
        "units":             "metric",
        "page_number":       "0"
    }

    resp = requests.get(url, headers=HEADERS, params=qs)
    data = resp.json().get("result", [])

    hotels = []
    for h in data[:5]:
        price = float(h.get("min_total_price", 0))
        hotels.append({
            "name":            h.get("hotel_name"),
            "price_per_night": price,
            "within_budget":   price <= per_night,
            "distance":        h.get("distance_from_city_center")
        })
    return hotels
