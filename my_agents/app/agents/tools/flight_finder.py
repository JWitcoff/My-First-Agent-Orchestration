# app/agents/tools/flight_finder.py

import os
from amadeus import Client, ResponseError
from dotenv import load_dotenv

load_dotenv()
amadeus = Client(
    client_id=os.getenv("AMADEUS_API_KEY"),
    client_secret=os.getenv("AMADEUS_API_SECRET")
)

def find_flights(params: dict) -> list:
    """
    params keys:
      - origin (e.g. "LAX")
      - destination (e.g. "TYO")
      - start_date (YYYY-MM-DD)
      - end_date   (YYYY-MM-DD)
      - budget     (float)
    """
    origin  = params["origin"]
    dest    = params["destination"]
    start   = params["start_date"]
    end     = params["end_date"]

    try:
        response = amadeus.shopping.flight_offers_search.get(
            originLocationCode=origin,
            destinationLocationCode=dest,
            departureDate=start,
            returnDate=end,
            currencyCode="USD",
            max=5
        )
    except ResponseError as e:
        return [{"error": str(e)}]

    flights = []
    for offer in response.data:
        total = float(offer["price"]["total"])
        flights.append({
            "airline":   offer["validatingAirlineCodes"][0],
            "price":     total,
            "direct":    offer["itineraries"][0]["segments"][0]["numberOfStops"] == 0,
            "departure": start,
            "return":    end,
        })
    return flights
