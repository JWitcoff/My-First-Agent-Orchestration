# ─── Standard Library Imports ───────────────────────────────────────────────
from typing import Optional

# ─── Third Party Imports ───────────────────────────────────────────────────
from amadeus import Client

# ─── Custom or Project-Specific Imports ────────────────────────────────────
from app.config import AMA_CLIENT as amadeus

# small in-memory cache
_CITY_CACHE = {
    "tokyo":"TYO","paris":"PAR","rome":"ROM",
    "san francisco":"SFO","new york":"NYC","los angeles":"LAX"
}

def get_city_code(city: str) -> Optional[str]:
    """Return IATA city code for a name, using Amadeus as fallback."""
    key = city.lower().strip()
    if key in _CITY_CACHE:
        return _CITY_CACHE[key]

    try:
        resp = amadeus.reference_data.locations.get(keyword=city, subType="CITY")
        data = resp.data
        if data:
            code = data[0].get("iataCode")
            _CITY_CACHE[key] = code
            return code
    except Exception:
        return None
    return None
