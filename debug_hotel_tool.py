# debug_hotel_tool.py
from app.agents.tools.hotel_search import hotel_finder_tool

print("=== Raw hotel_finder_tool ===")
res = hotel_finder_tool(
    destination="Tokyo",
    checkin_date="2025-06-25",
    checkout_date="2025-06-30",
    budget=4000.0,
    preferences="pool"
)
print(res)
