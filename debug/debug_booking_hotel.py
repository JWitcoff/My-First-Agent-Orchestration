from app.agents.tools.booking_hotel_search import booking_hotel_finder_tool

res = booking_hotel_finder_tool(
    destination="Tokyo",
    checkin_date="2025-06-25",
    checkout_date="2025-06-27",
    budget=600,
    preferences="Pool",
    landmark_hint="Shibuya"
)
print(res)
