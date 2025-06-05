from app.agents.tools.booking_hotel_search import booking_hotel_finder_tool

res = booking_hotel_finder_tool(
    destination="Tokyo",
    checkin_date="2025-06-10",
    checkout_date="2025-06-12",
    budget=400,
    preferences="modern, clean",
    landmark_hint="Shibuya"
)
print(res)

