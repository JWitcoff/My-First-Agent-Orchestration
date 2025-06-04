from app.agents.tools.google_hotel_searcher import google_hotel_finder_tool

result = google_hotel_finder_tool(
    destination="Tokyo",
    checkin_date="2025-07-01",
    checkout_date="2025-07-03",
    budget=500,
    preferences="Pool",
    landmark_hint="Shibuya"
)

print(result)
