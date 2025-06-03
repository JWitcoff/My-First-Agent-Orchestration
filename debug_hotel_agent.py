# debug_hotel_agent.py

import asyncio
from app.agents.travel_agent import hotel_agent

async def run_hotel_agent():
    inp = {
        "destination": "Tokyo",
        "checkin_date": "2025-06-25",
        "checkout_date": "2025-06-30",
        "budget": 4000.0,
        "preferences": "pool, near Shibuya"
    }
    print("→ Calling hotel_agent.run() with:", inp)
    try:
        result = await hotel_agent.run(inp)
        print("✅ hotel_agent result:", result)
    except Exception as e:
        print("❌ hotel_agent threw an error:", repr(e))

if __name__ == "__main__":
    asyncio.run(run_hotel_agent())
