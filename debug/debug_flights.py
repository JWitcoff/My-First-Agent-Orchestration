# debug_flight_finder.py

from app.agents.tools.flight_search import FlightSearcher, flight_finder_tool
from pprint import pprint


def test_class_direct():
    print("🔍 [TEST 1] FlightSearcher Class — Direct Usage")
    searcher = FlightSearcher()
    result = searcher.find_flight(
        origin="Los Angeles",
        destination="Tokyo",
        departure_date="2025-06-25",
        return_date="2025-07-02"
    )
    pprint(result)
    if result["price"] > 0:
        print("✅ Real flight found via class.")
    else:
        print(f"❌ Class search failed: {result['recommendation_reason']}")
    print("-" * 60)


def test_function_tool():
    print("🔍 [TEST 2] Agent-Compatible Function — flight_finder_tool")
    result = flight_finder_tool(
        origin="New York",
        destination="Paris",
        dates=["2025-07-10", "2025-07-20"]
    )
    pprint(result)
    if result["price"] > 0:
        print("✅ Real flight found via function.")
    else:
        print(f"❌ Function search failed: {result['recommendation_reason']}")
    print("-" * 60)


def test_error_handling():
    print("🔍 [TEST 3] Invalid Destination (Trigger Fallback)")
    result = flight_finder_tool(
        origin="Los Angeles",
        destination="Unknownland",
        dates=["2025-07-01", "2025-07-08"]
    )
    pprint(result)
    if result["airline"] == "Estimated":
        print("✅ Fallback estimate triggered as expected.")
    else:
        print("❌ Fallback did not trigger correctly.")
    print("-" * 60)


def test_past_date_handling():
    print("🔍 [TEST 4] Past Date Handling")
    result = flight_finder_tool(
        origin="San Francisco",
        destination="Rome",
        dates=["2024-01-01", "2024-01-07"]  # Intentionally in the past
    )
    pprint(result)
    if result["price"] > 0:
        print("✅ Auto-adjusted past date as expected.")
    else:
        print(f"❌ Past date logic failed: {result['recommendation_reason']}")
    print("-" * 60)


if __name__ == "__main__":
    print("🧪 Running Flight Finder Debug Tests...\n")
    test_class_direct()
    test_function_tool()
    test_error_handling()
    test_past_date_handling()
    print("🧾 All tests completed.")
