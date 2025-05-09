import streamlit as st
import asyncio
from Travel_Agent import Runner, travel_agent, FlightRecommendation, HotelRecommendation, TravelPlan

st.set_page_config(page_title="AI Travel Planner", layout="centered")
st.title("🌍 AI Travel Planner")

st.markdown("Describe your dream trip and let the agents do the rest.")

# Input form
with st.form("trip_form"):
    user_query = st.text_area(
        "Your request",
        value="I want to go to Tokyo in late June on a budget of $5000. I love food and history.",
        help="Mention destination, dates, budget, and activities if possible."
    )
    submitted = st.form_submit_button("✈️ Plan My Trip")

if submitted:
    with st.spinner("Planning your trip with real agents..."):
        try:
            result = asyncio.run(Runner.run(travel_agent, user_query))

            travel_plan = None
            flight = None
            hotel = None

            for output in result.outputs:
                if isinstance(output, TravelPlan):
                    travel_plan = output
                elif isinstance(output, FlightRecommendation):
                    flight = output
                elif isinstance(output, HotelRecommendation):
                    hotel = output

            if travel_plan:
                st.subheader("📋 Trip Overview")
                st.json(travel_plan.model_dump())

            if flight:
                st.subheader("✈️ Flight Recommendation")
                st.write(f"**Airline:** {flight.airline}")
                st.write(f"**Route:** {flight.origin} → {flight.destination}")
                st.write(f"**Dates:** {flight.departure_date} to {flight.return_date}")
                st.write(f"**Price:** ${flight.price}")
                st.write(f"**Direct Flight:** {'Yes' if flight.direct_flight else 'No'}")
                st.write(f"**Reason:** {flight.recommendation_reason}")

            if hotel:
                st.subheader("🏨 Hotel Recommendation")
                st.write(f"**Hotel:** {hotel.name}")
                st.write(f"**Dates:** {hotel.checkin_date} to {hotel.checkout_date}")
                st.write(f"**Price/Night:** ${hotel.price_per_night}")
                st.write(f"**Amenities:** {', '.join(hotel.amenities)}")
                st.write(f"**Reason:** {hotel.recommendation_reason}")

            if not travel_plan:
                st.error("⚠️ No travel plan could be created. Try refining your query.")

        except Exception as e:
            st.exception(e)
