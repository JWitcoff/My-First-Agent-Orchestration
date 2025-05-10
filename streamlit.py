import streamlit as st
import asyncio
from Travel_Agent import Runner, travel_agent, FlightRecommendation, HotelRecommendation, TravelPlan

st.set_page_config(page_title="AI Travel Planner", layout="centered")

st.title("🌍 AI Travel Planner")
st.markdown("""
Plan your next adventure with the help of AI agents that find flights, hotels, and create a full travel plan based on your preferences.

_Just type your trip idea below or select an example to get started._
""")

# Example prompts
examples = {
    "Summer in Italy, $3k": "Plan me a summer trip to Italy with a $3000 budget.",
    "Tokyo in June": "I want to go to Tokyo in late June on a budget of $5000. I love food and history.",
    "Paris for 3 days": "I'm going to Paris for 3 days during the winter. I like food and exploring local culture."
}

selected_example = st.selectbox("Try an example:", ["", *examples.keys()])
default_prompt = examples[selected_example] if selected_example else ""

# Input form
with st.form("trip_form"):
    user_query = st.text_area(
        "Your request",
        value=default_prompt,
        placeholder="E.g. Plan me a trip to Rome in late summer with a $3000 budget..."
    )
    submitted = st.form_submit_button("✈️ Plan My Trip")

if submitted and user_query:
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
                st.write(f"**Destination:** {travel_plan.destination}")
                st.write(f"**Duration:** {travel_plan.duration_days} days")
                st.write(f"**Dates:** {', '.join(travel_plan.dates)}")
                st.write(f"**Budget:** ${travel_plan.budget:,.2f}")
                st.write("**Activities:**")
                for act in travel_plan.activities:
                    st.write(f"- {act}")
                if travel_plan.notes:
                    st.info(f"Note: {travel_plan.notes}")

                with st.expander("📄 Raw TravelPlan JSON"):
                    st.json(travel_plan.model_dump())

            if flight:
                st.subheader("✈️ Flight Recommendation")
                st.write(f"**Airline:** {flight.airline}")
                st.write(f"**Route:** {flight.origin} → {flight.destination}")
                st.write(f"**Dates:** {flight.departure_date} to {flight.return_date}")
                st.write(f"**Price:** ${flight.price:,.2f}")
                st.write(f"**Direct Flight:** {'Yes' if flight.direct_flight else 'No'}")
                st.write(f"**Reason:** {flight.recommendation_reason}")

            if hotel:
                st.subheader("🏨 Hotel Recommendation")
                st.write(f"**Hotel:** {hotel.name}")
                st.write(f"**Dates:** {hotel.checkin_date} to {hotel.checkout_date}")
                st.write(f"**Price/Night:** ${hotel.price_per_night:,.2f}")
                st.write(f"**Amenities:** {', '.join(hotel.amenities)}")
                st.write(f"**Reason:** {hotel.recommendation_reason}")

            if not travel_plan:
                st.error("⚠️ No travel plan could be created. Try refining your query.")

        except Exception as e:
            st.exception(e)

st.markdown("""
---
Built by [Justin Witcoff](https://www.linkedin.com/in/justinwitcoff) | [GitHub](https://github.com/justinwitcoff)
""")
