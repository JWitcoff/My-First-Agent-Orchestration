import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import asyncio
import json

template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
app = Flask(__name__, template_folder=template_dir)
CORS(app)  # Enable CORS for frontend communication

# Serve the frontend HTML
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/api/plan-trip', methods=['POST'])
def plan_trip():
    """Main endpoint that processes travel requests"""
    try:
        from app.agents.travel_agent import Runner, travel_agent  # ⬅️ Delayed import to avoid circular import

        data = request.json
        print(f"📥 Received request: {json.dumps(data, indent=2)}")
        
        # Convert frontend form data to natural language query for your agent
        query = build_query_from_form_data(data)
        print(f"🤖 Generated query: {query}")
        
        # Run your actual travel agent
        print("🔄 Running travel agent...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(Runner.run(travel_agent, query))
            print("✅ Travel agent completed successfully")
        finally:
            loop.close()
        
        # Extract results from your agent's output
        response_data = extract_agent_results(result)
        print(f"📤 Sending response: {json.dumps(response_data, indent=2)}")
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Error processing request: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Sorry, there was an error planning your trip. Please try again.'
        }), 500

def build_query_from_form_data(data):
    """Convert frontend form data into natural language query for your travel agent"""
    
    # Build the base query
    query_parts = []
    
    # Destination and duration
    destination = data.get('destination', '')
    duration = data.get('duration', 5)
    query_parts.append(f"I want to go to {destination} for {duration} days")
    
    # Dates
    dates = data.get('dates', '')
    if dates:
        query_parts.append(f"during {dates}")
    
    # Origin
    origin = data.get('origin', 'Los Angeles')
    if origin:
        query_parts.append(f"traveling from {origin}")
    
    # Budget
    budget = data.get('budget', 3000)
    if budget:
        query_parts.append(f"with a budget of ${budget}")
    
    # Flight preferences
    flight_prefs = []
    flight_type = data.get('flightType', 'direct')
    if flight_type == 'direct':
        flight_prefs.append("I prefer direct flights")
    elif flight_type == 'cheapest':
        flight_prefs.append("I want the cheapest flights")
    
    preferred_airlines = data.get('preferredAirlines', '')
    if preferred_airlines:
        flight_prefs.append(f"preferably with {preferred_airlines}")
    
    if flight_prefs:
        query_parts.append(". ".join(flight_prefs))
    
    # Hotel preferences
    hotel_prefs = []
    amenities = data.get('amenities', [])
    if amenities:
        amenities_text = ", ".join(amenities)
        hotel_prefs.append(f"I want a hotel with {amenities_text}")
    
    location_prefs = data.get('locationPrefs', '')
    if location_prefs:
        hotel_prefs.append(f"located {location_prefs}")
    
    special_requests = data.get('specialRequests', '')
    if special_requests:
        hotel_prefs.append(f"with {special_requests}")
    
    if hotel_prefs:
        query_parts.append(". ".join(hotel_prefs))
    
    return ". ".join(query_parts) + "."

def extract_agent_results(result):
    """Extract and format results from your travel agent"""
    from app.agents.travel_agent import FlightRecommendation, HotelRecommendation, TravelPlan  # ⬅️ Delayed import here too
    
    flight_data = None
    hotel_data = None
    travel_plan_data = None
    
    for output in result.outputs:
        if isinstance(output, FlightRecommendation):
            flight_data = {
                'airline': output.airline,
                'origin': output.origin,
                'destination': output.destination,
                'departure_date': output.departure_date,
                'return_date': output.return_date,
                'price': output.price,
                'direct_flight': output.direct_flight,
                'recommendation_reason': output.recommendation_reason
            }
        elif isinstance(output, HotelRecommendation):
            hotel_data = {
                'name': output.name,
                'destination': output.destination,
                'checkin_date': output.checkin_date,
                'checkout_date': output.checkout_date,
                'price_per_night': output.price_per_night,
                'amenities': output.amenities,
                'recommendation_reason': output.recommendation_reason
            }
        elif isinstance(output, TravelPlan):
            travel_plan_data = {
                'destination': output.destination,
                'duration_days': output.duration_days,
                'dates': output.dates,
                'budget': output.budget,
                'activities': output.activities,
                'notes': output.notes
            }
    
    return {
        'success': True,
        'travel_plan': travel_plan_data,
        'flight': flight_data,
        'hotel': hotel_data
    }

@app.route('/api/test', methods=['GET'])
def test_connection():
    """Test endpoint to verify the API is working"""
    return jsonify({
        'status': 'success', 
        'message': 'Travel Agent API is running!',
        'endpoints': {
            'POST /api/plan-trip': 'Plan a complete trip',
            'GET /api/test': 'Test API connection'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'travel-agent-api'})

if __name__ == '__main__':
    print("🚀 Starting Travel Agent API Server...")
    print("=" * 50)
    print("📡 Frontend will be available at: http://localhost:8000")
    print("🔧 API endpoint: http://localhost:8000/api/plan-trip")
    print("🧪 Test endpoint: http://localhost:8000/api/test")
    print("=" * 50)
    print("💡 Serving your frontend from templates/index.html via Flask's render_template()")
    print("=" * 50)
    
    # Start the Flask server on port 8000
    print("📣 Flask is starting — auto-reloader is", "ON" if app.debug else "OFF")
    app.run(debug=True, host='0.0.0.0', port=8000)
