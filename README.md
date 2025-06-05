# ✈️ AI Travel Planner

An intelligent travel planning assistant that generates complete trip itineraries from natural language requests. Tell it "I want to go to Tokyo for 5 days in late June with a $5,000 budget" and get flights, hotels, and activities.

## 🎯 What It Does

- **Complete Travel Plans**: Destination, dates, budget, and personalized activities
- **Real Flight Data**: Live flight search via Amadeus API with actual prices and airlines  
- **Hybrid Hotel Search**: Combines Google Places (location precision) + Booking.com (pricing) for accurate landmark-based hotel finder
- **Fuzzy Date Parsing**: Understands "late June", "next summer", "winter", etc.
- **Graceful Defaults**: Makes reasonable assumptions for missing information
- **Robust Fallbacks**: Multiple API strategies ensure reliable results even when individual services fail

## 🏗️ Architecture

Multi-agent system with specialized AI agents:
- **Travel Agent**: Main orchestrator that plans the overall trip
- **Flight Agent**: Searches real flights using Amadeus API
- **Hotel Agent**: Uses hybrid approach - Google Places API for precise location finding + Booking.com API for real pricing and availability

All agents use OpenAI function calling for external API integration and return structured JSON output. The system includes comprehensive utilities for date parsing, landmark extraction, and IATA code resolution.

## 🚀 Quick Start (Local)

### Prerequisites
- Python 3.9+
- API Keys:
  - [OpenAI API](https://platform.openai.com/) (GPT-4 access)
  - [Amadeus API](https://developers.amadeus.com/) (flights)
  - [Google Places API](https://developers.google.com/maps/documentation/places/web-service) (hotels)
  - [Booking.com API](https://rapidapi.com/apidojo/api/booking/) (optional)

### Installation
```bash
git clone https://github.com/your-username/ai-travel-planner.git
cd ai-travel-planner
python -m venv env
source env/bin/activate  # Windows: env\Scripts\activate
pip install -r requirements.txt
```

### Configuration
Create `.env` file:
```env
OPENAI_API_KEY=your_openai_key_here
AMADEUS_API_KEY=your_amadeus_key_here  
AMADEUS_API_SECRET=your_amadeus_secret_here
GOOGLE_API_KEY=your_google_key_here
BOOKING_API_KEY=your_booking_key_here
```

### Run Locally
```bash
python run_flask.py
```
Visit `http://localhost:8000`

## 🌐 Deploy to Render

### One-Click Deploy
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

### Manual Deploy
1. **Fork this repository** to your GitHub account
2. **Sign up at [Render.com](https://render.com)** and connect your GitHub
3. **Create New Web Service** → Connect your forked repository
4. **Configure the service:**
   - **Name**: `ai-travel-planner` 
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python run_flask.py`
   
   > 💡 **Note**: We use Flask's built-in server instead of gunicorn to avoid memory issues on free hosting tiers. Perfect for demos and personal use!

5. **Add Environment Variables** in Render dashboard:
   ```
   OPENAI_API_KEY = your_openai_key
   AMADEUS_API_KEY = your_amadeus_key  
   AMADEUS_API_SECRET = your_amadeus_secret
   GOOGLE_API_KEY = your_google_key
   BOOKING_API_KEY = your_booking_key
   ```
6. **Deploy!** Your app will be live at `https://your-app-name.onrender.com`

## 📝 Sample Queries

Try these examples:
- `"Plan a trip to Paris for 3 days in winter with a $2,000 budget"`
- `"I want to go to Tokyo for 5 days in late June. I love history and want to stay near Shibuya Crossing"`
- `"San Francisco trip from Jan 1-8, like hiking, prefer hotel near Golden Gate Bridge"`
- `"Rome for a week in late summer, amazing food, walkable area near Colosseum, $3000 budget"`

## 🛠️ Tech Stack

- **Backend**: Python, Flask (development server for easy deployment)
- **AI**: OpenAI GPT-4 with function calling
- **APIs**: Amadeus (flights), Google Places (location precision), Booking.com (hotel pricing)
- **Validation**: Pydantic for structured outputs
- **Utilities**: Custom date parsing, landmark extraction, IATA code resolution
- **Deployment**: Render, Railway, or local (optimized for free hosting tiers)

## 🔧 Architecture Details

```
├── README.md
├── app/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── travel_agent.py           # Main orchestrator & agent logic
│   │   ├── tools/                    # External API integrations
│   │   │   ├── __init__.py
│   │   │   ├── booking_hotel_search.py   # Booking.com API wrapper
│   │   │   ├── flight_search.py          # Amadeus flight search
│   │   │   ├── google_hotel_searcher.py  # Google Places API
│   │   │   └── hotel_search.py           # Hybrid hotel search logic
│   │   └── utils/                    # Helper utilities
│   │       ├── __init__.py
│   │       ├── dates.py              # Fuzzy date parsing
│   │       ├── iata.py               # Airport code resolution
│   │       ├── nlp.py                # Landmark extraction
│   │       └── openai_helpers.py     # LLM utilities
│   ├── config.py                     # Configuration & API clients
│   ├── routes/
│   │   ├── __init__.py
│   │   └── api.py                    # Flask API endpoints
│   └── templates/
│       └── index.html                # Web interface
├── debug/                            # Development & testing scripts
│   ├── debug_booking_hotel.py
│   ├── debug_dates.py
│   ├── debug_flights.py
│   ├── debug_google_hotel.py
│   ├── debug_hotel_agent.py
│   └── debug_hotel_tool.py
├── procfile                          # Deployment configuration
├── requirements.txt                  # Python dependencies
├── run.py                           # CLI testing script
└── run_flask.py                     # Flask app launcher
```

## 📄 License

MIT License - feel free to use and modify!