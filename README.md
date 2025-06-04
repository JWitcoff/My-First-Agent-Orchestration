# ✈️ AI Travel Planner

A full-stack AI travel planning assistant that generates personalized itineraries—including flights, hotels, and activities—based on natural language input.

This project was built to explore agentic architectures, tool-calling via OpenAI's function calling API, and real-world API integrations (Amadeus for flights, OpenWeatherMap for weather). It is designed to demonstrate modular reasoning, prompt engineering, and end-to-end product development—from LLM prompt orchestration to front-end and API deployment via Flask.

## 🧠 What It Does

Users input a travel request in natural language (e.g., “I want to go to Tokyo for 5 days in late June with a $5,000 budget”), and the agent returns:

- A summarized travel plan
- A recommended flight (via Amadeus API)
- A recommended hotel (logic-driven)
- Relevant weather data

The system handles vague or incomplete inputs by making reasonable assumptions using LLM pre-processing logic.

## ⚙️ Architecture Overview

```
├── app
│   ├── agents
│   │   ├── travel_agent.py       # Main agent logic, schema definitions, agent orchestration
│   │   └── tools
│   │       ├── flight_search.py  # Amadeus flight tool wrapper
│   │       └── hotel_search.py   # Hotel logic (fallback-based, no Booking.com API)
│   ├── routes
│   │   └── api.py                # Flask API endpoints and input/output serialization
│   └── templates
│       └── index.html            # Frontend UI for entering travel queries
├── run_flask.py                  # Launches Flask server
├── run.py                        # Standalone CLI testing script
```

## 🔧 Key Features

- **Multi-agent orchestration** with tool handoffs and structured output using Pydantic models
- **Custom tool functions** exposed via OpenAI function calling (`flight_finder_tool`, `hotel_finder_tool`, `get_current_weather`)
- **Fuzzy date parsing** to translate loose timeframes (e.g. “next summer”, “late July”) into ISO date ranges
- **Amadeus API integration** for real-time flight options
- **Weather overlay** via OpenWeatherMap API
- **Flask-based frontend** with CORS-enabled API routes

## 🚀 Running the App

### Prerequisites

- Python 3.9+
- API keys for:
  - [Amadeus API](https://developers.amadeus.com)
  - [OpenWeatherMap](https://openweathermap.org/api)
  - [OpenAI](https://platform.openai.com/account/api-keys)

### Setup

```bash
git clone https://github.com/your-username/ai-travel-planner.git
cd ai-travel-planner
python -m venv env
source env/bin/activate
pip install -r requirements.txt

# Add a `.env` file with:
# OPENAI_API_KEY=your_key
# AMADEUS_API_KEY=your_key
# AMADEUS_API_SECRET=your_secret
# OPENWEATHER_API_KEY=your_key
```

### Run Locally

```bash
python run_flask.py
```

Visit `http://localhost:8000` in your browser to use the planner.

## 🧪 Testing Queries

The app supports queries like:

- “I want to go to Paris for 3 days in the winter with a $2,000 budget.”
- “Plan a trip to San Francisco from Jan 1 to Jan 8. I like hiking and want to be near Golden Gate Park.”

## 📚 Learnings and Focus Areas

This project demonstrates:

- Orchestration of chained reasoning steps across agents
- Tool calling with argument cleaning and coroutine support
- Robust prompt design for schema enforcement and behavior shaping
- Clean, maintainable Python project structuring and modular tool integration
- Bridging LLM logic with deterministic APIs for real-world usability
