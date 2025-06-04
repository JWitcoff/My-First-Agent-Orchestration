def function_tool(func):
    func.openai_schema = {
        "name": func.__name__,
        "description": func.__doc__ or "",
        "parameters": {
            "type": "object",
            "properties": {
                "destination": {
                    "type": "string",
                    "description": "The destination city where the user wants to stay"
                },
                "checkin_date": {
                    "type": "string",
                    "description": "The check-in date in YYYY-MM-DD format"
                },
                "checkout_date": {
                    "type": "string",
                    "description": "The check-out date in YYYY-MM-DD format"
                },
                "budget": {
                    "type": "number",
                    "description": "The total hotel budget for the stay"
                },
                "preferences": {
                    "type": "string",
                    "description": "Hotel preferences like amenities or descriptive phrases (e.g., pool, near Shibuya)"
                },
                "landmark_hint": {
                    "type": "string",
                    "description": "Specific landmark extracted from the user's preference (e.g., Shibuya, Eiffel Tower)"
                }
            },
            "required": ["destination", "checkin_date", "checkout_date", "budget"]
        }
    }
    return func