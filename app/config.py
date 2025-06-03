from dotenv import load_dotenv
load_dotenv()

import os
from amadeus import Client

AMA_CLIENT = Client(client_id=os.getenv("AMADEUS_API_KEY"),
                    client_secret=os.getenv("AMADEUS_API_SECRET"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_CHOICE = os.getenv("MODEL_CHOICE", "gpt-4o")
