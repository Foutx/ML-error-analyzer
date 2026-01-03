import os

import json
from dotenv import load_dotenv

from src.model_client import ModelClient

load_dotenv()
api_key = os.getenv("LLAMA_API_KEY")

if not api_key:
    raise RuntimeError("GIGACHAT_API_KEY not found in environment")

try:

    model = ModelClient(api_key)
    text = model.get_recommendations("Привет, как дела?")
    status, text_str = text
    data = json.loads(text_str)

    print(f"Answer: {data['choices'][0]['message']['content']}")
    print(f"HTTP status: {status}")

    print("Test passed!")

except Exception as e:

    print(e)