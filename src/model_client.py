import requests

class ModelClient:

    def __init__(self, api_key, url="https://api.groq.com/openai/v1/chat/completions"):

        self.api_key = api_key
        self.url = url

    def get_recommendations(self, prompt: str):

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2
        }

        response = requests.post(self.url, headers=headers, json=data, timeout=10)
        return (response.status_code, response.text)
