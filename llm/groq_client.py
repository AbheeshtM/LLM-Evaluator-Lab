import os
import time
import requests
from llm.rate_limiter import RateLimiter

# -----------------------
# Rate limiter (GLOBAL)
# -----------------------
rate_limiter = RateLimiter(min_interval_sec=10)

# -----------------------
# Groq config
# -----------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.1-8b-instant"
API_URL = "https://api.groq.com/openai/v1/chat/completions"


def call_groq(prompt: str) -> str:
    # ✅ WAIT HERE (THIS IS THE KEY LINE)
    rate_limiter.wait()

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    # -----------------------
    # Handle rate limits
    # -----------------------
    if response.status_code == 429:
        print("⏳ Rate limit hit. Backing off...")
        time.sleep(20)
        return call_groq(prompt)

    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]
