import os
import httpx
from openai import OpenAI

# 1. Create a custom HTTP client that ignores SSL verification (the -k equivalent)
# This is often necessary for academic/private cluster endpoints.
http_client = httpx.Client(verify=False)

client = OpenAI(
    api_key=os.getenv("NAUT_API_KEY"),
    base_url="https://ellm.nrp-nautilus.io/v1",
    http_client=http_client
)

try:
    completion = client.chat.completions.create(
        model="gemma3",
        messages=[
            {"role": "system", "content": "Talk like a pirate."},
            {"role": "user", "content": "How do I check if a Python object is an instance of a class?"},
        ],
    )
    print(completion.choices[0].message.content)
except Exception as e:
    print(f"Detailed Error: {e}")