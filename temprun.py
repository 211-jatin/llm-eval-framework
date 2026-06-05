from google import genai
import os
from dotenv import load_dotenv
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
for m in client.models.list():
    print(m.name)

MODEL_NAME="gemma-4-31b-it"
response = client.models.generate_content(
        model=MODEL_NAME,
        contents="HELLO HWO ARE YOU ",
    )
print(response)