import os
import time
from google import genai
from adapters.base import ModelAdapter

class GeminiAdaptor(ModelAdapter):
    def __init__(self, model_name: str, api_key=None):
        super().__init__(model_name)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.api_key)

    def generate(self, input_prompt: str) -> dict:
        before_time = time.perf_counter()
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=input_prompt
        )
        after_time = time.perf_counter()

        text = response.text
        prompt_tokens = response.usage_metadata.prompt_token_count
        completion_tokens = response.usage_metadata.candidates_token_count
        latency = after_time - before_time

        return {
            "text": text,
            "latency": latency,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "model_name": self.model_name
        }
