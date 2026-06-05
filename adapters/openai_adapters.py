from abc import abstractmethod
import os
from adapters.base import ModelAdapter
from openai import OpenAI
import time
class OpenAIAdapter(ModelAdapter):
    def __init__(self, model_name: str, api_key: str  = None): # type: ignore
        super().__init__(model_name)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)

    def generate(self, input_prompt: str) -> dict:
        before_time = time.time()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": input_prompt}]
            )
        after_time = time.time()

        text=response.choices[0].message.content
        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0
        latency = after_time - before_time

        return {
            "text": text,
            "latency": latency,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "model_name": self.model_name
            }   