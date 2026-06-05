from adapters.base import ModelAdapter
import time 
from ollama import chat

class OllamaAdaptors(ModelAdapter):
    def __init__(self, model_name: str):
        super().__init__(model_name)
    

    def generate(self, input_prompt: str) -> dict:
        before_time = time.perf_counter()
        response = chat(
            model=self.model_name,
            messages=[{'role': 'user', 'content': input_prompt}],
            )
        after_time = time.perf_counter()
        latency=after_time-before_time
        text = response.message.content
        prompt_tokens = getattr(response, 'prompt_eval_count', 0)
        completion_tokens = getattr(response, 'eval_count', 0)

        return {
            "text": text,
            "latency": latency,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "model_name": self.model_name
            } 
