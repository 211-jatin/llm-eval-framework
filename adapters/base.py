from abc import ABC, abstractmethod


class ModelAdapter(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate(self, input_prompt: str) -> dict:
        """
        Returns:
        {
            "text": str,
            "latency": float,
            "prompt_tokens": int,
            "completion_tokens": int,
            "model": str
        }
        """
        pass