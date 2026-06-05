import yaml


def compute(
    prompt_tokens: int,
    completion_tokens: int,
    model_name: str
) -> float:

    with open("config/pricing.yaml", "r") as f:
        pricing = yaml.safe_load(f)

    model_pricing = pricing["models"].get(model_name)

    if not model_pricing:
        return 0.0

    input_price = model_pricing["input_per_1k"]
    output_price = model_pricing["output_per_1k"]

    input_cost = (
        prompt_tokens / 1000
    ) * input_price

    output_cost = (
        completion_tokens / 1000
    ) * output_price

    total_cost = input_cost + output_cost

    return round(total_cost, 6)

