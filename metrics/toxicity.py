from detoxify import Detoxify
model = Detoxify("original")


def compute(text: str) -> float:

    results = model.predict(text)

    toxicity_score = results["toxicity"]

    return float(round(toxicity_score, 6))
