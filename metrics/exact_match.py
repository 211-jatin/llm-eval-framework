def compute(prediction: str, reference: str) -> float:
    prediction = prediction.lower().strip()
    reference = reference.lower().strip()
    return 1.0 if prediction == reference else 0.0