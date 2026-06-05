def compute(prediction: str, reference: str) -> float:
    prediction = prediction.lower().strip()
    reference = reference.lower().strip()
    
    prediction_token=prediction.split()
    reference_token=reference.split()
    
    if not prediction_token or not reference_token:return 0.0

    overlap= set(prediction_token) & set(reference_token)
    overlap_len=len(overlap)

    precision= overlap_len / len(prediction_token)  
    recall=overlap_len / len(reference_token)
    
    if precision + recall == 0: return 0.0

    f1_score= 2*(precision * recall)/ (precision + recall)

    return f1_score