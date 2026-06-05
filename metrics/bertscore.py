from bert_score import score

def compute(prediction: str, reference: str) -> float:
    
    P, R, F1 = score([prediction], [reference], lang="en", verbose=False)

    return F1[0].item() # type: ignore