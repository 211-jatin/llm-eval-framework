from storage.models import EvalRun, EvalResult

def serialize_eval_result(result: EvalResult) -> dict:
    return {
    "id": result.id,
    "run_id": result.run_id,
    "task_id": result.task_id,
    "category": result.category,
    "prediction": result.prediction,
    "ground_truth": result.ground_truth,
    "exact_match": result.exact_match,
    "f1": result.f1,
    "latency": result.latency,
    "cost": result.cost,
    "toxicity": result.toxicity,
    "bert_score": result.bert_score,
    "faithfulness": result.faithfulness,
    "relevance": result.answer_relevancy,
    "prompt_tokens": result.prompt_tokens,
    "completion_tokens": result.completion_tokens,
    "model_name": result.model_name
}

def serialize_eval_run(run: EvalRun) -> dict:
    return {
        "id": run.id,
        "model_name": run.model_name,
        "run_date": run.run_date.isoformat(),
        "total_tasks": run.total_tasks,
        "avg_f1": run.avg_f1,
        "avg_latency": run.avg_latency,
        "avg_toxicity": run.avg_toxicity,
        "avg_cost": run.avg_cost,
        "avg_faithfulness": run.avg_faithfulness,
        "avg_relevance": run.avg_relevance,
        "results": [
            serialize_eval_result(result)
            for result in run.results
        ]
    }