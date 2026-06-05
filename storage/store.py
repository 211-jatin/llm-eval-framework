from storage.database import SessionLocal
from storage.models import EvalRun, EvalResult

def save_run(results: list[dict], model_name: str) -> int:
    session=SessionLocal()

    avg_f1 = sum(r["f1"] for r in results) / len(results)
    avg_latency = sum(r["latency"] for r in results) / len(results)
    avg_toxicity = sum(r["toxicity"] for r in results) / len(results) if results and "toxicity" in results[0] else None
    avg_cost = sum(r["cost"] for r in results) / len(results) if results and "cost" in results[0] else None
    faithfulness_scores = [r["faithfulness"] for r in results if r.get("faithfulness") is not None]
    avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else None
    relevancy_scores = [r["answer_relevancy"] for r in results if r.get("answer_relevancy") is not None]
    avg_relevancy = sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else None
    run = EvalRun(
        model_name=model_name, 
        total_tasks=len(results), 
        avg_f1=avg_f1, 
        avg_latency=avg_latency,
        avg_toxicity=avg_toxicity,
        avg_cost=avg_cost,
        avg_faithfulness=avg_faithfulness,
        avg_relevance=avg_relevancy
        )
    try:
        session.add(run)
        session.flush()
        for r in results:

            result= EvalResult(
                run_id= run.id,
                task_id=r["task_id"],
                category=r["category"],
                prediction=r["prediction"],
                ground_truth=r["ground_truth"],
                exact_match=r["exact_match"],
                f1=r["f1"],
                latency=r["latency"],
                bert_score=r.get("bert_score"),
                cost=r.get("cost"),
                toxicity=r.get("toxicity"),
                faithfulness=r.get("faithfulness"),
                answer_relevancy=r.get("answer_relevancy"),
                prompt_tokens=r["prompt_tokens"],
                completion_tokens=r["completion_tokens"],
                model_name=r["model_name"]
            )
            session.add(result)
        session.commit()
        return run.id
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()
        