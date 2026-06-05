from typing import List
from adapters.gemini_adapter import GeminiAdaptor
from adapters.ollama_adapter import OllamaAdaptors
from runner.runner import Runner
from storage.store import save_run
from storage.models import EvalRun
from storage.database import SessionLocal

import typer

app = typer.Typer(invoke_without_command=True)

@app.command()
def run(
    tasks: List[str] = typer.Argument(...),
    model: str = typer.Option(...),
    use_bertscore: bool = False,
    save: bool = False,
):
    if "gemini" in model.lower():
        adapter = GeminiAdaptor(model_name=model)
    else:
        adapter = OllamaAdaptors(model_name=model)

    runner=Runner(adapter)
    
    results = runner.run(
    task_files=tasks,
    use_bertscore=use_bertscore,
    )

    runner.print_results(results)
    if save:
        save_run(results=results,model_name=model)
        
@app.command()
def list_runs():
    """List past eval runs."""

    db = SessionLocal()

    try:
        runs = db.query(EvalRun).all()

        if not runs:
            print("No eval runs found.")
            return

        print(
            f"{'ID':<5}"
            f"{'MODEL':<30}"
            f"{'DATE':<25}"
            f"{'TASKS':<10}"
            f"{'AVG_F1':<12}"
            f"{'AVG_LATENCY':<15}"
        )

        print("-" * 100)

        for r in runs:
            print(
                f"{r.id:<5}"
                f"{r.model_name:<30}"
                f"{r.run_date.strftime("%Y-%m-%d %H:%M"):<25}"
                f"{r.total_tasks:<10}"
                f"{round(r.avg_f1, 4):<12}"
                f"{round(r.avg_latency, 4):<15}"
            )

    finally:
        db.close()

@app.command()
def compare(run_a: int,run_b: int):

    try:
        db=SessionLocal()
        run1=db.query(EvalRun).filter(
            EvalRun.id == run_a
            ).first()
        run2=db.query(EvalRun).filter(
            EvalRun.id==run_b,
            ).first()
        
        if not run1:
            print(f"Run {run_a} not found")
            return
        if not run2:
            print(f"Run {run_b} not found")
            return
        
        comparisons = [
            ("model_name", run1.model_name, run2.model_name),
            ("total_tasks", run1.total_tasks, run2.total_tasks),
            ("avg_f1", round(run1.avg_f1, 4), round(run2.avg_f1, 4)),
            ("avg_latency", round(run1.avg_latency, 4), round(run2.avg_latency, 4)),
            ("avg_toxicity", round(run1.avg_toxicity or 0, 4), round(run2.avg_toxicity or 0, 4)),
            ("avg_faithfulness", round(run1.avg_faithfulness or 0, 4), round(run2.avg_faithfulness or 0, 4)),
            ("avg_relevance", round(run1.avg_relevance or 0, 4), round(run2.avg_relevance or 0, 4)),
        ]
        print(f"{'METRIC':<20}{'RUN_A':<20}{'RUN_B':<20}")
        print("-" * 60)

        for metric, a, b in comparisons:
            print(f"{metric:<20}{str(a):<20}{str(b):<20}")

    finally:
        db.close()
    
    



if __name__ == "__main__":
    app()



