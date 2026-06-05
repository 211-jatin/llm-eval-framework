from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from storage.database import SessionLocal
from storage.models import EvalRun

from adapters import gemini_adapter
from adapters import ollama_adapter

from runner.runner import Runner

from storage import store
from storage.serializers import serialize_eval_run

app = FastAPI()

app.mount("/dashboard", StaticFiles(directory="dashboard", html=True), name="dashboard")



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class RunRequest(BaseModel):
    model: str
    tasks: list[str]
    use_bertscore: bool = False


@app.get("/")
def root():
    return {
        "message": "LLM Eval Framework API"
    }


@app.get("/runs")
def get_runs():

    db = SessionLocal()

    try:
        runs = db.query(EvalRun).all()

        return [
            serialize_eval_run(run)
            for run in runs
        ]

    finally:
        db.close()


@app.get("/runs/{run_id}")
def get_run(run_id: int):

    db = SessionLocal()

    try:
        run = db.query(EvalRun).filter(
            EvalRun.id == run_id
        ).first()

        if not run:
            raise HTTPException(
                status_code=404,
                detail="Run not found"
            )

        return serialize_eval_run(run)

    finally:
        db.close()


@app.post("/run")
def run_eval(request: RunRequest):

    try:

        if "gemini" in request.model.lower():
            adapter = gemini_adapter.GeminiAdaptor(
                model_name=request.model
            )

        else:
            adapter = ollama_adapter.OllamaAdaptors(
                model_name=request.model
            )

        runner = Runner(adapter=adapter)

        results = runner.run(
            task_files=request.tasks,
            use_bertscore=request.use_bertscore
        )

        run_id = store.save_run(
            results=results,
            model_name=request.model
        )

        return {
            "run_id": run_id,
            "model": request.model,
            "results": results,
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

