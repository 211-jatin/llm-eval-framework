# LLM Evaluation Framework

A model-agnostic benchmarking framework for evaluating and comparing Large Language Models across quality, latency, cost, faithfulness, and safety metrics.

The system supports both local and hosted models through interchangeable adapters and provides reproducible benchmark execution using YAML-defined task suites.

---

# Features

* Model-agnostic evaluation pipeline
* Swappable adapters for local and hosted LLMs
* YAML-defined benchmark tasks
* Automated metric collection
* Persistent PostgreSQL run history
* FastAPI service layer
* Web dashboard for comparing runs
* CLI interface for evaluation orchestration
* RAG evaluation support
* Toxicity and cost analysis

---

# Architecture

```text
llm-eval-framework/
├── tasks/          # YAML benchmark tasks
├── adapters/       # ModelAdapter base + Gemini/Ollama implementations
├── metrics/        # exact_match, f1, bertscore, latency, cost, toxicity, ragas
├── runner/         # Eval orchestration
├── storage/        # PostgreSQL models, store, serializers
├── api/            # FastAPI endpoints
├── dashboard/      # HTML comparison UI
├── config/         # pricing.yaml
└── cli.py          # Typer CLI entry point
```

---

# Tech Stack

* Python
* FastAPI
* PostgreSQL
* SQLAlchemy
* Typer
* Ollama
* Gemini API
* BERTScore
* Detoxify
* YAML
* RAGAS (custom implementation)

---

# Evaluation Metrics

The framework evaluates models across multiple dimensions:

| Category    | Metrics                          |
| ----------- | -------------------------------- |
| Quality     | Exact Match, F1 Score, BERTScore |
| RAG         | Faithfulness, Answer Relevancy   |
| Performance | Latency (per request)            |
| Cost        | Estimated token cost             |
| Safety      | Toxicity score using Detoxify    |

---

# Supported Models

The framework is adapter-based and supports multiple providers without changing evaluation logic.

Current implementations:

* Ollama
* Gemini API

The adapter layer is designed so additional providers can be added with minimal changes.

Example future adapters:

* OpenAI
* HuggingFace Transformers
* vLLM
* Anthropic

---

# Installation

## Clone Repository

```bash
git clone <repo>
cd llm-eval-framework
```

## Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file in the project root:

```env
DATABASE_URL=postgresql://postgres:password@localhost:5432/evaldb
GEMINI_API_KEY=your_key
```

---

# Database Setup

Create the PostgreSQL database:

```bash
psql -U postgres -d evaldb
```

The framework automatically creates tables during initialization.

---

# Running Evaluations

## Run Benchmark Suite

```bash
python cli.py run \
  --model mistral:7b \
  tasks/qa.yaml \
  tasks/summarization.yaml \
  tasks/rag_faithfulness.yaml \
  --save
```

This command:

* Loads benchmark task suites
* Executes prompts against the selected model
* Computes evaluation metrics
* Stores results in PostgreSQL

---

# CLI Commands

## List Previous Runs

```bash
python cli.py list-runs
```

## Compare Two Runs

```bash
python cli.py compare 7 8
```

---

# API Server

Start the FastAPI server:

```bash
uvicorn api.main:app --reload
```

Default URL:

```text
http://localhost:8000
```

---

# Dashboard

Open the dashboard:

```text
http://localhost:8000/dashboard
```

The dashboard provides:

* Run history
* Metric comparison
* Model-vs-model analysis
* Aggregate benchmark statistics

---

# Task Format

Benchmark tasks are defined using YAML.

Example:

```yaml
- id: qa_001
  category: qa
  prompt: "What is the capital of France?"
  expected_answer: "Paris"
```

Supported task categories:

* Question Answering
* Summarization
* RAG Faithfulness
* Instruction Following
* Code Generation

---

# Adapter Design

All models implement a shared abstract interface.

Example responsibilities:

```python
class ModelAdapter(ABC):
    @abstractmethod
    async def generate(self, prompt: str) -> ModelResponse:
        pass
```

This design allows:

* Swapping models without changing runner logic
* Consistent evaluation flow
* Unified metric computation

---

# Storage Layer

Evaluation runs are persisted using PostgreSQL and SQLAlchemy.

Stored metadata includes:

* Model name
* Task category
* Raw responses
* Latency
* Metric outputs
* Timestamped run history

---

# Example Benchmark Results

Comparison between:

* Run 7 → `mistral:7b`
* Run 8 → `llama3.2`

Total tasks: 20

| Metric           | mistral:7b | llama3.2 |
| ---------------- | ---------- | -------- |
| avg_f1           | 0.1866     | 0.1624   |
| avg_latency (s)  | 39.34      | 25.99    |
| avg_toxicity     | 0.0006     | 0.0008   |
| avg_faithfulness | 0.1667     | 0.1333   |
| avg_relevancy    | 0.7833     | 0.800    |

---

# Design Goals

The project was built around the following principles:

* Reproducible evaluations
* Provider-independent benchmarking
* Modular metric integration
* Persistent experiment tracking
* Simple local deployment
* Extensible architecture

---

# Future Improvements

Planned additions:

* Parallel evaluation execution
* Streaming token metrics
* Prompt version tracking
* Judge-model evaluation
* Multi-turn conversation benchmarks
* Dockerized deployment
* Grafana integration
* Distributed evaluation workers

---

# Use Cases

This framework can be used for:

* Comparing local LLMs
* Regression testing prompts
* Evaluating RAG pipelines
* Measuring inference latency
* Tracking model quality over time
* Benchmarking fine-tuned models

---
