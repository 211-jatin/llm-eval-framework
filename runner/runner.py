import yaml

from metrics import bertscore
from metrics import exact_match
from metrics import f1_score
from metrics import cost
from metrics import toxicity
from metrics import ragas_metrics

class Runner:

    def __init__(self, adapter):
        self.adapter = adapter


    def load_tasks(self, task_files: list[str]) -> list:

        all_tasks = []

        for file in task_files:

            with open(file) as stream:
                data = yaml.safe_load(stream)

                all_tasks.extend(data["tasks"])

        return all_tasks


    def run(
        self,
        task_files: list[str],
        use_bertscore: bool = False
    ) -> list[dict]:

        tasks = self.load_tasks(task_files)

        results = []

        for task in tasks:

            response = self.adapter.generate(
                task["prompt"]
            )

            gt = task["ground_truth"]

            if isinstance(gt, list):
                gt = " ".join(gt)

            result = {

                "task_id": task["id"],

                "category": task["category"],

                "prompt": task["prompt"],

                "ground_truth": gt,

                "prediction": response["text"],

                "exact_match": exact_match.compute(
                    prediction=response["text"],
                    reference=gt
                ),

                "f1": f1_score.compute(
                    prediction=response["text"],
                    reference=gt
                ),

                "bert_score": (
                    bertscore.compute(
                        response["text"],
                        gt
                    )
                    if use_bertscore
                    else None
                ),

                "latency": response["latency"],

                "prompt_tokens": response[
                    "prompt_tokens"
                ],

                "completion_tokens": response[
                    "completion_tokens"
                ],

                "cost": cost.compute(
                    prompt_tokens=response[
                        "prompt_tokens"
                    ],
                    completion_tokens=response[
                        "completion_tokens"
                    ],
                    model_name=response[
                        "model_name"
                    ]
                ),

                "toxicity": toxicity.compute(
                    response["text"]
                ),
                "faithfulness": None,
                "answer_relevancy": None,  

                "model_name": response["model_name"]
            }
            if task.get("category") == "rag_faithfulness":
                ragas_scores = ragas_metrics.compute(
                        question=task["prompt"],
                        answer=response["text"],
                        context=task.get("context", "")
                        )
                result["faithfulness"] = ragas_scores["faithfulness"]
                result["answer_relevancy"] = ragas_scores["answer_relevancy"]
            results.append(result)

        return results  


    def print_results(self, results: list[dict]):

        print(
            f"\n{'task_id':<12} "
            f"{'category':<20} "
            f"{'f1':<8} "
            f"{'latency':<10} "
            f"{'cost':<10} "
            f"{'toxicity':<12} "
            f"{'faithful':<10} "
            f"{'relevancy':<10} "
            f"{'model'}"
        )

        print("-" * 120)

        for r in results:

            faithful = f"{r['faithfulness']:.3f}" if r.get('faithfulness') is not None else "N/A"
            relevancy = f"{r['answer_relevancy']:.3f}" if r.get('answer_relevancy') is not None else "N/A"

            print(
                f"{r['task_id']:<12} "
                f"{r['category']:<20} "
                f"{r['f1']:<8.3f} "
                f"{r['latency']:<10.2f} "
                f"{r['cost']:<10.6f} "
                f"{r['toxicity']:<12.6f} "
                f"{faithful:<10} "
                f"{relevancy:<10} "
                f"{r['model_name']}"
            )