#RAGAS Evaluation Metrics
"""
The four metrics measure different parts of a RAG system:

  Metric	                   What it evaluates
Faithfulness        Whether the generated answer is supported by the retrieved contexts
Answer relevancy	Whether the answer addresses the question
Context precision	Whether the retrieved contexts are relevant to the question
Context recall	    Whether the retrieved contexts contain the information needed for the reference answer

python -m pip install -U ragas openai
$env:OPENAI_API_KEY = "your-api-key"
"""

import os
import json
import csv
import asyncio
import logging
import statistics
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

JUDGE_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

JSON_OUTPUT = "ragas_evaluation_results.json"
CSV_OUTPUT = "ragas_evaluation_results.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


# --------------------------------------------------
# 2. EVALUATION DATA STRUCTURE
# --------------------------------------------------

@dataclass
class EvalSample:
    question: str
    ground_truth: str
    answer: str = ""
    contexts: List[str] = field(default_factory=list)
    scores: Dict[str, Optional[float]] = field(default_factory=dict)


# --------------------------------------------------
# 3. SIMULATED RAG PIPELINE
# --------------------------------------------------

class SimulatedRAGPipeline:
    """
    A small demo RAG pipeline.

    Retrieval and answer generation are simulated.
    RAGAS evaluation, however, uses actual LLM-based
    metric scorers.
    """

    def __init__(self):
        self.knowledge_base = {
            "python": [
                "Python is a high-level, interpreted programming "
                "language created by Guido van Rossum and first "
                "released in 1991.",

                "Python supports multiple programming paradigms "
                "including procedural, object-oriented, and "
                "functional programming.",

                "Python's design philosophy emphasizes code "
                "readability with its notable use of significant "
                "indentation.",
            ],

            "machine learning": [
                "Machine learning is a subset of artificial "
                "intelligence that enables systems to learn from data.",

                "Common ML algorithms include linear regression, "
                "decision trees, neural networks, and support "
                "vector machines.",

                "ML models are trained on datasets and evaluated "
                "on held-out test data to measure generalization.",
            ],

            "docker": [
                "Docker is a platform for developing, shipping, "
                "and running applications in containers.",

                "Containers are lightweight, standalone packages "
                "that include everything needed to run software.",

                "Docker uses a client-server architecture with "
                "the Docker daemon managing containers.",
            ],
        }

    def retrieve(self, question: str) -> List[str]:
        """Retrieve documents using simple keyword matching."""

        q = question.lower()
        retrieved = []

        # Python questions
        if "python" in q:
            retrieved.extend(self.knowledge_base["python"])

        # Handle both 'machine learning' and 'ML'.
        if (
            "machine learning" in q
            or "machine-learning" in q
            or "ml" in q.split()
        ):
            retrieved.extend(
                self.knowledge_base["machine learning"]
            )

        # Docker and container questions
        if any(word in q for word in [
            "docker", "container", "containers"
        ]):
            retrieved.extend(self.knowledge_base["docker"])

        return retrieved

    def generate_answer(
        self, question: str, contexts: List[str]
    ) -> str:
        """
        Simulate an answer generator.
        This is not an LLM answer-generation call.
        """

        q = question.lower()

        if not contexts:
            return "I don't have enough information to answer this question."

        if "paradigms" in q:
            return (
                "Python supports procedural, object-oriented, "
                "and functional programming paradigms."
            )

        if "python" in q:
            return (
                "Python is a high-level programming language "
                "created by Guido van Rossum and first released "
                "in 1991. It emphasizes code readability and "
                "supports multiple paradigms including "
                "object-oriented and functional programming."
            )

        if "machine learning" in q or " ml " in f" {q} ":
            if "evaluated" in q or "evaluation" in q:
                return (
                    "ML models are trained on datasets and "
                    "evaluated on held-out test data to measure "
                    "generalization performance."
                )

            return (
                "Machine learning is a subset of artificial "
                "intelligence that enables systems to learn "
                "from data. Common algorithms include decision "
                "trees and neural networks."
            )

        if "docker" in q or "container" in q:
            return (
                "Docker is a platform for running applications "
                "in containers. Containers package applications "
                "with their dependencies. Docker uses a "
                "client-server architecture."
            )

        return "I don't have enough information to answer this question."

    def query(self, question: str) -> Dict:
        contexts = self.retrieve(question)
        answer = self.generate_answer(question, contexts)

        return {
            "answer": answer,
            "contexts": contexts
        }


# --------------------------------------------------
# 4. CUSTOM EVALUATION DATASET
# --------------------------------------------------

def create_dataset() -> List[EvalSample]:
    return [
        EvalSample(
            question="What is Python and who created it?",
            ground_truth=(
                "Python is a high-level programming language "
                "created by Guido van Rossum, first released "
                "in 1991. It emphasizes code readability."
            ),
        ),

        EvalSample(
            question="What is machine learning?",
            ground_truth=(
                "Machine learning is a subset of artificial "
                "intelligence that enables systems to learn "
                "from data. Common algorithms include decision "
                "trees and neural networks."
            ),
        ),

        EvalSample(
            question="How does Docker work?",
            ground_truth=(
                "Docker is a platform for running applications "
                "in containers. It uses a client-server "
                "architecture where the Docker daemon manages "
                "containers."
            ),
        ),

        EvalSample(
            question="What programming paradigms does Python support?",
            ground_truth=(
                "Python supports procedural, object-oriented, "
                "and functional programming paradigms."
            ),
        ),

        EvalSample(
            question="How are ML models evaluated?",
            ground_truth=(
                "ML models are trained on datasets and evaluated "
                "on held-out test data to measure generalization "
                "performance."
            ),
        ),
    ]


# --------------------------------------------------
# 5. INITIALIZE ACTUAL RAGAS METRICS
# --------------------------------------------------

def create_metrics(client):
    # RAGAS uses this LLM as the evaluation judge.
    judge_llm = llm_factory(
        JUDGE_MODEL,
        client=client
    )

    # Answer relevancy also requires embeddings.
    embeddings = embedding_factory(
        "openai",
        model=EMBEDDING_MODEL,
        client=client
    )

    return {
        "faithfulness": Faithfulness(llm=judge_llm),

        "answer_relevancy": AnswerRelevancy(
            llm=judge_llm,
            embeddings=embeddings
        ),

        "context_precision": ContextPrecision(
            llm=judge_llm
        ),

        "context_recall": ContextRecall(
            llm=judge_llm
        ),
    }


# --------------------------------------------------
# 6. SCORE ONE SAMPLE
# --------------------------------------------------

async def evaluate_sample(sample, metrics):
    """
    Run each RAGAS metric independently.
    A failed metric is marked None instead of zero.
    """

    metric_inputs = {
        "faithfulness": {
            "user_input": sample.question,
            "response": sample.answer,
            "retrieved_contexts": sample.contexts,
        },

        "answer_relevancy": {
            "user_input": sample.question,
            "response": sample.answer,
        },

        "context_precision": {
            "user_input": sample.question,
            "reference": sample.ground_truth,
            "retrieved_contexts": sample.contexts,
        },

        "context_recall": {
            "user_input": sample.question,
            "reference": sample.ground_truth,
            "retrieved_contexts": sample.contexts,
        },
    }

    for name, metric in metrics.items():
        try:
            result = await metric.ascore(**metric_inputs[name])
            sample.scores[name] = (
                float(result.value)
                if result.value is not None
                else None
            )

        except Exception:
            logger.exception(
                "Metric '%s' failed for: %s",
                name,
                sample.question
            )
            sample.scores[name] = None

    return sample


# --------------------------------------------------
# 7. PRINT PER-SAMPLE RESULTS
# --------------------------------------------------

def print_sample_results(index, sample):
    print(f"\n{'-' * 70}")
    print(f"SAMPLE {index}")
    print(f"Question: {sample.question}")
    print(f"Answer:   {sample.answer}")
    print(f"Retrieved contexts: {len(sample.contexts)}")

    for context in sample.contexts:
        print(f"  - {context}")

    print("\nRAGAS scores:")

    for name, score in sample.scores.items():
        value = f"{score:.3f}" if score is not None else "N/A"
        print(f"  {name:20s}: {value}")


# --------------------------------------------------
# 8. AGGREGATE STATISTICS AND WEAKEST METRIC
# --------------------------------------------------

def calculate_statistics(samples):
    summaries = {}

    for metric_name in [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]:
        values = [
            sample.scores.get(metric_name)
            for sample in samples
            if sample.scores.get(metric_name) is not None
        ]

        if not values:
            summaries[metric_name] = {
                "count": 0,
                "mean": None,
                "median": None,
                "std": None,
                "min": None,
                "max": None,
            }
            continue

        summaries[metric_name] = {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std": statistics.pstdev(values),
            "min": min(values),
            "max": max(values),
        }

    print(f"\n{'=' * 70}")
    print("AGGREGATE RAGAS RESULTS")
    print(f"{'=' * 70}")

    for name, stats in summaries.items():
        print(f"\n{name}:")

        if stats["mean"] is None:
            print("  No valid scores.")
            continue

        print(f"  Valid samples: {stats['count']}")
        print(f"  Mean:          {stats['mean']:.3f}")
        print(f"  Median:        {stats['median']:.3f}")
        print(f"  Std deviation: {stats['std']:.3f}")
        print(f"  Minimum:       {stats['min']:.3f}")
        print(f"  Maximum:       {stats['max']:.3f}")

    valid_means = {
        name: stats["mean"]
        for name, stats in summaries.items()
        if stats["mean"] is not None
    }

    if valid_means:
        weakest = min(valid_means, key=valid_means.get)

        print(f"\nWEAKEST METRIC: {weakest}")
        print(f"Mean score: {valid_means[weakest]:.3f}")

        hypotheses = {
            "faithfulness": (
                "The answer generator may include claims that "
                "are not supported by retrieved context."
            ),
            "answer_relevancy": (
                "The generated answers may be incomplete, "
                "too broad, or not focused on the question."
            ),
            "context_precision": (
                "Retrieval may return irrelevant chunks or rank "
                "useful information below less relevant material."
            ),
            "context_recall": (
                "Retrieval may miss information required by the "
                "reference answer, or the reference may contain "
                "details absent from the retrieved contexts."
            ),
        }

        print("Hypothesis:", hypotheses[weakest])
        print(
            "Validate this hypothesis by inspecting the samples "
            "with the lowest scores."
        )

    else:
        print("\nNo aggregate metric could be calculated.")

    return summaries


# --------------------------------------------------
# 9. SAVE JSON AND CSV REPORTS
# --------------------------------------------------

def save_results(samples, summaries):
    output = {
        "judge_model": JUDGE_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "sample_count": len(samples),
        "aggregate_statistics": summaries,
        "samples": [asdict(sample) for sample in samples],
    }

    with open(JSON_OUTPUT, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    fieldnames = [
        "question",
        "ground_truth",
        "answer",
        "contexts",
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]

    with open(
        CSV_OUTPUT, "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for sample in samples:
            writer.writerow({
                "question": sample.question,
                "ground_truth": sample.ground_truth,
                "answer": sample.answer,
                "contexts": " || ".join(sample.contexts),
                **sample.scores,
            })

    print(f"\nDetailed JSON saved to: {JSON_OUTPUT}")
    print(f"Per-sample CSV saved to: {CSV_OUTPUT}")


# --------------------------------------------------
# 10. MAIN EVALUATION RUNNER
# --------------------------------------------------

async def run_evaluation():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "Set OPENAI_API_KEY before running the evaluation."
        )

    pipeline = SimulatedRAGPipeline()
    samples = create_dataset()

    # AsyncOpenAI client is shared by the judge and embedder.
    async with AsyncOpenAI() as client:
        metrics = create_metrics(client)

        print("=" * 70)
        print("RAGAS EVALUATION REPORT")
        print("=" * 70)

        for index, sample in enumerate(samples, start=1):
            response = pipeline.query(sample.question)

            sample.answer = response["answer"]
            sample.contexts = response["contexts"]

            await evaluate_sample(sample, metrics)
            print_sample_results(index, sample)

    summaries = calculate_statistics(samples)
    save_results(samples, summaries)

    return samples


if __name__ == "__main__":
    try:
        asyncio.run(run_evaluation())
    except Exception as error:
        logger.exception("Evaluation could not complete: %s", error)
