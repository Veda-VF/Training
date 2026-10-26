"""
TOPIC: Reflection in Agent Fundamentals (Module 1)

A benchmark to measures whether a Reflexion agent actually improves its answers when it critiques and revises its own work 
and whether those improvements justify the extra API calls.

20 benchmark questions (5 factual · 5 creative · 5 analytical · 5 mathematical)
    |
Generate baseline answer — Round 0
    |
Score answer using an evaluation model (Category-specific rubric · score from 0 to 10)
    |
Reflexion loop — 5 rounds (Critique → identify weaknesses → revise → score)
    |
Analyze score improvements
    |
Final benchmark report (Optimal rounds by category and usage recommendations)


What counts as a round?
- Round 0: Generate and score the original answer.
- Round 1: Reflect on the original answer, revise it, and score the revision.
- Rounds 2–5: Repeat the process using the latest answer and its feedback.
That gives us 120 scored answers (20 questions × 6 versions)

python -m pip install openai
$env:OPENAI_API_KEY = "your-api-key"

"""

import os
import csv
import json
import asyncio
import logging
import statistics
from collections import defaultdict

from openai import AsyncOpenAI


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_ROUNDS = 5
SCORE_MIN = 0.0
SCORE_MAX = 10.0

# A gain below this value counts as small improvement.
DIMINISHING_GAIN_THRESHOLD = 0.2

JSON_FILE = "reflection_benchmark_results.json"
CSV_FILE = "reflection_benchmark_results.csv"

# Set these environment variables to current prices for your model.
# Values are USD per 1 million tokens.
INPUT_PRICE_PER_1M = float(
    os.getenv("INPUT_PRICE_PER_1M", "0")
)
OUTPUT_PRICE_PER_1M = float(
    os.getenv("OUTPUT_PRICE_PER_1M", "0")
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================
# 2. CUSTOM DATASET: 20 QUESTIONS, 4 CATEGORIES
# ============================================================

QUESTIONS = [
    # ---------------- FACTUAL (5) ----------------
    {
        "id": "F1",
        "category": "factual",
        "question": "What is the difference between a compiler and an interpreter?",
        "reference": (
            "A compiler translates a program into another form, often "
            "machine code, before execution. An interpreter executes "
            "program instructions through an interpreter at runtime. "
            "Real language implementations may combine both approaches."
        ),
        "rubric": (
            "Explain compilation versus interpretation accurately. "
            "Mention when translation/execution happens and avoid the "
            "false claim that every language uses only one approach."
        ),
    },
    {
        "id": "F2",
        "category": "factual",
        "question": "What is the purpose of a primary key in a relational database?",
        "reference": (
            "A primary key uniquely identifies each row in a table. "
            "Its values must be unique and cannot be NULL. A table "
            "has one primary-key constraint, which may contain multiple columns."
        ),
        "rubric": (
            "Explain unique row identification, uniqueness, NULL restriction, "
            "and optionally composite keys."
        ),
    },
    {
        "id": "F3",
        "category": "factual",
        "question": "What is the difference between supervised and unsupervised learning?",
        "reference": (
            "Supervised learning trains on labeled examples containing "
            "inputs and target outputs. Unsupervised learning finds "
            "patterns or structure in data without target labels."
        ),
        "rubric": (
            "Correctly distinguish labeled targets from unlabeled data "
            "and give an accurate example of each."
        ),
    },
    {
        "id": "F4",
        "category": "factual",
        "question": "Why is HTTP considered stateless?",
        "reference": (
            "HTTP is stateless because each request is independent by "
            "default; the protocol does not inherently retain application "
            "state from one request to the next. Applications can use "
            "cookies, sessions, or tokens to maintain state."
        ),
        "rubric": (
            "Explain independent requests and clarify that sessions, "
            "cookies, or tokens can add application-level state."
        ),
    },
    {
        "id": "F5",
        "category": "factual",
        "question": "What role does an operating system kernel play?",
        "reference": (
            "The kernel is the core of an operating system. It manages "
            "resources such as CPU time and memory, handles system calls, "
            "and coordinates access to hardware and other system services."
        ),
        "rubric": (
            "Cover the kernel's role in resource management, hardware "
            "coordination, and the interface between software and system resources."
        ),
    },

    # ---------------- CREATIVE (5) ----------------
    {
        "id": "C1",
        "category": "creative",
        "question": "Invent a memorable name and tagline for an app that helps students focus.",
        "reference": (
            "A distinctive, memorable app name and a concise tagline "
            "clearly connected to focus, study, or productivity."
        ),
        "rubric": (
            "Score originality, memorability, relevance to students, "
            "clarity of the tagline, and suitability as a product brand. "
            "Do not require one specific answer."
        ),
    },
    {
        "id": "C2",
        "category": "creative",
        "question": "Write a four-line poem about a robot discovering a garden.",
        "reference": (
            "A four-line poem showing a robot encountering a garden, "
            "with vivid imagery and a sense of discovery."
        ),
        "rubric": (
            "Evaluate four-line format, imagery, coherent theme, originality, "
            "and emotional or imaginative impact."
        ),
    },
    {
        "id": "C3",
        "category": "creative",
        "question": "Suggest an unusual but practical way to reduce food waste in a college cafeteria.",
        "reference": (
            "A feasible, original intervention that reduces food waste "
            "and could realistically operate in a college cafeteria."
        ),
        "rubric": (
            "Evaluate novelty, feasibility, waste reduction, student participation, "
            "and a clear implementation plan."
        ),
    },
    {
        "id": "C4",
        "category": "creative",
        "question": "Create a short story premise in which a library remembers its visitors.",
        "reference": (
            "A compelling story premise in which a library somehow remembers "
            "its visitors, with a clear central conflict or narrative hook."
        ),
        "rubric": (
            "Evaluate originality, clarity, imaginative use of the library concept, "
            "and presence of a meaningful conflict or hook."
        ),
    },
    {
        "id": "C5",
        "category": "creative",
        "question": "Design a low-cost campus campaign encouraging students to use reusable bottles.",
        "reference": (
            "A low-cost, practical campaign that encourages reusable-bottle "
            "use through clear messaging and engaging participation."
        ),
        "rubric": (
            "Evaluate feasibility, low cost, appeal to students, creativity, "
            "and a measurable way to assess participation or impact."
        ),
    },

    # ---------------- ANALYTICAL (5) ----------------
    {
        "id": "A1",
        "category": "analytical",
        "question": (
            "A website has high traffic but few purchases. "
            "Identify likely causes and propose a diagnostic plan."
        ),
        "reference": (
            "Investigate traffic quality, page speed, usability, pricing, "
            "trust, checkout errors, and conversion-funnel drop-offs. "
            "Use analytics and controlled tests to identify causes."
        ),
        "rubric": (
            "Include multiple plausible causes, data to inspect, a prioritized "
            "diagnostic sequence, and evidence-based experiments."
        ),
    },
    {
        "id": "A2",
        "category": "analytical",
        "question": (
            "A machine-learning model performs well on training data "
            "but poorly on test data. Explain why and propose remedies."
        ),
        "reference": (
            "This suggests overfitting or a data-distribution mismatch. "
            "Investigate leakage, split quality, and distribution shift. "
            "Potential remedies include regularization, simpler models, "
            "more representative data, and appropriate validation."
        ),
        "rubric": (
            "Explain overfitting, consider data leakage or distribution shift, "
            "and propose relevant diagnostics and remedies."
        ),
    },
    {
        "id": "A3",
        "category": "analytical",
        "question": (
            "A team must choose between building a feature quickly "
            "and delaying release for more testing. Analyze the trade-off."
        ),
        "reference": (
            "Compare user value and urgency against failure probability, "
            "impact, reversibility, and testing needs. Consider staged rollout, "
            "monitoring, and rollback as possible compromises."
        ),
        "rubric": (
            "Identify competing objectives, assess risk and impact, explain "
            "decision criteria, and offer a practical risk-reduction option."
        ),
    },
    {
        "id": "A4",
        "category": "analytical",
        "question": (
            "A city wants to reduce traffic congestion. Compare public "
            "transport improvements with congestion pricing."
        ),
        "reference": (
            "Public transport can increase alternatives to driving but requires "
            "capacity and investment. Congestion pricing can reduce peak demand "
            "but raises equity and implementation concerns. Context and policy "
            "design determine effectiveness."
        ),
        "rubric": (
            "Compare mechanisms, benefits, limitations, equity, implementation, "
            "and how evidence could guide the decision."
        ),
    },
    {
        "id": "A5",
        "category": "analytical",
        "question": (
            "Customer satisfaction rises after a product redesign. "
            "Can the redesign be credited with the improvement? Explain."
        ),
        "reference": (
            "The timing alone does not prove causation. Consider seasonality, "
            "customer mix, other changes, survey bias, and sample size. "
            "A randomized experiment or careful comparison can strengthen "
            "causal evidence."
        ),
        "rubric": (
            "Distinguish correlation from causation, identify confounders, "
            "and propose a credible evaluation design."
        ),
    },

    # ---------------- MATHEMATICAL (5) ----------------
    {
        "id": "M1",
        "category": "mathematical",
        "question": "Solve 3x + 7 = 25 and show the steps.",
        "reference": (
            "Subtract 7 from both sides: 3x = 18. Divide by 3: x = 6."
        ),
        "rubric": (
            "Correct result x=6, valid algebraic steps, and a clear explanation."
        ),
    },
    {
        "id": "M2",
        "category": "mathematical",
        "question": (
            "A bag has 4 red and 6 blue balls. What is the probability "
            "of drawing a red ball on one random draw?"
        ),
        "reference": (
            "There are 10 balls in total. The probability of red is "
            "4/10 = 2/5 = 0.4 = 40%."
        ),
        "rubric": (
            "Use favorable outcomes divided by total outcomes and obtain "
            "2/5 or 0.4 or 40%."
        ),
    },
    {
        "id": "M3",
        "category": "mathematical",
        "question": (
            "A product costs 800 units and its price increases by 15%. "
            "What is the new price?"
        ),
        "reference": (
            "15% of 800 is 120. New price = 800 + 120 = 920 units."
        ),
        "rubric": (
            "Calculate the increase correctly and obtain 920 units."
        ),
    },
    {
        "id": "M4",
        "category": "mathematical",
        "question": (
            "Find the mean and median of 2, 4, 7, 9, 13."
        ),
        "reference": (
            "Mean = (2+4+7+9+13)/5 = 35/5 = 7. "
            "Median = 7, the middle value in the sorted list."
        ),
        "rubric": (
            "Correctly calculate both mean and median as 7 and explain them."
        ),
    },
    {
        "id": "M5",
        "category": "mathematical",
        "question": (
            "A car travels 150 km in 3 hours at constant speed. "
            "How long would it take to travel 250 km at that speed?"
        ),
        "reference": (
            "Speed = 150/3 = 50 km/h. Time = 250/50 = 5 hours."
        ),
        "rubric": (
            "Calculate 50 km/h and then 5 hours with correct units."
        ),
    },
]


# ============================================================
# 3. TRACK API CALLS AND TOKEN USAGE
# ============================================================

class UsageTracker:
    def __init__(self):
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def record(self, response):
        self.calls += 1
        usage = response.usage

        if usage:
            self.input_tokens += usage.prompt_tokens or 0
            self.output_tokens += usage.completion_tokens or 0

    def report(self):
        estimated_cost = (
            self.input_tokens * INPUT_PRICE_PER_1M / 1_000_000
            + self.output_tokens * OUTPUT_PRICE_PER_1M / 1_000_000
        )

        return {
            "api_calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost_usd": round(estimated_cost, 6),
            "pricing_configured": (
                INPUT_PRICE_PER_1M > 0
                or OUTPUT_PRICE_PER_1M > 0
            ),
        }


# ============================================================
# 4. REFLEXION AGENT
# ============================================================

class ReflexionAgent:
    def __init__(self, client, tracker):
        self.client = client
        self.tracker = tracker

    async def call_json(self, system_prompt, user_prompt):
        response = await self.client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )

        self.tracker.record(response)
        content = response.choices[0].message.content

        if not content:
            raise ValueError("The model returned an empty response.")

        return json.loads(content)

    async def initial_answer(self, item):
        result = await self.call_json(
            system_prompt=(
                "You are a careful assistant. Answer the question "
                "directly, clearly, and accurately. Return JSON."
            ),
            user_prompt=json.dumps({
                "question": item["question"],
                "category": item["category"],
                "reference_criteria": item["reference"],
                "required_format": {
                    "answer": "your answer"
                },
            }),
        )

        answer = result.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Initial answer was missing or empty.")

        return answer.strip()

    async def reflect_and_revise(
        self, item, previous_answer, previous_feedback, round_number
    ):
        result = await self.call_json(
            system_prompt=(
                "You are a Reflexion agent. Critically review your previous "
                "answer, identify concrete weaknesses, and produce an improved "
                "answer. Do not change correct content without reason. "
                "Do not claim unsupported facts. Return JSON."
            ),
            user_prompt=json.dumps({
                "question": item["question"],
                "category": item["category"],
                "reference_criteria": item["reference"],
                "rubric": item["rubric"],
                "round": round_number,
                "previous_answer": previous_answer,
                "previous_evaluator_feedback": previous_feedback,
                "instructions": [
                    "State the most important weakness.",
                    "Check factual accuracy and completeness.",
                    "For math, verify each calculation.",
                    "For creative work, preserve good ideas while improving originality.",
                    "For analytical work, check assumptions and alternatives.",
                    "Return a revised answer, not only a critique."
                ],
                "required_format": {
                    "reflection": "specific critique and improvement plan",
                    "revised_answer": "complete revised answer"
                }
            }),
        )

        reflection = result.get("reflection", "")
        revised_answer = result.get("revised_answer")

        if not isinstance(revised_answer, str) or not revised_answer.strip():
            raise ValueError("Revised answer was missing or empty.")

        return reflection, revised_answer.strip()


# ============================================================
# 5. INDEPENDENT LLM JUDGE
# ============================================================

class AnswerJudge:
    def __init__(self, client, tracker):
        self.client = client
        self.tracker = tracker

    async def score(self, item, answer):
        prompt = {
            "question": item["question"],
            "category": item["category"],
            "reference_answer_or_criteria": item["reference"],
            "scoring_rubric": item["rubric"],
            "candidate_answer": answer,
            "instructions": [
                "Score answer quality from 0 to 10.",
                "10 means fully correct or excellent for its category.",
                "0 means entirely incorrect, irrelevant, or unusable.",
                "For creative tasks, judge against the rubric, not exact wording.",
                "For math, verify the calculations and final result.",
                "Do not reward verbosity unless it improves the answer.",
                "Be consistent across rounds.",
                "Do not give a higher score merely because the answer is longer."
            ],
            "required_format": {
                "score": 0,
                "reason": "short explanation"
            }
        }

        response = await self.client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict, impartial answer evaluator. "
                        "Follow the supplied rubric and return JSON."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt)},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )

        self.tracker.record(response)
        content = response.choices[0].message.content

        if not content:
            raise ValueError("Judge returned an empty response.")

        result = json.loads(content)
        score = float(result["score"])

        if not SCORE_MIN <= score <= SCORE_MAX:
            raise ValueError(f"Score outside 0-10: {score}")

        return score, result.get("reason", "")


# ============================================================
# 6. RUN ONE QUESTION THROUGH ROUND 0 TO ROUND 5
# ============================================================

async def run_one_question(item, agent, judge):
    answer = await agent.initial_answer(item)
    feedback = ""
    history = []

    # Score the initial answer as round 0.
    score, reason = await judge.score(item, answer)

    history.append({
        "round": 0,
        "answer": answer,
        "reflection": "",
        "score": score,
        "judge_feedback": reason,
    })

    feedback = reason

    for round_number in range(1, MAX_ROUNDS + 1):
        reflection, revised_answer = await agent.reflect_and_revise(
            item=item,
            previous_answer=answer,
            previous_feedback=feedback,
            round_number=round_number,
        )

        score, reason = await judge.score(item, revised_answer)

        history.append({
            "round": round_number,
            "answer": revised_answer,
            "reflection": reflection,
            "score": score,
            "judge_feedback": reason,
        })

        answer = revised_answer
        feedback = reason

        logger.info(
            "%s | round %d | score %.2f",
            item["id"],
            round_number,
            score
        )

    return {
        "id": item["id"],
        "category": item["category"],
        "question": item["question"],
        "reference": item["reference"],
        "history": history,
    }


# ============================================================
# 7. AGGREGATE CATEGORY SCORES
# ============================================================

def summarize(results):
    categories = sorted({result["category"] for result in results})
    summary = {}

    for category in categories:
        category_results = [
            result for result in results
            if result["category"] == category
        ]

        mean_by_round = {}

        for round_number in range(MAX_ROUNDS + 1):
            scores = [
                result["history"][round_number]["score"]
                for result in category_results
            ]
            mean_by_round[round_number] = statistics.mean(scores)

        gains = {
            round_number: (
                mean_by_round[round_number]
                - mean_by_round[round_number - 1]
            )
            for round_number in range(1, MAX_ROUNDS + 1)
        }

        # Choose the round with the highest category-average score.
        # Ties go to the earliest round to avoid unnecessary cost.
        optimal_round = max(
            mean_by_round,
            key=lambda r: (mean_by_round[r], -r)
        )

        # First round at which two consecutive gains are both small.
        diminishing_from = None

        for round_number in range(2, MAX_ROUNDS + 1):
            if (
                gains[round_number - 1] < DIMINISHING_GAIN_THRESHOLD
                and gains[round_number] < DIMINISHING_GAIN_THRESHOLD
            ):
                diminishing_from = round_number - 1
                break

        summary[category] = {
            "mean_by_round": mean_by_round,
            "marginal_gain_by_round": gains,
            "optimal_round": optimal_round,
            "diminishing_returns_from": diminishing_from,
            "baseline_mean": mean_by_round[0],
            "best_mean": mean_by_round[optimal_round],
            "total_gain": (
                mean_by_round[optimal_round] - mean_by_round[0]
            ),
        }

    return summary


# ============================================================
# 8. PRINT REPORT AND RECOMMENDATIONS
# ============================================================

def print_report(summary, results, usage):
    print("\n" + "=" * 100)
    print("REFLECTION EFFECTIVENESS BENCHMARK")
    print("=" * 100)

    print(
        f"\n{'Category':<15}"
        f"{'R0':>7}{'R1':>7}{'R2':>7}{'R3':>7}"
        f"{'R4':>7}{'R5':>7}{'Best':>8}{'Gain':>9}"
        f"{'Plateau':>11}"
    )
    print("-" * 100)

    for category, data in summary.items():
        means = data["mean_by_round"]
        plateau = data["diminishing_returns_from"]
        plateau_text = str(plateau) if plateau is not None else "Not seen"

        print(
            f"{category:<15}"
            + "".join(f"{means[r]:>7.2f}" for r in range(6))
            + f"{data['optimal_round']:>8}"
            + f"{data['total_gain']:>9.2f}"
            + f"{plateau_text:>11}"
        )

    print("\nMarginal score improvement by category:")
    for category, data in summary.items():
        gains = data["marginal_gain_by_round"]
        print(
            f"  {category:<15}: "
            + ", ".join(
                f"R{r} {gains[r]:+.2f}"
                for r in range(1, MAX_ROUNDS + 1)
            )
        )

    print("\nPER-QUESTION SCORES")
    print("-" * 100)

    for result in results:
        scores = [
            entry["score"] for entry in result["history"]
        ]
        print(
            f"{result['id']} ({result['category']:<12}) "
            f"scores={scores} "
            f"gain={scores[-1] - scores[0]:+.2f}"
        )

    print("\nCOST-BENEFIT ANALYSIS")
    print("-" * 100)

    # Per question: 1 initial answer, 5 revise calls, 6 judge calls.
    expected_calls = len(results) * (1 + MAX_ROUNDS + MAX_ROUNDS + 1)

    # The expression above is 1 + 5 revisions + 6 scores = 12 calls/question.
    # For 20 questions, expected total is 240 calls.
    print(f"Total API calls recorded: {usage['api_calls']}")
    print(f"Expected calls for complete run: {expected_calls}")
    print(f"Input tokens:  {usage['input_tokens']:,}")
    print(f"Output tokens: {usage['output_tokens']:,}")

    if usage["pricing_configured"]:
        print(f"Estimated API cost: ${usage['estimated_cost_usd']:.4f} USD")
    else:
        print(
            "Estimated API cost: not calculated because pricing is unset. "
            "Configure INPUT_PRICE_PER_1M and OUTPUT_PRICE_PER_1M."
        )

    print("\nRecommendations based on observed results:")

    for category, data in summary.items():
        best_round = data["optimal_round"]
        gain = data["total_gain"]
        plateau = data["diminishing_returns_from"]

        if best_round == 0:
            recommendation = (
                "Use no reflection by default; it did not improve the "
                "best category-average score."
            )
        elif plateau is not None and best_round <= plateau:
            recommendation = (
                f"Usually stop after about {best_round} round(s); "
                "later rounds add little average benefit."
            )
        else:
            recommendation = (
                f"Consider up to {best_round} round(s), then stop if "
                "the score no longer improves."
            )

        print(
            f"  {category}: best round={best_round}, "
            f"gain={gain:+.2f}/10. {recommendation}"
        )

    print(
        "\nInterpretation note: these are LLM-judge scores from one run. "
        "Small score differences may reflect judge variability."
    )


# ============================================================
# 9. SAVE RESULTS
# ============================================================

def save_results(results, summary, usage):
    with open(JSON_FILE, "w", encoding="utf-8") as file:
        json.dump(
            {
                "model": MODEL,
                "max_refinement_rounds": MAX_ROUNDS,
                "score_scale": [SCORE_MIN, SCORE_MAX],
                "usage": usage,
                "summary": summary,
                "results": results,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    with open(CSV_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "id", "category", "question", "round",
                "score", "score_change_from_previous",
                "answer", "reflection", "judge_feedback",
            ],
        )
        writer.writeheader()

        for result in results:
            previous_score = None

            for entry in result["history"]:
                writer.writerow({
                    "id": result["id"],
                    "category": result["category"],
                    "question": result["question"],
                    "round": entry["round"],
                    "score": entry["score"],
                    "score_change_from_previous": (
                        ""
                        if previous_score is None
                        else entry["score"] - previous_score
                    ),
                    "answer": entry["answer"],
                    "reflection": entry["reflection"],
                    "judge_feedback": entry["judge_feedback"],
                })

                previous_score = entry["score"]

    print(f"\nSaved JSON report: {JSON_FILE}")
    print(f"Saved per-round CSV: {CSV_FILE}")


# ============================================================
# 10. MAIN
# ============================================================

async def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "Set OPENAI_API_KEY in your environment first."
        )

    if len(QUESTIONS) != 20:
        raise ValueError("Benchmark must contain exactly 20 questions.")

    category_counts = defaultdict(int)
    for item in QUESTIONS:
        category_counts[item["category"]] += 1

    expected_categories = {
        "factual", "creative", "analytical", "mathematical"
    }

    if (
        set(category_counts) != expected_categories
        or any(count != 5 for count in category_counts.values())
    ):
        raise ValueError(
            "Expected exactly 5 questions in each of 4 categories."
        )

    tracker = UsageTracker()
    results = []

    async with AsyncOpenAI() as client:
        agent = ReflexionAgent(client, tracker)
        judge = AnswerJudge(client, tracker)

        for index, item in enumerate(QUESTIONS, start=1):
            print(
                f"\n[{index}/20] {item['id']} "
                f"({item['category']})"
            )

            try:
                result = await run_one_question(item, agent, judge)
                results.append(result)

            except Exception as error:
                logger.exception(
                    "Question %s failed: %s",
                    item["id"],
                    error
                )

    if not results:
        raise RuntimeError("No questions completed successfully.")

    summary = summarize(results)
    usage = tracker.report()

    print_report(summary, results, usage)
    save_results(results, summary, usage)


if __name__ == "__main__":
    asyncio.run(main())

"""
Understand the report
The benchmark produces a table like this. The numbers below are illustrative, not measured API results.
Category	    R0	R1	  R2	R3	  R4	R5	Optimal rounds
Factual	       7.2	8.1	 8.5	8.5	  8.4	8.4	   2
Creative	   6.8	7.4	 8.0	8.3	  8.5	8.5	   4
Analytical     6.5	7.5	 8.2	8.6	  8.8	8.8	   4
Mathematical   7.5	8.6	 9.0	9.0	  9.0	9.0	   2

- R0–R5: Mean score at each stage, on a scale of 0–10.
- Optimal rounds: The round with the highest mean score. Ties go to the earliest round.
- Marginal improvement: The difference between the mean score in the current round and the previous round.
- Diminishing returns: The first point where two consecutive marginal gains are both below 0.2 points.
For example, factual answers improve by 0.9 points in Round 1 and 0.4 in Round 2. If later rounds stop improving, continuing reflection adds cost without meaningful gains."""
