"""
SUPERVISOR MULTI-AGENT SYSTEM (Parallel)
Agents: Supervisor, Researcher, Coder, Tester, Reviewer

WORKFLOW:

                         START
                           |
                       SUPERVISOR
                           |
              +------------+------------+
              |                         |
          RESEARCHER                  CODER
              |                         |
              |                       TESTER
              |                         |
              |                 +-------+-------+
              |                 |               |
              |              TEST PASS       TEST FAIL
              |                 |               |
              |                 |        Retries remaining?
              |                 |               |
              |                 |           Yes: CODER
              |                 |           No: Continue
              +-----------------+---------------+
                                |
                            REVIEWER
                                |
                       Revision needed?
                         /           \
                       Yes            No
                        |              |
                      CODER         FINALIZE
                                       |
                                      END

TESTING:
- Initial test + up to 2 retries.
- The reviewer receives research, code, and test results.

PARALLEL EXECUTION:
- Researcher and coder run simultaneously.
- Reviewer waits for both branches to finish.
- Sequential mode is also measured for comparison.
"""

from typing import TypedDict, Annotated, Optional, Any
from langgraph.graph import StateGraph, END
from langgraph.types import Send
import operator
import json


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

MAX_TEST_RETRIES = 2
MAX_REVIEW_ITERATIONS = 5

# Deterministic simulation:
# The first test run fails; the next run passes.
# Set this to 2 to simulate two failed runs before passing.
SIMULATED_FAILURES_BEFORE_PASS = 1


# ------------------------------------------------------------------
# State Schema
# ------------------------------------------------------------------

class SupervisorState(TypedDict):
    messages: Annotated[list[str], operator.add]
    current_agent: str
    original_task: str
    research_results: Optional[str]
    code_output: Optional[str]
    test_results: Optional[dict[str, Any]]
    test_retries: int
    review_feedback: Optional[str]
    final_result: Optional[str]
    iteration: int
    max_iterations: int
    needs_revision: bool


# ------------------------------------------------------------------
# Simulated LLM responses
# ------------------------------------------------------------------

def simulate_llm_call(
    agent_name: str,
    prompt: str,
    context: str = "",
) -> str:
    """Return predefined responses to simulate specialist agents."""

    responses = {
        "researcher": (
            "## Research Findings\n"
            "1. Fibonacci: F(n) = F(n-1) + F(n-2).\n"
            "2. Iterative implementation takes O(n) time.\n"
            "3. Naive recursion takes O(2^n) time.\n"
            "4. Iteration avoids repeated calculations."
        ),
        "coder": (
            "## Code Implementation\n"
            "```python\n"
            "def fibonacci(n: int) -> list[int]:\n"
            '    """Generate n Fibonacci numbers."""\n'
            "    if n <= 0:\n"
            "        return []\n"
            "    if n == 1:\n"
            "        return [0]\n"
            "    fib = [0, 1]\n"
            "    for i in range(2, n):\n"
            "        fib.append(fib[-1] + fib[-2])\n"
            "    return fib\n"
            "```\n"
            "Complexity: O(n) time and O(n) space."
        ),
        "reviewer": (
            "## Code Review\n"
            "Strengths: readable, efficient, and documented.\n"
            "Verdict: APPROVED."
        ),
    }

    return responses.get(agent_name, "No response available.")


# ------------------------------------------------------------------
# Supervisor
# ------------------------------------------------------------------

def supervisor_node(state: SupervisorState) -> dict:
    """Start and coordinate the workflow."""

    message = (
        "[Supervisor] Task received. "
        "Dispatching work to specialist agents."
    )
    print(message)

    return {
        "messages": [message],
        "current_agent": "supervisor",
    }


def dispatch_parallel(state: SupervisorState):
    """Dispatch researcher and coder concurrently using Send."""

    print("[Supervisor] Dispatching researcher and coder in parallel.")

    # Both agents receive the current task and state.
    return [
        Send("researcher", dict(state)),
        Send("coder", dict(state)),
    ]


def dispatch_sequential(state: SupervisorState) -> str:
    """Sequential mode starts with research."""

    return "researcher"


# ------------------------------------------------------------------
# Researcher
# ------------------------------------------------------------------

def researcher_node(state: SupervisorState) -> dict:
    task = state.get("original_task", "General task")

    print("[Researcher] Researching the task...")
    findings = simulate_llm_call("researcher", task)

    return {
        "messages": [f"[Researcher] Research completed.\n{findings}"],
        "research_results": findings,
    }


# ------------------------------------------------------------------
# Coder
# ------------------------------------------------------------------

def coder_node(state: SupervisorState) -> dict:
    """Generate or revise code."""

    task = state.get("original_task", "")
    previous_agent = state.get("current_agent", "")
    retries = state.get("test_retries", 0)

    # A test failure sends control back from tester to coder.
    if previous_agent == "tester":
        retries += 1

    # A reviewer-requested revision starts a fresh test cycle.
    elif previous_agent == "reviewer":
        retries = 0

    print("[Coder] Generating or revising code...")

    code = simulate_llm_call(
        "coder",
        task,
        state.get("research_results") or "",
    )

    return {
        "messages": [f"[Coder] Code generated.\n{code}"],
        "current_agent": "coder",
        "code_output": code,
        "test_retries": retries,
    }


# ------------------------------------------------------------------
# Tester
# ------------------------------------------------------------------

def tester_node(state: SupervisorState) -> dict:
    """
    Generate test cases and simulate executing them.

    This is a simulation, not execution of arbitrary Python code.
    """

    code = state.get("code_output") or ""
    task = state.get("original_task", "")
    retries = state.get("test_retries", 0)

    print("[Tester] Generating test cases and simulating test execution...")

    # Illustrative cases. A real tester would derive these from
    # the task requirements and execute them against the code.
    test_cases = [
        {
            "name": "Normal input",
            "purpose": f"Check expected behavior for: {task}",
        },
        {
            "name": "Edge case",
            "purpose": "Check empty or boundary input",
        },
        {
            "name": "Output validation",
            "purpose": "Check that output follows the expected format",
        },
    ]

    # Deterministic simulated outcome makes parallel/sequential
    # graph-step comparisons reproducible.
    passed = (
        bool(code)
        and retries >= SIMULATED_FAILURES_BEFORE_PASS
    )

    status = "PASS" if passed else "FAIL"

    for case in test_cases:
        case["status"] = status

    results = {
        "passed": len(test_cases) if passed else 0,
        "failed": 0 if passed else len(test_cases),
        "total": len(test_cases),
        "attempt": retries + 1,
        "cases": test_cases,
    }

    message = (
        f"[Tester] Attempt {results['attempt']}: "
        f"{results['passed']} passed, "
        f"{results['failed']} failed."
    )
    print(message)

    return {
        "messages": [message],
        "current_agent": "tester",
        "test_results": results,
    }


def route_after_tests(state: SupervisorState) -> str:
    """Retry failed tests, or continue when the retry budget is used."""

    results = state.get("test_results") or {}
    failed = results.get("failed", 0)
    retries = state.get("test_retries", 0)

    if failed > 0 and retries < MAX_TEST_RETRIES:
        print(
            f"[Supervisor] Tests failed. "
            f"Sending code back for retry {retries + 1}."
        )
        return "retry"

    if failed > 0:
        print("[Supervisor] Test retry limit reached. Continuing to review.")

    # On the first pass through testing, the reviewer must wait for
    # the research branch too. After a review revision, research is
    # already available, so the reviewer can be called directly.
    if state.get("review_feedback"):
        return "review"

    return "complete"


# ------------------------------------------------------------------
# Fan-in gate
# ------------------------------------------------------------------

def initial_test_complete_node(state: SupervisorState) -> dict:
    """Mark the initial coding/testing branch as ready for review."""

    message = "[Workflow] Initial coding/testing branch completed."

    return {
        "messages": [message],
        "current_agent": "test_complete",
    }


# ------------------------------------------------------------------
# Reviewer
# ------------------------------------------------------------------

def reviewer_node(state: SupervisorState) -> dict:
    """Review research, code, and test results together."""

    research = state.get("research_results") or "No research available."
    code = state.get("code_output") or "No code available."
    test_results = state.get("test_results") or {}

    print("[Reviewer] Reviewing research, code, and test results...")

    review = simulate_llm_call(
        "reviewer",
        state.get("original_task", ""),
        code,
    )

    review += (
        "\n\nTest summary: "
        f"{test_results.get('passed', 0)} passed, "
        f"{test_results.get('failed', 0)} failed."
    )

    approved = "APPROVED" in review
    iteration = state.get("iteration", 0) + 1

    return {
        "messages": [
            f"[Reviewer] Review completed.\n{review}"
        ],
        "current_agent": "reviewer",
        "review_feedback": review,
        "needs_revision": not approved,
        "iteration": iteration,
    }


def route_after_review(state: SupervisorState) -> str:
    """Decide whether to request a revision or finalize."""

    if (
        state.get("needs_revision", False)
        and state.get("iteration", 0)
        < state.get("max_iterations", MAX_REVIEW_ITERATIONS)
    ):
        return "revise"

    return "finalize"


# ------------------------------------------------------------------
# Finalizer
# ------------------------------------------------------------------

def finalize_node(state: SupervisorState) -> dict:
    print("[Finalizer] Combining specialist outputs...")

    final = (
        "=== FINAL DELIVERABLE ===\n\n"
        f"## Original Task\n{state.get('original_task', 'N/A')}\n\n"
        f"## Research\n{state.get('research_results') or 'N/A'}\n\n"
        f"## Implementation\n{state.get('code_output') or 'N/A'}\n\n"
        "## Test Results\n"
        f"{json.dumps(state.get('test_results'), indent=2)}\n\n"
        f"## Review\n{state.get('review_feedback') or 'N/A'}\n\n"
        f"## Review Iterations\n{state.get('iteration', 0)}\n"
        "=== END DELIVERABLE ==="
    )

    return {
        "messages": ["[Finalizer] Task complete."],
        "current_agent": "finalize",
        "final_result": final,
    }


# ------------------------------------------------------------------
# Build graph
# ------------------------------------------------------------------

def build_multi_agent_system(parallel: bool = True):
    """Build parallel or sequential versions of the same workflow."""

    workflow = StateGraph(SupervisorState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("coder", coder_node)
    workflow.add_node("tester", tester_node)
    workflow.add_node(
        "initial_test_complete",
        initial_test_complete_node,
    )
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("finalize", finalize_node)

    workflow.set_entry_point("supervisor")

    if parallel:
        # Fan-out: researcher and coder are dispatched together.
        workflow.add_conditional_edges(
            "supervisor",
            dispatch_parallel,
        )

        # Fan-in: reviewer waits for BOTH research and initial testing.
        workflow.add_edge(
            ["researcher", "initial_test_complete"],
            "reviewer",
        )

    else:
        # Baseline workflow: researcher runs before coder.
        workflow.add_conditional_edges(
            "supervisor",
            dispatch_sequential,
            {"researcher": "researcher"},
        )

        workflow.add_edge("researcher", "coder")
        workflow.add_edge("initial_test_complete", "reviewer")

    # Coder must always pass through tester.
    workflow.add_edge("coder", "tester")

    workflow.add_conditional_edges(
        "tester",
        route_after_tests,
        {
            "retry": "coder",
            "complete": "initial_test_complete",
            "review": "reviewer",
        },
    )

    workflow.add_conditional_edges(
        "reviewer",
        route_after_review,
        {
            "revise": "coder",
            "finalize": "finalize",
        },
    )

    workflow.add_edge("finalize", END)

    return workflow.compile()


# ------------------------------------------------------------------
# Execution and measurement
# ------------------------------------------------------------------

def make_initial_state(task: str) -> SupervisorState:
    return {
        "messages": [],
        "current_agent": "",
        "original_task": task,
        "research_results": None,
        "code_output": None,
        "test_results": None,
        "test_retries": 0,
        "review_feedback": None,
        "final_result": None,
        "iteration": 0,
        "max_iterations": MAX_REVIEW_ITERATIONS,
        "needs_revision": False,
    }


def run_and_measure(app, initial_state):
    """
    Count graph supersteps using full-state streaming.

    Parallel nodes may execute in the same superstep. Message count
    estimates node executions because each node appends one message.
    """
    supersteps = 0
    final_state = None

    for snapshot in app.stream(
        initial_state,
        stream_mode="values",
    ):
        supersteps += 1
        final_state = snapshot

    if final_state is None:
        raise RuntimeError("The graph produced no state snapshots.")

    node_executions = len(final_state.get("messages", []))

    return final_state, supersteps, node_executions


def main():
    tasks = [
        "Implement an efficient Fibonacci sequence generator in Python",
        "Create a data validation utility for CSV files",
    ]

    parallel_app = build_multi_agent_system(parallel=True)
    sequential_app = build_multi_agent_system(parallel=False)

    print("=" * 65)
    print("SUPERVISOR MULTI-AGENT SYSTEM")
    print("=" * 65)

    for index, task in enumerate(tasks, start=1):
        print(f"\nTASK {index}: {task}\n")

        # Use separate initial states for fair comparison.
        parallel_state, parallel_steps, parallel_nodes = (
            run_and_measure(
                parallel_app,
                make_initial_state(task),
            )
        )

        sequential_state, sequential_steps, sequential_nodes = (
            run_and_measure(
                sequential_app,
                make_initial_state(task),
            )
        )

        saved_steps = sequential_steps - parallel_steps

        if sequential_steps:
            reduction = saved_steps / sequential_steps * 100
        else:
            reduction = 0.0

        print("\n--- Execution Comparison ---")
        print(
            f"Parallel:   {parallel_steps} supersteps, "
            f"{parallel_nodes} node executions"
        )
        print(
            f"Sequential: {sequential_steps} supersteps, "
            f"{sequential_nodes} node executions"
        )
        print(f"Supersteps saved: {saved_steps}")
        print(f"Superstep reduction: {reduction:.1f}%")

        print("\n--- Test Results (Parallel Run) ---")
        print(
            json.dumps(
                parallel_state.get("test_results"),
                indent=2,
            )
        )

        print("\n--- Final Deliverable (Parallel Run) ---")
        print(parallel_state.get("final_result", "No result."))

    print("\nExecution comparison complete.")


if __name__ == "__main__":
    main()