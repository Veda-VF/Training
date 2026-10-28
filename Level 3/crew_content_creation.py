"""
CrewAI Content Creation Crew
Run:
  python -m pip install -U crewai ddgs
  $env:OPENAI_API_KEY = "your-api-key"   # PowerShell
  python crew_content_creation.py

Put context.txt beside this script to provide optional local context.
This script uses real LLM calls and web search; it does not send data to email.

********CREATE A context.txt file in the same folder:*********

Sample context file for the CrewAI researcher
==============================================

AI agents are software systems that can use tools and perform multi-step tasks toward a goal. They may combine a language model with tools, memory, planning, and a control loop.

Research reminders:
- Distinguish a prototype demonstration from a deployed production system.
- In healthcare, distinguish administrative support from clinical decision-making.
- Do not claim an AI system improves patient outcomes unless credible evidence supports the claim.
- Prefer primary sources, peer-reviewed research, government guidance, and technical documentation.
- If a claim cannot be corroborated, label it unverified instead of guessing.

Suggested article angle:
Explain where AI agents may be useful, what can go wrong, and what governance is needed. This file is background context, not independent evidence.
"""

from __future__ import annotations
import os
import time
from pathlib import Path
from typing import Any
from crewai import Agent, Crew, Process, Task
from crewai.tools import tool

try:
    from ddgs import DDGS
except ImportError:
    DDGS = None

BASE_DIR = Path(__file__).resolve().parent
CONTEXT_FILE = BASE_DIR / "context.txt"
MODEL = os.getenv("CREWAI_MODEL", "openai/gpt-4o")


# ----------------------------- Custom tools -----------------------------

@tool("Read and summarize a local text file")
def read_and_summarize_file(file_path: str) -> str:
    """Read a .txt, .md, or .markdown file and return a short summary plus content."""
    path = Path(file_path).expanduser()
    if not path.is_absolute():
        path = (BASE_DIR / path).resolve()
    if path.suffix.lower() not in {".txt", ".md", ".markdown"}:
        return "ERROR: Only .txt, .md, and .markdown files are supported."
    if not path.is_file():
        return f"ERROR: File not found: {path}"
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return f"ERROR reading file: {type(exc).__name__}: {exc}"
    if not content.strip():
        return f"File {path.name} is empty."
    normalized = " ".join(line.strip() for line in content.splitlines() if line.strip())
    parts = [p.strip() for p in normalized.replace("? ", ". ").replace("! ", ". ").split(". ") if p.strip()]
    summary = ". ".join(parts[:5])
    if summary and not summary.endswith((".", "!", "?")):
        summary += "."
    return (
        f"FILE: {path}\nSUMMARY (extractive, first five sentence-like units):\n{summary}\n\n"
        f"FULL CONTENT:\n{content[:20000]}"
        + ("\n[Truncated at 20,000 characters.]" if len(content) > 20000 else "")
    )


@tool("Search the web for corroborating sources")
def search_web_for_claims(query: str) -> str:
    """Search the web and return result titles, URLs, and snippets for fact-checking."""
    if DDGS is None:
        return "ERROR: Install web-search dependency with `python -m pip install -U ddgs`."
    try:
        with DDGS() as client:
            results = list(client.text(query, max_results=5))
    except Exception as exc:
        return f"ERROR: Search failed: {type(exc).__name__}: {exc}"
    if not results:
        return f"No results found for: {query}"
    lines = [f"Query: {query}"]
    for i, item in enumerate(results, 1):
        lines.append(
            f"{i}. {item.get('title', 'Untitled')}\n"
            f"   URL: {item.get('href') or item.get('url') or 'URL unavailable'}\n"
            f"   Snippet: {item.get('body', 'No snippet')}"
        )
    return "\n".join(lines)


# ----------------------------- Agents -----------------------------------

def create_researcher() -> Agent:
    return Agent(
        role="Senior Research Analyst",
        goal=(
            "Create a source-backed research brief. If context.txt exists, read it first "
            "with the local-file tool; treat it as user context, not verified evidence. "
            "Then search the web and separate contextual notes from corroborated facts."
        ),
        backstory=(
            "A careful analyst who cross-checks sources, avoids invented statistics, "
            "and clearly labels uncertainty."
        ),
        tools=[read_and_summarize_file, search_web_for_claims],
        llm=MODEL, verbose=True, allow_delegation=False,
    )


def create_writer() -> Agent:
    return Agent(
        role="Technology Article Writer",
        goal="Turn research into an engaging, structured, technically accurate article.",
        backstory="You write for a curious general audience and avoid unsupported claims.",
        llm=MODEL, verbose=True, allow_delegation=False,
    )


def create_fact_checker() -> Agent:
    return Agent(
        role="Fact-Checking Analyst",
        goal="Verify specific article claims against corroborating sources.",
        backstory=(
            "You are skeptical and evidence-driven. Prefer primary sources, official "
            "statistics, peer-reviewed papers, and independent corroboration. Never "
            "invent citations; search snippets alone are not proof."
        ),
        tools=[search_web_for_claims],
        llm=MODEL, verbose=True, allow_delegation=False,
    )


def create_reviewer() -> Agent:
    return Agent(
        role="Senior Content Editor",
        goal="Improve accuracy, clarity, completeness, structure, and readability.",
        backstory=(
            "You are a meticulous editor. Use the fact-check report to correct, qualify, "
            "or remove claims marked contradicted or unverified."
        ),
        llm=MODEL, verbose=True, allow_delegation=False,
    )


# ----------------------------- Tasks ------------------------------------

def create_research_task(agent: Agent, topic: str) -> Task:
    return Task(
        description=(
            f"Research: {topic}\n\n"
            f"If {CONTEXT_FILE.name} exists beside the script, use the local-file tool "
            f"to read and summarize it before web searching. If missing, continue with "
            "web research. Treat local context as background, not automatically as truth. "
            "Cover current state, developments, major organizations, challenges, and outlook. "
            "Provide source URLs and dates where available. Do not fabricate figures."
        ),
        expected_output=(
            "Structured research brief with findings, source URLs, dates where available, "
            "and separate sections for local context and independently corroborated facts."
        ),
        agent=agent,
    )


def create_writing_task(agent: Agent, topic: str) -> Task:
    return Task(
        description=(
            f"Write an 800-1200 word article about {topic}, using the research brief. "
            "Include an informative title, introduction, 3-5 sections, and conclusion. "
            "Use concrete examples, distinguish predictions from facts, and place source "
            "URLs or citation markers near factual claims."
        ),
        expected_output="Polished article draft with source URLs near factual claims.",
        agent=agent,
    )


def create_fact_check_task(agent: Agent, topic: str, writing_task: Task) -> Task:
    return Task(
        description=(
            f"Fact-check the article about {topic} from the preceding writing task. "
            "Extract 5-12 specific claims, prioritizing statistics, dates, causal claims, "
            "named organizations, and changing facts. For each, report claim text, status "
            "(VERIFIED, CONTRADICTED, or UNVERIFIED), evidence, source URL(s), and suggested "
            "correction/qualification. Prefer two independent sources. If evidence is weak, "
            "use UNVERIFIED. Never invent URLs or claim a search snippet proves a claim."
        ),
        expected_output=(
            "Claim-by-claim report with status, reasoning, source URLs, recommended action, "
            "and totals for verified, contradicted, and unverified claims."
        ),
        agent=agent,
        context=[writing_task],
    )


def create_review_task(agent: Agent, topic: str, writing_task: Task, fact_task: Task) -> Task:
    return Task(
        description=(
            f"Review the article about {topic} using both the article and fact-check report. "
            "Assess accuracy, clarity, completeness, structure, and engagement. Correct, "
            "qualify, or remove contradicted/unverified claims. Give a 1-10 score, strengths, "
            "specific issues fixed, and the final corrected article. Do not invent sources."
        ),
        expected_output="Editorial score, strengths, corrections, and final corrected article.",
        agent=agent,
        context=[writing_task, fact_task],
    )


def make_tasks(topic: str):
    researcher, writer, checker, reviewer = (
        create_researcher(), create_writer(), create_fact_checker(), create_reviewer()
    )
    research = create_research_task(researcher, topic)
    writing = create_writing_task(writer, topic)
    fact_check = create_fact_check_task(checker, topic, writing)
    review = create_review_task(reviewer, topic, writing, fact_check)
    return [researcher, writer, checker, reviewer], [research, writing, fact_check, review]


# ----------------------------- Crew modes -------------------------------

def build_sequential_crew(topic: str) -> Crew:
    agents, tasks = make_tasks(topic)
    return Crew(agents=agents, tasks=tasks, process=Process.sequential, verbose=True)


def build_hierarchical_crew(topic: str) -> Crew:
    agents, tasks = make_tasks(topic)
    return Crew(
        agents=agents,
        tasks=tasks,
        process=Process.hierarchical,
        manager_llm=MODEL,
        verbose=True,
    )


def as_text(result: Any) -> str:
    return getattr(result, "raw", str(result))


def run_and_measure(label: str, crew: Crew):
    print("\n" + "=" * 72 + "\n" + label + "\n" + "=" * 72)
    start = time.perf_counter()
    result = crew.kickoff()
    elapsed = time.perf_counter() - start
    output = as_text(result)
    print(f"\nCompleted in {elapsed:.2f}s. Output preview:\n")
    print(output[:4000] + ("\n[Preview truncated]" if len(output) > 4000 else ""))
    return output, elapsed


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print('Set your API key first:  $env:OPENAI_API_KEY = "your-api-key"')
        raise SystemExit(1)

    topic = input("Topic (Enter for 'AI agents in healthcare'): ").strip()
    topic = topic or "AI agents in healthcare"
    if CONTEXT_FILE.exists():
        print(f"[Context] Found: {CONTEXT_FILE}")
    else:
        print(f"[Context] {CONTEXT_FILE.name} not found; researcher should continue with web search.")

    seq_output, seq_time = run_and_measure(
        "SEQUENTIAL: researcher -> writer -> fact-checker -> reviewer",
        build_sequential_crew(topic),
    )
    hier_output, hier_time = run_and_measure(
        "HIERARCHICAL: manager LLM coordinates the agents",
        build_hierarchical_crew(topic),
    )

    observations = f"""
RUN COMPARISON
Topic: {topic}
Sequential elapsed time: {seq_time:.2f} seconds
Hierarchical elapsed time: {hier_time:.2f} seconds
Sequential output characters: {len(seq_output)}
Hierarchical output characters: {len(hier_output)}

Observations to use when interpreting this run:
- Sequential mode follows the fixed task list in order. Fact-checking is explicitly
  placed after writing and before final review.
- Hierarchical mode gives task coordination to a manager LLM. It can delegate work,
  ask agents for follow-up work, and coordinate revisions based on task results.
- Do not assume the manager skipped a step or requested extra revisions without
  checking the verbose execution logs. Record the actual behavior seen in this run.
- Runtime and output length are not quality scores. Compare source quality, claim
  accuracy, completeness, clarity, and whether unsupported claims were corrected.
- LLM output is nondeterministic. Repeat runs for a fairer comparison. Hierarchical
  mode may spend more tokens/time on coordination, but is not guaranteed to be slower
  or produce better output.
"""
    print(observations)
    report = BASE_DIR / "crew_comparison.txt"
    report.write_text(
        observations + "\n\nSEQUENTIAL OUTPUT\n\n" + seq_output
        + "\n\nHIERARCHICAL OUTPUT\n\n" + hier_output,
        encoding="utf-8",
    )
    print(f"Saved comparison and both outputs to: {report}")


if __name__ == "__main__":
    main()
