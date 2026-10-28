"""
Customer Support Agent using LangGraph
=======================================
A complete customer support agent with stateful graph-based workflow.

States:
  1. greeting - Welcome the customer
  2. understand_issue - Classify the issue
  3. search_knowledge_base - Look up solutions
  4. provide_solution - Present a solution to the customer
  5. confirm_resolution - Verify the issue is resolved

Conditional transitions handle retries and escalation.

CUSTOMER SUPPORT AGENT — LANGGRAPH WORKFLOW

                         ┌──────────────┐
                         │    START     │
                         └──────┬───────┘
                                │
                                ▼
                         ┌──────────────┐
                         │   greeting   │
                         └──────┬───────┘
                                │
                                ▼
                      ┌────────────────────┐
                      │  understand_issue  │
                      └─────────┬──────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │  search_knowledge_base │◄─────────┐
                    └────────────┬───────────┘          │
                                 │                      │
                                 ▼                      │
                      ┌────────────────────┐            │
                      │  provide_solution  │            │
                      └─────────┬──────────┘            │
                                │                       │
                       Check is_resolved                │
                                │                       │
                ┌───────────────┼────────────────┐      │
                │               │                │      │
                ▼               ▼                ▼      │
          ┌───────────┐   ┌───────────┐   ┌───────────┐ │
          │  RESOLVED │   │   RETRY   │   │ ESCALATE  │ │
          └─────┬─────┘   └─────┬─────┘   └─────┬─────┘ │
                │               │               │       │
                ▼               └───────────────┘       │
       ┌──────────────────┐       Search again          │
       │ confirm_resolution│                            │
       └────────┬─────────┘                             │
                │                                       │
                ▼                                       │
             ┌─────┐                                    │
             │ END │                                    │
             └─────┘                                    │

                                                ESCALATION PATH:
                                                    provide_solution
                                                        │
                                                        ▼
                                                    ┌───────────────────┐
                                                    │  escalation_node  │
                                                    └─────────┬─────────┘
                                                              │
                                                              ├── Log unresolved issue details
                                                              ├── Simulate email to human agent
                                                              ├── Generate and display ticket number
                                                              │
                                                              ▼
                                                        ┌─────┐
                                                        │ END │
                                                        └─────┘

ROUTING RULES:
1. If is_resolved == True  → confirm_resolution
2. If unresolved and attempts < 3 → search_knowledge_base
3. If unresolved and attempts >= 3 → escalation_node

"""

from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
import operator
import random
from datetime import datetime
import uuid

# ---------------------------------------------------------------------------
# State Schema
# ---------------------------------------------------------------------------

class SupportState(TypedDict):
    messages: Annotated[Sequence[str], operator.add]
    current_step: str
    issue_category: str
    solution_found: bool
    resolution_attempts: int
    is_resolved: bool
    customer_input: str

    # New fields for escalation
    ticket_number: str
    escalation_logged: bool
    # These fields let the graph store the ticket number and record whether the escalation was logged.


# ---------------------------------------------------------------------------
# Simulated Knowledge Base
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE = {
    "billing": [
        "Check your billing statement for any unauthorized charges.",
        "You can request a refund through Account > Billing > Refund Request.",
        "Your subscription auto-renews on the 1st of each month.",
    ],
    "technical": [
        "Try clearing your browser cache and cookies.",
        "Restart the application and check for updates.",
        "Check your internet connection and try again.",
    ],
    "account": [
        "You can reset your password from the login page.",
        "Account verification emails may take up to 15 minutes.",
        "Contact support if your account is locked for more than 24 hours.",
    ],
}


# ---------------------------------------------------------------------------
# Node Functions
# ---------------------------------------------------------------------------

def greeting_node(state: SupportState) -> dict:
    """Welcome the customer and acknowledge their request."""
    customer_msg = state.get("customer_input", "")
    greeting = (
        f"[Greeting] Hello! Thank you for contacting customer support.\n"
        f"[Greeting] You mentioned: '{customer_msg}'\n"
        f"[Greeting] Let me help you with that right away."
    )
    return {
        "messages": [greeting],
        "current_step": "greeting"
    }


def understand_issue_node(state: SupportState) -> dict:
    """Classify the customer's issue into a category."""
    customer_msg = state.get("customer_input", "").lower()

    # Simple keyword-based classification
    if any(word in customer_msg for word in ["bill", "charge", "payment", "refund", "subscription"]):
        category = "billing"
    elif any(word in customer_msg for word in ["error", "bug", "crash", "not working", "broken", "slow"]):
        category = "technical"
    elif any(word in customer_msg for word in ["password", "login", "account", "sign", "locked", "email"]):
        category = "account"
    else:
        category = "general"

    message = (
        f"[Understand] I've classified your issue as: {category.upper()}\n"
        f"[Understand] Let me search our knowledge base for a solution."
    )
    return {
        "messages": [message],
        "current_step": "understand_issue",
        "issue_category": category
    }


def search_knowledge_base_node(state: SupportState) -> dict:
    """Look up solutions in the knowledge base."""
    category = state.get("issue_category", "general")
    solutions = KNOWLEDGE_BASE.get(category, ["No specific solutions found."])

    # Pick a solution (simulate search)
    selected_solution = random.choice(solutions)
    found = category in KNOWLEDGE_BASE

    message = (
        f"[Search KB] Searching knowledge base for: '{category}'\n"
        f"[Search KB] Found solution: {selected_solution}"
    )
    return {
        "messages": [message],
        "current_step": "search_knowledge_base",
        "solution_found": found
    }


def provide_solution_node(state: SupportState) -> dict:
    """Present the solution to the customer."""
    last_messages = state["messages"]
    solution_text = last_messages[-1] if last_messages else "No solution available."

    # Simulate resolution outcome
    attempts = state.get("resolution_attempts", 0)
    resolved = random.random() > 0.3  # 70% chance of resolution

    message = (
        f"[Solution] Based on my research, here is my recommendation:\n"
        f"[Solution] {solution_text.split('Found solution: ')[-1] if 'Found solution:' in solution_text else solution_text}\n"
        f"[Solution] Attempt #{attempts + 1} to resolve your issue."
    )
    return {
        "messages": [message],
        "current_step": "provide_solution",
        "resolution_attempts": attempts + 1,
        "is_resolved": resolved
    }


def confirm_resolution_node(state: SupportState) -> dict:
    """Confirm with the customer that the issue is resolved."""
    message = (
        f"[Confirm] Great! I'm glad we could resolve your issue.\n"
        f"[Confirm] Is there anything else I can help you with?\n"
        f"[Confirm] Thank you for contacting customer support!"
    )
    return {
        "messages": [message],
        "current_step": "confirm_resolution"
    }

def escalation_node(state: SupportState) -> dict:
    """Log the unresolved issue and simulate notifying a human agent."""

    # Generate a unique ticket number
    ticket_number = f"TKT-{uuid.uuid4().hex[:8].upper()}"

    # Collect unresolved issue details
    issue_details = {
        "ticket_number": ticket_number,
        "customer_issue": state.get("customer_input", "Not provided"),
        "issue_category": state.get("issue_category", "general"),
        "resolution_attempts": state.get("resolution_attempts", 0),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    # 1. Log the unresolved issue
    print("\n" + "=" * 60)
    print("ESCALATION LOG")
    print("=" * 60)

    for key, value in issue_details.items():
        print(f"{key}: {value}")

    # 2. Simulate sending an email to a human agent
    print("\n[Escalation] Preparing email for the human support team...")
    print("[Escalation] Simulated email sent successfully.")
    print("  To: human-support@example.com")
    print(f"  Subject: Unresolved customer issue - {ticket_number}")
    print(f"  Issue: {issue_details['customer_issue']}")

    # 3. Provide the customer with a ticket number
    customer_message = (
        f"[Escalation] I couldn't resolve your issue automatically. "
        f"I've referred it to our human support team. "
        f"Your ticket number is {ticket_number}. "
        f"Please keep this number for future reference."
    )

    print(f"\n[Customer Notification] {customer_message}")
    print("=" * 60)

    return {
        "messages": [customer_message],
        "current_step": "escalation",
        "ticket_number": ticket_number,
        "escalation_logged": True,
    }

# ---------------------------------------------------------------------------
# Conditional Edge Functions
# ---------------------------------------------------------------------------

def route_after_solution(state: SupportState) -> str:
    """Route after providing a solution: resolved, retry, or escalate."""
    if state.get("is_resolved", False):
        return "resolved"
    elif state.get("resolution_attempts", 0) >= 3:
        return "escalate"
    else:
        return "retry"


# ---------------------------------------------------------------------------
# Build the Graph
# ---------------------------------------------------------------------------

def build_support_agent() -> StateGraph:
    """Construct and compile the customer support agent graph."""
    workflow = StateGraph(SupportState)

    # Add nodes
    workflow.add_node("greeting", greeting_node)
    workflow.add_node("understand_issue", understand_issue_node)
    workflow.add_node("search_knowledge_base", search_knowledge_base_node)
    workflow.add_node("provide_solution", provide_solution_node)
    workflow.add_node("confirm_resolution", confirm_resolution_node)
    workflow.add_node("escalation", escalation_node)

    # Set entry point
    workflow.set_entry_point("greeting")

    # Normal edges
    workflow.add_edge("greeting", "understand_issue")
    workflow.add_edge("understand_issue", "search_knowledge_base")
    workflow.add_edge("search_knowledge_base", "provide_solution")

    # Conditional edges after providing solution
    workflow.add_conditional_edges(
        "provide_solution",
        route_after_solution,
        {
            "resolved": "confirm_resolution",
            "retry": "search_knowledge_base",
            "escalate": "escalation"
        }
    )

    # End after confirmation
    workflow.add_edge("escalation", END)
    workflow.add_edge("confirm_resolution", END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# Escalation handler (added to output)
# ---------------------------------------------------------------------------

def handle_escalation(state: SupportState) -> None:
    """Handle cases where the issue could not be resolved."""
    print("\n" + "=" * 60)
    print("ESCALATION: Issue could not be resolved after 3 attempts.")
    print(f"  Issue Category: {state.get('issue_category', 'unknown')}")
    print(f"  Attempts Made: {state.get('resolution_attempts', 0)}")
    print("  Action: Transferring to a human agent.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main():
    """Run the customer support agent."""
    app = build_support_agent()

    # Test cases representing different customer issues
    test_cases = [
        "I was charged twice on my last bill",
        "The application keeps crashing when I open it",
        "I can't reset my password and my account is locked",
        "Something is not working properly",
    ]

    for i, customer_input in enumerate(test_cases, 1):
        print(f"\n{'#' * 60}")
        print(f"# TEST CASE {i}: '{customer_input}'")
        print(f"{'#' * 60}\n")

        initial_state = {
            "messages": [],
            "current_step": "start",
            "issue_category": "",
            "solution_found": False,
            "resolution_attempts": 0,
            "is_resolved": False,
            "customer_input": customer_input,
            # New escalation fields
            "ticket_number": "",
            "escalation_logged": False,
        }

        # Execute the graph
        result = app.invoke(initial_state)

        # Check for escalation
        if result.get("resolution_attempts", 0) >= 3 and not result.get("is_resolved", False):
            handle_escalation(result)

        # Print the conversation
        print("\n--- Conversation Flow ---")
        for j, msg in enumerate(result["messages"], 1):
            print(f"\n  Step {j}: {msg}")

        print(f"\n--- Final State ---")
        print(f"  Resolved: {result.get('is_resolved', False)}")
        print(f"  Attempts: {result.get('resolution_attempts', 0)}")
        print(f"  Category: {result.get('issue_category', 'N/A')}")
        print(f"  Current Step: {result.get('current_step', 'N/A')}")
        print(f"  Ticket Number: {result.get('ticket_number', 'N/A')}")
        print(f"  Escalation Logged: {result.get('escalation_logged', False)}")

if __name__ == "__main__":
    main()
