"""
Human-in-the-Loop Email Response Agent
======================================

Workflow:
    Receive -> Analyze -> Draft -> Support Review
                                  | approve
                                  v
                      VIP complaint? -- yes --> Manager Review
                            | no                    | approve
                            +-----------------------+
                                                    v
                                                Send (simulated)
    Any review: reject -> Stop
                edit   -> Update draft -> Support Review again

VIP rule:
    A complaint from a sender whose domain is in VIP_DOMAINS needs both
    support-agent approval and manager approval.

Run:
    python -m pip install -U langgraph
    python hitl_email_agent.py

The "send" node only prints a simulated send; it does not send real email.
"""

from __future__ import annotations

import json
import operator
import time
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Callable, TypedDict

from langgraph.graph import END, StateGraph


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VIP_DOMAINS = {"enterprise-client.com"}
AUDIT_LOG_PATH = Path("audit_log.json")

SUPPORT_REVIEWER_ID = "support_agent_001"
MANAGER_REVIEWER_ID = "manager_001"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class EmailAgentState(TypedDict):
    messages: Annotated[list[str], operator.add]
    current_step: str
    incoming_email: str
    sender: str
    subject: str
    email_category: str
    priority: str
    draft_response: str
    approval_status: str
    manager_approval_status: str
    vip_approval: bool
    human_feedback: str
    edited_response: str
    send_count: int
    max_revisions: int
    rejected_by: str


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------

def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    """Convert state values to JSON-compatible values."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class AuditLogger:
    """Collect workflow transitions and review decisions, then write JSON."""

    def __init__(self, output_path: Path | str = AUDIT_LOG_PATH) -> None:
        self.output_path = Path(output_path)
        self.events: list[dict[str, Any]] = []

    def record_transition(
        self,
        node_name: str,
        before_state: dict[str, Any],
        node_update: dict[str, Any],
        after_state: dict[str, Any],
    ) -> None:
        self.events.append(
            {
                "event_type": "state_transition",
                "timestamp": utc_now(),
                "node": node_name,
                "state_before": json_safe(before_state),
                "node_update": json_safe(node_update),
                "state_after": json_safe(after_state),
            }
        )

    def record_human_decision(
        self,
        *,
        reviewer_id: str,
        review_level: str,
        decision: str,
        draft_before: str,
        draft_after: str,
        duration_seconds: float,
        feedback: str = "",
    ) -> None:
        self.events.append(
            {
                "event_type": "human_decision",
                "timestamp": utc_now(),
                "who_reviewed": reviewer_id,
                "review_level": review_level,
                "decision": decision,
                "draft_before": draft_before,
                "draft_after": draft_after,
                "feedback": feedback,
                "duration_seconds": round(duration_seconds, 4),
            }
        )

    def write_json(self) -> None:
        """Write all accumulated audit events to disk."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "created_at": utc_now(),
            "event_count": len(self.events),
            "events": self.events,
        }
        self.output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\n[Audit] Wrote {len(self.events)} events to: {self.output_path.resolve()}")


# ---------------------------------------------------------------------------
# Email classification and drafting
# ---------------------------------------------------------------------------

EMAIL_CATEGORIES = {
    "inquiry": {
        "greeting": "Thank you for reaching out to us.",
        "closing": "Please don't hesitate to contact us if you have further questions.",
    },
    "complaint": {
        "greeting": "We sincerely apologize for the inconvenience you have experienced.",
        "closing": "We value your feedback and are committed to resolving this matter.",
    },
    "support": {
        "greeting": "Thank you for contacting our support team.",
        "closing": "We hope this resolves your issue. Please contact us if you need more help.",
    },
    "general": {
        "greeting": "Thank you for your message.",
        "closing": "Best regards, Support Team",
    },
}


def receive_email_node(state: EmailAgentState) -> dict[str, Any]:
    sender = state.get("sender", "unknown@example.com")
    subject = state.get("subject", "No Subject")
    message = (
        f"[Receive] New email received\n"
        f"  From: {sender}\n"
        f"  Subject: {subject}\n"
        f"  Body preview: {state.get('incoming_email', '')[:100]}"
    )
    print(message)
    return {"messages": [message], "current_step": "receive_email"}


def analyze_email_node(state: EmailAgentState) -> dict[str, Any]:
    email = state.get("incoming_email", "").lower()
    subject = state.get("subject", "").lower()

    if any(word in email or word in subject for word in
           ["complaint", "angry", "unhappy", "terrible", "disappointed", "refund", "poor service"]):
        category, priority = "complaint", "high"
    elif any(word in email or word in subject for word in
             ["help", "support", "bug", "error", "issue", "not working"]):
        category, priority = "support", "medium"
    elif any(word in email or word in subject for word in
             ["question", "inquiry", "information", "pricing", "price"]):
        category, priority = "inquiry", "low"
    else:
        category, priority = "general", "low"

    sender = state.get("sender", "")
    domain = sender.rsplit("@", 1)[-1].strip().lower() if "@" in sender else ""
    vip_approval = category == "complaint" and domain in VIP_DOMAINS

    message = (
        f"[Analyze] Category={category.upper()}, priority={priority.upper()}, "
        f"VIP dual approval required={vip_approval}"
    )
    print(message)
    return {
        "messages": [message],
        "current_step": "analyze_email",
        "email_category": category,
        "priority": priority,
        "vip_approval": vip_approval,
    }


def draft_response_node(state: EmailAgentState) -> dict[str, Any]:
    category = state.get("email_category", "general")
    sender_name = state.get("sender", "Customer").split("@")[0]
    templates = EMAIL_CATEGORIES.get(category, EMAIL_CATEGORIES["general"])

    # If a reviewer supplied a replacement, use it exactly as the new draft.
    edited = state.get("edited_response", "").strip()
    feedback = state.get("human_feedback", "").strip()

    if edited:
        draft = edited
        note = " (using human-edited draft)"
    else:
        feedback_text = f"\n\nWe have noted your feedback: {feedback}" if feedback else ""
        draft = (
            f"Dear {sender_name},\n\n"
            f"{templates['greeting']}\n\n"
            f"We have reviewed your message regarding '{state.get('subject', 'your request')}'. "
            f"Our team will investigate this matter and provide an update within 24-48 business hours."
            f"{feedback_text}\n\n"
            f"{templates['closing']}\n\n"
            f"Best regards,\nSupport Team"
        )
        note = ""

    message = f"[Draft] Draft prepared{note}:\n{'-' * 50}\n{draft}\n{'-' * 50}"
    print(message)
    return {
        "messages": [message],
        "current_step": "draft_response",
        "draft_response": draft,
        "approval_status": "pending",
        "manager_approval_status": "pending",
        # Clear the replacement after applying it, so it won't override a later revision.
        "edited_response": "",
    }


# ---------------------------------------------------------------------------
# Interactive review
# ---------------------------------------------------------------------------

def ask_for_decision(reviewer_id: str, review_level: str, draft: str) -> tuple[str, str]:
    """
    Prompt at the terminal.

    Accepted inputs:
      approve
      reject
      edit <replacement draft>
      edit   (then enter replacement draft lines; type END on a line by itself)
    """
    print("\n" + "=" * 64)
    print(f"{review_level.upper()} REVIEW REQUIRED  |  Reviewer: {reviewer_id}")
    print("=" * 64)
    print("\nCurrent draft:\n")
    print(draft)
    print("\nType: approve | reject | edit <replacement draft>")
    print("For a multi-line replacement, type 'edit' and finish with END.")

    while True:
        try:
            raw = input(f"\n[{reviewer_id}] Decision: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Review] No input received; treating this review as rejected.")
            return "rejected", "No terminal input was received."
        except Exception as exc:
            print(f"[Review] Input error ({exc}); treating this review as rejected.")
            return "rejected", f"Input error: {type(exc).__name__}"

        command = raw.lower()
        if command == "approve":
            return "approved", ""
        if command == "reject":
            return "rejected", ""
        if command == "edit" or command.startswith("edit "):
            replacement = raw[4:].strip() if len(raw) > 4 else ""
            if not replacement:
                print("Enter the full replacement draft. Type END on a line by itself to finish:")
                lines: list[str] = []
                while True:
                    try:
                        line = input()
                    except (EOFError, KeyboardInterrupt):
                        print("\n[Review] Edit input interrupted; keeping the current draft.")
                        return "invalid", "Edit input was interrupted."
                    if line.strip() == "END":
                        break
                    lines.append(line)
                replacement = "\n".join(lines).strip()

            if not replacement:
                print("The edited draft cannot be empty. Please choose approve, reject, or edit again.")
                continue
            return "edited", replacement

        print("Invalid input. Please type exactly 'approve', 'reject', or 'edit <replacement draft>'.")


def make_review_node(
    audit_logger: AuditLogger,
    reviewer_id: str,
    review_level: str,
    approval_field: str,
) -> Callable[[EmailAgentState], dict[str, Any]]:
    """Create a review node that records a decision and its duration."""

    def review_node(state: EmailAgentState) -> dict[str, Any]:
        draft_before = state.get("draft_response", "")
        started = time.perf_counter()
        decision, detail = ask_for_decision(reviewer_id, review_level, draft_before)
        duration = time.perf_counter() - started

        update: dict[str, Any] = {
            "messages": [f"[Review] {review_level} decision: {decision.upper()} ({reviewer_id})"],
            "current_step": f"{review_level.lower().replace(' ', '_')}_review",
            approval_field: decision,
        }

        if decision == "edited":
            update["edited_response"] = detail
            update["human_feedback"] = "Reviewer supplied a replacement draft."
            draft_after = detail
        elif decision == "rejected":
            update["rejected_by"] = reviewer_id
            update["human_feedback"] = detail
            draft_after = draft_before
        elif decision == "invalid":
            # Interrupted edit: ask the reviewer again rather than silently approving.
            update[approval_field] = "pending"
            update["human_feedback"] = detail
            draft_after = draft_before
        else:
            draft_after = draft_before

        audit_logger.record_human_decision(
            reviewer_id=reviewer_id,
            review_level=review_level,
            decision=decision,
            draft_before=draft_before,
            draft_after=draft_after,
            duration_seconds=duration,
            feedback=detail,
        )
        print(f"[Review] {review_level}: {decision.upper()}")
        return update

    return review_node


def process_support_decision_node(state: EmailAgentState) -> dict[str, Any]:
    status = state.get("approval_status", "pending")
    message = f"[Process] Support review status: {status.upper()}"
    print(message)
    return {"messages": [message], "current_step": "process_support_decision"}


def process_manager_decision_node(state: EmailAgentState) -> dict[str, Any]:
    status = state.get("manager_approval_status", "pending")
    message = f"[Process] Manager review status: {status.upper()}"
    print(message)
    return {"messages": [message], "current_step": "process_manager_decision"}


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------

def route_after_support_review(state: EmailAgentState) -> str:
    status = state.get("approval_status", "pending")
    if status == "rejected":
        return "stop"
    if status == "edited":
        return "redraft"
    if status == "approved":
        return "manager_review" if state.get("vip_approval", False) else "send"
    # Invalid/interrupted input: repeat the support review.
    return "review_again"


def route_after_manager_review(state: EmailAgentState) -> str:
    status = state.get("manager_approval_status", "pending")
    if status == "rejected":
        return "stop"
    if status == "edited":
        return "redraft"
    if status == "approved":
        return "send"
    return "review_again"


# ---------------------------------------------------------------------------
# Send and summary
# ---------------------------------------------------------------------------

def send_email_node(state: EmailAgentState) -> dict[str, Any]:
    draft = state.get("draft_response", "")
    sender = state.get("sender", "")
    message = (
        f"[Send - SIMULATED] Approved response would be sent to: {sender}\n"
        f"{'-' * 50}\n{draft}\n{'-' * 50}"
    )
    print(message)
    return {
        "messages": [message],
        "current_step": "send_email",
        "send_count": state.get("send_count", 0) + 1,
    }


def final_summary_node(state: EmailAgentState) -> dict[str, Any]:
    status = state.get("approval_status", "pending")
    manager_status = state.get("manager_approval_status", "not_required")
    if status == "rejected" or manager_status == "rejected":
        outcome = "rejected"
    elif state.get("send_count", 0) > 0:
        outcome = "approved_and_sent_simulation"
    else:
        outcome = "stopped_without_sending"

    summary = (
        f"\n{'=' * 56}\nWORKFLOW SUMMARY\n{'=' * 56}\n"
        f"From: {state.get('sender', 'N/A')}\n"
        f"Subject: {state.get('subject', 'N/A')}\n"
        f"Category: {state.get('email_category', 'N/A')}\n"
        f"Priority: {state.get('priority', 'N/A')}\n"
        f"VIP dual approval required: {state.get('vip_approval', False)}\n"
        f"Support approval: {status}\n"
        f"Manager approval: {manager_status}\n"
        f"Outcome: {outcome}\n"
        f"{'=' * 56}"
    )
    print(summary)
    return {"messages": [summary], "current_step": "complete"}


# ---------------------------------------------------------------------------
# Audit wrapper: records every node/state transition
# ---------------------------------------------------------------------------

def wrap_with_audit(
    node_name: str,
    node_fn: Callable[[EmailAgentState], dict[str, Any]],
    audit_logger: AuditLogger,
) -> Callable[[EmailAgentState], dict[str, Any]]:
    def wrapped(state: EmailAgentState) -> dict[str, Any]:
        before = deepcopy(dict(state))
        update = node_fn(state)
        after = deepcopy(before)

        # Mirror LangGraph's basic update behavior for the audit snapshot.
        for key, value in update.items():
            if key == "messages":
                after[key] = list(before.get(key, [])) + list(value)
            else:
                after[key] = value

        audit_logger.record_transition(node_name, before, update, after)
        return update

    return wrapped


def build_hitl_agent(audit_logger: AuditLogger):
    workflow = StateGraph(EmailAgentState)

    nodes: dict[str, Callable[[EmailAgentState], dict[str, Any]]] = {
        "receive_email": receive_email_node,
        "analyze_email": analyze_email_node,
        "draft_response": draft_response_node,
        "support_review": make_review_node(
            audit_logger, SUPPORT_REVIEWER_ID, "First-level support", "approval_status"
        ),
        "process_support_decision": process_support_decision_node,
        "manager_review": make_review_node(
            audit_logger, MANAGER_REVIEWER_ID, "Manager", "manager_approval_status"
        ),
        "process_manager_decision": process_manager_decision_node,
        "send_email": send_email_node,
        "final_summary": final_summary_node,
    }

    for node_name, node_fn in nodes.items():
        workflow.add_node(node_name, wrap_with_audit(node_name, node_fn, audit_logger))

    workflow.set_entry_point("receive_email")
    workflow.add_edge("receive_email", "analyze_email")
    workflow.add_edge("analyze_email", "draft_response")
    workflow.add_edge("draft_response", "support_review")
    workflow.add_edge("support_review", "process_support_decision")

    workflow.add_conditional_edges(
        "process_support_decision",
        route_after_support_review,
        {
            "stop": "final_summary",
            "redraft": "draft_response",
            "manager_review": "manager_review",
            "send": "send_email",
            "review_again": "support_review",
        },
    )

    workflow.add_edge("manager_review", "process_manager_decision")
    workflow.add_conditional_edges(
        "process_manager_decision",
        route_after_manager_review,
        {
            "stop": "final_summary",
            "redraft": "draft_response",
            "send": "send_email",
            "review_again": "manager_review",
        },
    )

    workflow.add_edge("send_email", "final_summary")
    workflow.add_edge("final_summary", END)
    return workflow.compile()


def make_initial_state(email: dict[str, str]) -> EmailAgentState:
    return {
        "messages": [],
        "current_step": "start",
        "incoming_email": email["incoming_email"],
        "sender": email["sender"],
        "subject": email["subject"],
        "email_category": "",
        "priority": "",
        "draft_response": "",
        "approval_status": "pending",
        "manager_approval_status": "pending",
        "vip_approval": False,
        "human_feedback": "",
        "edited_response": "",
        "send_count": 0,
        "max_revisions": 3,
        "rejected_by": "",
    }


def main() -> None:
    print("=" * 64)
    print("HUMAN-IN-THE-LOOP EMAIL RESPONSE AGENT")
    print("Interactive approvals + VIP manager review + JSON audit trail")
    print("=" * 64)
    print("Use the sample emails below, or edit them in main() to try other cases.")

    sample_emails = [
        {
            "incoming_email": (
                "I am unhappy with the delayed delivery. This is a terrible experience "
                "and I want the issue resolved immediately."
            ),
            "sender": "customer@example.com",
            "subject": "Complaint about delayed delivery",
        },
        {
            "incoming_email": (
                "I am disappointed with the service and would like an update on my complaint."
            ),
            "sender": "vip-contact@enterprise-client.com",
            "subject": "Complaint: service delay",
        },
    ]

    audit_logger = AuditLogger(AUDIT_LOG_PATH)
    app = build_hitl_agent(audit_logger)

    try:
        for index, email in enumerate(sample_emails, start=1):
            print(f"\n\n{'#' * 64}\nEMAIL {index}: {email['subject']}\n{'#' * 64}")
            result = app.invoke(make_initial_state(email))
            print(
                f"\n[Complete] Final step={result.get('current_step')}; "
                f"sent simulations={result.get('send_count', 0)}"
            )
    finally:
        # Ensure the audit trail is written even if a workflow raises an error.
        audit_logger.write_json()

    # Verify the JSON file can be reopened and summarized.
    saved = json.loads(AUDIT_LOG_PATH.read_text(encoding="utf-8"))
    decision_events = [event for event in saved["events"] if event["event_type"] == "human_decision"]
    print("\n[AUDIT VERIFICATION]")
    print(f"File exists: {AUDIT_LOG_PATH.resolve().exists()}")
    print(f"Total audit events: {saved['event_count']}")
    print(f"Recorded human decisions: {len(decision_events)}")
    for event in decision_events:
        print(
            f"  {event['review_level']} / {event['who_reviewed']}: "
            f"{event['decision']} in {event['duration_seconds']} seconds"
        )


if __name__ == "__main__":
    main()
