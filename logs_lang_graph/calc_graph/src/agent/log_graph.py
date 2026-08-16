"""Route application logs to the right operational workflow.

Run this graph with ``{"log": "..."}``. The classifier is the only LLM
node; the operational actions remain ordinary, predictable Python code.
"""

from typing import Literal

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class LogClassification(BaseModel):
    """The structured decision made by the LLM."""

    severity: Literal["critical", "warning", "info"] = Field(
        description="Operational urgency of the event."
    )
    category: Literal["security", "application", "infrastructure"]
    summary: str = Field(description="Short, factual explanation of the event.")


class LogState(TypedDict, total=False):
    """State passed between the classifier and deterministic handlers."""

    log: str
    classification: LogClassification
    outcome: str


model = init_chat_model("gpt-5.5", temperature=0)
classifier = model.with_structured_output(LogClassification)


def _classification_from_state(state: LogState) -> LogClassification | None:
    """Normalize restored JSON state and fresh in-memory state alike."""
    value = state.get("classification")
    if value is None:
        return None
    return LogClassification.model_validate(value)


def classify_log(state: LogState) -> dict[str, LogClassification]:
    """Use the LLM to turn free-form log text into a constrained decision."""
    classification = classifier.invoke(
        [
            SystemMessage(
                content=(
                    "Classify this production log. Use critical only for events that "
                    "need immediate human action, warning for actionable degradation, "
                    "and info for normal or low-priority events."
                )
            ),
            HumanMessage(content=state["log"]),
        ]
    )
    return {"classification": classification}


def route_log(
    state: LogState,
) -> Literal["classify_log", "request_approval", "create_ticket", "archive"]:
    """Route using the LLM result without another model call."""
    classification = _classification_from_state(state)
    if classification is None:
        # A state edit or resumed Studio run can enter this branch without the
        # previous node's output. Re-run classification rather than failing.
        return "classify_log"

    severity = classification.severity
    if severity == "critical":
        return "request_approval"
    if severity == "warning":
        return "create_ticket"
    return "archive"


def request_approval(
    state: LogState,
) -> Command[Literal["page_on_call", "reject_alert"]]:
    """Pause a critical alert until an operator approves or rejects it."""
    decision = _classification_from_state(state)
    if decision is None:
        return Command(goto="reject_alert")
    approved = interrupt(
        {
            "question": "Page the on-call engineer for this critical log?",
            "log": state["log"],
            "classification": decision.model_dump(),
        }
    )
    return Command(goto="page_on_call" if approved is True else "reject_alert")


def page_on_call(state: LogState) -> dict[str, str]:
    """Placeholder for PagerDuty, Slack, or another urgent notification."""
    decision = _classification_from_state(state)
    if decision is None:
        return {"outcome": "No page sent because the classification was missing."}
    return {"outcome": f"Paged on-call for {decision.category}: {decision.summary}"}


def reject_alert(state: LogState) -> dict[str, str]:
    """Record a rejected alert without performing the external action."""
    return {"outcome": "Critical alert rejected; no page was sent."}


def create_ticket(state: LogState) -> dict[str, str]:
    """Placeholder for creating a Jira, Linear, or GitHub issue."""
    decision = _classification_from_state(state)
    if decision is None:
        return {"outcome": "No ticket created because the classification was missing."}
    return {"outcome": f"Created ticket for {decision.category}: {decision.summary}"}


def archive(state: LogState) -> dict[str, str]:
    """Placeholder for storing a low-priority event in log analytics."""
    return {"outcome": "Archived informational log."}


builder = StateGraph(LogState)
builder.add_node("classify_log", classify_log)
builder.add_node("request_approval", request_approval)
builder.add_node("page_on_call", page_on_call)
builder.add_node("reject_alert", reject_alert)
builder.add_node("create_ticket", create_ticket)
builder.add_node("archive", archive)
builder.add_edge(START, "classify_log")
builder.add_conditional_edges(
    "classify_log",
    route_log,
    ["classify_log", "request_approval", "create_ticket", "archive"],
)
builder.add_edge("page_on_call", END)
builder.add_edge("reject_alert", END)
builder.add_edge("create_ticket", END)
builder.add_edge("archive", END)

log_graph = builder.compile()


def process_logs(logs: list[str]) -> list[LogState]:
    """Process independent logs concurrently through the same orchestration graph."""
    return log_graph.batch([{"log": log} for log in logs], config={"max_concurrency": 10})
