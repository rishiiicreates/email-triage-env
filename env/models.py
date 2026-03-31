"""Pydantic models for the Email Triage Agent Environment.

Defines typed Observation, Action, and Reward models that form the
OpenEnv interface contract between the environment and agents.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal


# ─── Core Email Model ────────────────────────────────────────────────


class Email(BaseModel):
    """A single email message in the inbox."""

    id: str = Field(
        ...,
        description="Unique identifier for this email",
        examples=["email_001"],
    )
    sender_name: str = Field(
        ...,
        description="Display name of the sender",
        examples=["Alice Johnson"],
    )
    sender_email: str = Field(
        ...,
        description="Email address of the sender (may be spoofed)",
        examples=["alice@company.com"],
    )
    subject: str = Field(
        ...,
        description="Email subject line",
        examples=["Q4 Budget Review"],
    )
    body: str = Field(
        ...,
        description="Full body text of the email (3-8 sentences)",
        examples=["Hi team, please review the attached budget..."],
    )
    department: str = Field(
        ...,
        description="Target department: Engineering, Sales, or HR",
        examples=["Engineering"],
    )
    timestamp: str = Field(
        ...,
        description="ISO 8601 timestamp of when the email was received",
        examples=["2025-01-15T09:05:00Z"],
    )


class EmailMeta(BaseModel):
    """Ground-truth metadata for an email, used internally by graders."""

    email_id: str = Field(..., description="ID linking to the Email object")
    urgency: str = Field(
        ..., description="Ground-truth urgency: urgent / normal / spam"
    )
    department: str = Field(
        ..., description="Correct department to route to"
    )
    has_pii: bool = Field(
        default=False, description="Whether the email body contains PII"
    )
    gold_reply: str = Field(
        default="",
        description="Reference reply text for similarity grading",
    )


# ─── Observation ─────────────────────────────────────────────────────


class EmailObservation(BaseModel):
    """What the agent sees at each step."""

    inbox: List[Email] = Field(
        ...,
        description="Full list of emails in the inbox for this episode",
    )
    current_email: Optional[Email] = Field(
        None,
        description="The email currently being processed (next in queue)",
    )
    step_count: int = Field(
        0,
        description="Current step number in the episode",
        examples=[0],
    )
    task_id: str = Field(
        ...,
        description="Active task identifier",
        examples=["classify_basic"],
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra metadata: remaining count, processed IDs, task info",
    )


# ─── Action ──────────────────────────────────────────────────────────


class EmailAction(BaseModel):
    """An action the agent takes on the current email."""

    action_type: Literal["classify", "reply", "route", "archive", "flag"] = Field(
        ...,
        description='Action type: "classify", "reply", "route", "archive", or "flag"',
        examples=["classify"],
    )
    label: Optional[str] = Field(
        None,
        description='Priority/category label: "urgent", "normal", or "spam"',
        examples=["urgent"],
    )
    reply_text: Optional[str] = Field(
        None,
        description="Draft reply content (required when replying to urgent emails)",
        examples=["Thank you for reporting this. I'm investigating now."],
    )
    route_to: Optional[str] = Field(
        None,
        description='Department/person to route to: "Engineering", "Sales", or "HR"',
        examples=["Engineering"],
    )
    reasoning: Optional[str] = Field(
        None,
        description="Agent's reasoning for this action (logged, not graded)",
        examples=["Email mentions production outage, routing to Engineering"],
    )


# ─── Reward ──────────────────────────────────────────────────────────


class EmailReward(BaseModel):
    """Reward signal returned after each step."""

    score: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Step reward score (0.0 to 1.0, with penalties possible)",
        examples=[0.3],
    )
    partial_credits: Dict[str, float] = Field(
        default_factory=dict,
        description="Breakdown of reward by sub-criterion",
        examples=[{"label_correct": 0.1, "priority_bonus": 0.05}],
    )
    explanation: str = Field(
        default="",
        description="Human-readable explanation of the reward",
        examples=["Correct classification of urgent email (+0.1)"],
    )


# ─── Response Wrappers ───────────────────────────────────────────────


class ResetResponse(BaseModel):
    """Response from POST /reset."""

    observation: EmailObservation


class StepResponse(BaseModel):
    """Response from POST /step."""

    observation: EmailObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class StateResponse(BaseModel):
    """Response from GET /state."""

    state: Dict[str, Any]
