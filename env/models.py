"""Pydantic models for the Email Triage Environment."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class Email(BaseModel):
    """A single email in the inbox."""
    id: str
    sender: str
    subject: str
    body: str
    timestamp: str


class Observation(BaseModel):
    """What the agent sees at each step."""
    emails: List[Email]
    step_number: int
    task_id: str
    remaining_emails: int = 0


class Action(BaseModel):
    """An action the agent takes on one email."""
    email_id: str
    label: str = Field(
        ...,
        description='Classification label: "urgent", "normal", "spam", or "reply-needed"'
    )
    priority: int = Field(
        ...,
        ge=1,
        le=5,
        description="Priority ranking from 1 (highest) to 5 (lowest)"
    )
    draft_reply: Optional[str] = Field(
        default=None,
        description="Optional draft reply text (required for task_hard)"
    )


class Reward(BaseModel):
    """Reward signal returned after each step."""
    value: float = Field(..., ge=-1.0, le=1.0)
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class StepResult(BaseModel):
    """Full result of an environment step."""
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class EpisodeAction(BaseModel):
    """A single action within an episode log, used for grading."""
    email_id: str
    label: str
    priority: int
    draft_reply: Optional[str] = None


class GraderRequest(BaseModel):
    """Request body for the grader endpoint."""
    task_id: str
    episode_log: List[EpisodeAction]


class GraderResponse(BaseModel):
    """Response from the grader endpoint."""
    score: float
    details: Dict[str, Any] = Field(default_factory=dict)
