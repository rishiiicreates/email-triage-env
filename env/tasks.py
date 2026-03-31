"""Task registry for the Email Triage Environment.

Defines task configurations for easy/medium/hard difficulty levels
and maps task IDs to their email generators and graders.
"""

from __future__ import annotations

from typing import Dict, Any, List, Tuple

from .models import Email, EmailMeta
from .data import (
    generate_classify_basic,
    generate_triage_and_reply,
    generate_full_triage_pipeline,
)
from .graders import (
    grade_classify_basic,
    grade_triage_and_reply,
    grade_full_triage_pipeline,
)


# ─── Task Definitions ────────────────────────────────────────────────


TASK_REGISTRY: Dict[str, Dict[str, Any]] = {
    "classify_basic": {
        "name": "Basic Email Classification",
        "difficulty": "easy",
        "max_steps": 10,
        "description": (
            "Inbox of 10 emails. Classify each as urgent / normal / spam. "
            "Graded by accuracy: correct_labels / total_emails."
        ),
        "generator": generate_classify_basic,
        "grader": grade_classify_basic,
    },
    "triage_and_reply": {
        "name": "Triage and Reply",
        "difficulty": "medium",
        "max_steps": 15,
        "description": (
            "Inbox of 5 emails. Classify AND write a short reply to the 2 "
            "marked urgent. Graded on urgency detection (0.3), reply "
            "relevance via cosine similarity (0.4), and tone (0.3)."
        ),
        "generator": generate_triage_and_reply,
        "grader": grade_triage_and_reply,
    },
    "full_triage_pipeline": {
        "name": "Full Triage Pipeline",
        "difficulty": "hard",
        "max_steps": 25,
        "description": (
            "Inbox of 15 emails across 3 departments. Classify all, reply "
            "to urgent, route to correct department, flag PII-containing "
            "emails. Graded on classification (0.25), routing (0.30), "
            "reply quality (0.25), and PII flagging recall (0.20)."
        ),
        "generator": generate_full_triage_pipeline,
        "grader": grade_full_triage_pipeline,
    },
}


def get_task_config(task_id: str) -> Dict[str, Any]:
    """Get configuration for a task by ID. Raises ValueError if unknown."""
    if task_id not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task_id '{task_id}'. "
            f"Valid tasks: {list(TASK_REGISTRY.keys())}"
        )
    return TASK_REGISTRY[task_id]


def get_task_ids() -> List[str]:
    """Return all available task IDs."""
    return list(TASK_REGISTRY.keys())


def generate_task_data(task_id: str) -> Tuple[List[Email], List[EmailMeta]]:
    """Generate fresh email data for the given task."""
    config = get_task_config(task_id)
    return config["generator"]()


def grade_episode(
    task_id: str,
    actions: List[Dict[str, Any]],
    metas: List[Dict[str, Any]],
) -> Tuple[float, Dict[str, float], str]:
    """Grade a full episode. Returns (score, partial_credits, explanation)."""
    config = get_task_config(task_id)
    return config["grader"](actions, metas)
