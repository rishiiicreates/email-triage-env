"""Task definitions and deterministic graders for the Email Triage Environment."""

import json
import os
from typing import Dict, List, Any, Tuple
from scipy.stats import kendalltau

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _load_gold_labels() -> Dict[str, Any]:
    """Load gold labels from JSON file."""
    with open(os.path.join(DATA_DIR, "gold_labels.json"), "r") as f:
        return json.load(f)


def _load_emails() -> List[Dict[str, Any]]:
    """Load emails from JSON file."""
    with open(os.path.join(DATA_DIR, "emails.json"), "r") as f:
        return json.load(f)


# ─── Task Definitions ────────────────────────────────────────────────

TASK_DEFINITIONS = {
    "task_easy": {
        "name": "Classify urgency",
        "difficulty": "easy",
        "description": (
            "Classify each email in the inbox as 'urgent', 'normal', 'spam', "
            "or 'reply-needed'. Scored by exact-match accuracy against gold labels."
        ),
        "email_count": 15,
        "max_steps": 15,
    },
    "task_medium": {
        "name": "Label and prioritize inbox",
        "difficulty": "medium",
        "description": (
            "Classify each email AND assign a priority ranking (1=highest to 5=lowest). "
            "Scored by a weighted combination of label accuracy (50%) and "
            "normalized Kendall tau rank correlation (50%)."
        ),
        "email_count": 15,
        "max_steps": 15,
    },
    "task_hard": {
        "name": "Triage and draft replies",
        "difficulty": "hard",
        "description": (
            "Classify, prioritize, AND draft replies for emails labeled as 'urgent' "
            "or 'reply-needed'. Scored by a weighted combination of label accuracy (30%), "
            "rank correlation (20%), and reply quality via keyword coverage (50%)."
        ),
        "email_count": 15,
        "max_steps": 15,
    },
}


def get_task_emails(task_id: str) -> List[Dict[str, Any]]:
    """Get the emails for a given task."""
    emails = _load_emails()
    task_def = TASK_DEFINITIONS[task_id]
    return emails[: task_def["email_count"]]


# ─── Grader Functions ────────────────────────────────────────────────

def _label_accuracy(episode_log: List[Dict[str, Any]], gold: Dict[str, Any]) -> float:
    """Compute exact-match label accuracy."""
    if not episode_log:
        return 0.0
    correct = 0
    total = 0
    for action in episode_log:
        email_id = action["email_id"]
        if email_id in gold:
            total += 1
            if action["label"].lower().strip() == gold[email_id]["label"]:
                correct += 1
    return correct / total if total > 0 else 0.0


def _rank_correlation(episode_log: List[Dict[str, Any]], gold: Dict[str, Any]) -> float:
    """Compute normalized Kendall tau rank correlation between predicted and gold priorities."""
    if len(episode_log) < 2:
        return 0.0

    predicted_priorities = []
    gold_priorities = []

    for action in episode_log:
        email_id = action["email_id"]
        if email_id in gold:
            predicted_priorities.append(action["priority"])
            gold_priorities.append(gold[email_id]["priority"])

    if len(predicted_priorities) < 2:
        return 0.0

    tau, _ = kendalltau(predicted_priorities, gold_priorities)

    # Handle NaN (e.g. when all values are identical)
    if tau != tau:  # NaN check
        tau = 0.0

    # Normalize from [-1, 1] to [0, 1]
    return (tau + 1.0) / 2.0


def _reply_quality(episode_log: List[Dict[str, Any]], gold: Dict[str, Any]) -> float:
    """
    Compute reply quality using keyword coverage against gold reply templates.
    Only scores emails that require replies (urgent or reply-needed).
    """
    if not episode_log:
        return 0.0

    total_score = 0.0
    reply_count = 0

    for action in episode_log:
        email_id = action["email_id"]
        if email_id not in gold:
            continue

        gold_entry = gold[email_id]
        # Only grade replies for emails that need them
        if gold_entry["label"] not in ("urgent", "reply-needed"):
            continue
        if not gold_entry.get("reply_keywords"):
            continue

        reply_count += 1
        draft = action.get("draft_reply", "") or ""
        draft_lower = draft.lower()

        if not draft_lower.strip():
            # No reply provided for an email that needs one → 0
            continue

        # Keyword coverage
        keywords = gold_entry["reply_keywords"]
        hits = sum(1 for kw in keywords if kw.lower() in draft_lower)
        coverage = hits / len(keywords) if keywords else 0.0
        total_score += coverage

    return total_score / reply_count if reply_count > 0 else 0.0


def grade_episode(task_id: str, episode_log: List[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
    """
    Grade an episode for the given task. Returns (score, details).
    Score is always in [0.0, 1.0].
    """
    gold = _load_gold_labels()

    if task_id == "task_easy":
        accuracy = _label_accuracy(episode_log, gold)
        return accuracy, {"label_accuracy": round(accuracy, 4)}

    elif task_id == "task_medium":
        accuracy = _label_accuracy(episode_log, gold)
        rank_corr = _rank_correlation(episode_log, gold)
        score = 0.5 * accuracy + 0.5 * rank_corr
        return round(score, 4), {
            "label_accuracy": round(accuracy, 4),
            "rank_correlation": round(rank_corr, 4),
        }

    elif task_id == "task_hard":
        accuracy = _label_accuracy(episode_log, gold)
        rank_corr = _rank_correlation(episode_log, gold)
        reply_qual = _reply_quality(episode_log, gold)
        score = 0.3 * accuracy + 0.2 * rank_corr + 0.5 * reply_qual
        return round(score, 4), {
            "label_accuracy": round(accuracy, 4),
            "rank_correlation": round(rank_corr, 4),
            "reply_quality": round(reply_qual, 4),
        }

    else:
        raise ValueError(f"Unknown task_id: {task_id}")


# ─── Step-Level Reward Shaping ───────────────────────────────────────

def compute_step_reward(action: Dict[str, Any], task_id: str) -> Tuple[float, Dict[str, Any]]:
    """
    Compute reward for a single step action during an episode.
    Returns (reward_value, info_dict).
    """
    gold = _load_gold_labels()
    email_id = action.get("email_id", "")

    if email_id not in gold:
        return -0.1, {"error": f"Unknown email_id: {email_id}"}

    gold_entry = gold[email_id]
    reward = 0.0
    info = {}

    # Label reward: +0.3 for correct, -0.2 for misclassifying urgent as spam
    predicted_label = action.get("label", "").lower().strip()
    gold_label = gold_entry["label"]

    if predicted_label == gold_label:
        reward += 0.3
        info["label_correct"] = True
    else:
        info["label_correct"] = False
        if gold_label == "urgent" and predicted_label == "spam":
            reward -= 0.2
            info["penalty"] = "urgent_as_spam"

    # Priority reward: +0.2 for exact match, partial for close
    predicted_priority = action.get("priority", 3)
    gold_priority = gold_entry["priority"]
    priority_diff = abs(predicted_priority - gold_priority)

    if priority_diff == 0:
        reward += 0.2
        info["priority_correct"] = True
    elif priority_diff == 1:
        reward += 0.1
        info["priority_close"] = True
    else:
        info["priority_correct"] = False

    # Reply quality reward (only for task_hard and emails needing replies)
    if task_id == "task_hard" and gold_label in ("urgent", "reply-needed"):
        keywords = gold_entry.get("reply_keywords", [])
        draft = action.get("draft_reply", "") or ""
        if keywords and draft.strip():
            hits = sum(1 for kw in keywords if kw.lower() in draft.lower())
            coverage = hits / len(keywords)
            reward += 0.5 * coverage
            info["reply_keyword_coverage"] = round(coverage, 4)
        elif keywords and not draft.strip():
            reward -= 0.1
            info["missing_reply"] = True

    # Clamp reward to [-1.0, 1.0]
    reward = max(-1.0, min(1.0, reward))

    return round(reward, 4), info
