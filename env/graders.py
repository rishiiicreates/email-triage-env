"""Deterministic graders for the Email Triage Environment.

Each grader is a pure function: same input → same output, always.
No external API calls — all logic is offline.
Graders return scores in [0.0, 1.0].
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ─── Shared Utilities ────────────────────────────────────────────────


def _text_similarity(text_a: str, text_b: str) -> float:
    """Compute cosine similarity between two texts using TF-IDF vectors."""
    if not text_a.strip() or not text_b.strip():
        return 0.0
    try:
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform([text_a, text_b])
        sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        return float(max(0.0, min(1.0, sim)))
    except ValueError:
        return 0.0


_PROFESSIONAL_KEYWORDS = [
    "acknowledged", "thank", "apolog", "investigate", "coordinate",
    "update", "immediately", "escalat", "confirm", "review", "ensure",
    "please", "team", "will", "contact", "priority", "asap", "resolve",
]

_UNPROFESSIONAL_KEYWORDS = [
    "lol", "lmao", "haha", "dude", "bro", "wtf", "omg", "yolo",
    "whatever", "idk", "nah", "chill",
]


def _tone_score(reply_text: str) -> float:
    """Score reply tone on [0, 1] using keyword heuristics."""
    if not reply_text.strip():
        return 0.0
    text_lower = reply_text.lower()
    pro_hits = sum(1 for kw in _PROFESSIONAL_KEYWORDS if kw in text_lower)
    unpro_hits = sum(1 for kw in _UNPROFESSIONAL_KEYWORDS if kw in text_lower)
    # Normalize professional hits out of a reasonable max
    pro_score = min(1.0, pro_hits / 4.0)
    # Penalize unprofessional language
    penalty = min(1.0, unpro_hits * 0.25)
    return max(0.0, pro_score - penalty)


# ─── Task 1: classify_basic ──────────────────────────────────────────


def grade_classify_basic(
    actions: List[Dict[str, Any]],
    metas: List[Dict[str, Any]],
) -> Tuple[float, Dict[str, float], str]:
    """
    Grade classification accuracy.
    Returns (score, partial_credits, explanation).
    """
    meta_map = {m["email_id"]: m for m in metas}
    total = len(metas)
    correct = 0
    per_email: Dict[str, float] = {}

    for action in actions:
        if action.get("action_type") != "classify":
            continue
        email_id = action.get("_email_id", "")
        if email_id not in meta_map:
            continue
        gold = meta_map[email_id]["urgency"]
        pred = (action.get("label") or "").lower().strip()
        is_correct = pred == gold
        if is_correct:
            correct += 1
        per_email[email_id] = 1.0 if is_correct else 0.0

    score = correct / total if total > 0 else 0.0
    return (
        score,
        per_email,
        f"Classification accuracy: {correct}/{total} = {score:.2f}",
    )


# ─── Task 2: triage_and_reply ────────────────────────────────────────


def grade_triage_and_reply(
    actions: List[Dict[str, Any]],
    metas: List[Dict[str, Any]],
) -> Tuple[float, Dict[str, float], str]:
    """
    Grade urgency detection + reply quality + tone.
    Weights: urgency 0.3, reply relevance 0.4, tone 0.3.
    Penalize empty replies on urgent emails: -0.2 each.
    """
    meta_map = {m["email_id"]: m for m in metas}
    actions_by_email: Dict[str, List[Dict]] = {}
    for a in actions:
        eid = a.get("_email_id", "")
        if eid:
            actions_by_email.setdefault(eid, []).append(a)

    # --- Urgency detection ---
    urgent_emails = [m for m in metas if m["urgency"] == "urgent"]
    urgency_correct = 0
    for m in metas:
        eid = m["email_id"]
        email_actions = actions_by_email.get(eid, [])
        classify_actions = [a for a in email_actions if a.get("action_type") == "classify"]
        if classify_actions:
            pred = (classify_actions[0].get("label") or "").lower().strip()
            if pred == m["urgency"]:
                urgency_correct += 1
    urgency_score = urgency_correct / len(metas) if metas else 0.0

    # --- Reply quality (only for urgent emails) ---
    reply_scores = []
    empty_penalty = 0.0
    for m in urgent_emails:
        eid = m["email_id"]
        email_actions = actions_by_email.get(eid, [])
        reply_actions = [a for a in email_actions if a.get("action_type") == "reply"]
        gold_reply = m.get("gold_reply", "")

        if not reply_actions or not (reply_actions[0].get("reply_text") or "").strip():
            empty_penalty += 0.2
            reply_scores.append(0.0)
        else:
            reply_text = reply_actions[0]["reply_text"]
            if len(reply_text.strip()) < 10:
                empty_penalty += 0.05
            sim = _text_similarity(reply_text, gold_reply) if gold_reply else 0.5
            reply_scores.append(sim)

    reply_score = sum(reply_scores) / len(reply_scores) if reply_scores else 0.0

    # --- Tone ---
    tone_scores = []
    for m in urgent_emails:
        eid = m["email_id"]
        email_actions = actions_by_email.get(eid, [])
        reply_actions = [a for a in email_actions if a.get("action_type") == "reply"]
        if reply_actions:
            tone_scores.append(_tone_score(reply_actions[0].get("reply_text", "")))
        else:
            tone_scores.append(0.0)
    tone_avg = sum(tone_scores) / len(tone_scores) if tone_scores else 0.0

    # Weighted sum
    raw_score = 0.3 * urgency_score + 0.4 * reply_score + 0.3 * tone_avg
    final_score = max(0.0, min(1.0, raw_score - empty_penalty))

    partials = {
        "urgency_detection": round(urgency_score, 4),
        "reply_relevance": round(reply_score, 4),
        "tone_appropriateness": round(tone_avg, 4),
        "empty_reply_penalty": round(-empty_penalty, 4),
    }
    explanation = (
        f"Urgency: {urgency_score:.2f} (w=0.3), "
        f"Reply: {reply_score:.2f} (w=0.4), "
        f"Tone: {tone_avg:.2f} (w=0.3), "
        f"Penalty: -{empty_penalty:.2f}"
    )
    return final_score, partials, explanation


# ─── Task 3: full_triage_pipeline ────────────────────────────────────


def grade_full_triage_pipeline(
    actions: List[Dict[str, Any]],
    metas: List[Dict[str, Any]],
) -> Tuple[float, Dict[str, float], str]:
    """
    Grade full pipeline: classify, route, reply, flag PII.
    Weights: classification 0.25, routing 0.30, reply 0.25, PII 0.20.
    """
    meta_map = {m["email_id"]: m for m in metas}
    actions_by_email: Dict[str, List[Dict]] = {}
    for a in actions:
        eid = a.get("_email_id", "")
        if eid:
            actions_by_email.setdefault(eid, []).append(a)

    total = len(metas)

    # --- Classification accuracy (0.25) ---
    classify_correct = 0
    for m in metas:
        eid = m["email_id"]
        email_actions = actions_by_email.get(eid, [])
        classify_actions = [a for a in email_actions if a.get("action_type") == "classify"]
        if classify_actions:
            pred = (classify_actions[0].get("label") or "").lower().strip()
            if pred == m["urgency"]:
                classify_correct += 1
    classify_score = classify_correct / total if total > 0 else 0.0

    # --- Routing accuracy (0.30) ---
    route_correct = 0
    route_total = 0
    for m in metas:
        eid = m["email_id"]
        email_actions = actions_by_email.get(eid, [])
        route_actions = [a for a in email_actions if a.get("action_type") == "route"]
        if route_actions:
            route_total += 1
            pred = (route_actions[0].get("route_to") or "").strip()
            if pred.lower() == m["department"].lower():
                route_correct += 1
    route_score = route_correct / total if total > 0 else 0.0

    # --- Reply quality for top-3 urgent (0.25) ---
    urgent_metas = [m for m in metas if m["urgency"] == "urgent"]
    urgent_metas = urgent_metas[:3]  # top-3 urgent
    reply_scores = []
    for m in urgent_metas:
        eid = m["email_id"]
        email_actions = actions_by_email.get(eid, [])
        reply_actions = [a for a in email_actions if a.get("action_type") == "reply"]
        gold_reply = m.get("gold_reply", "")
        if reply_actions and (reply_actions[0].get("reply_text") or "").strip():
            reply_text = reply_actions[0]["reply_text"]
            sim = _text_similarity(reply_text, gold_reply) if gold_reply else 0.5
            reply_scores.append(sim)
        else:
            reply_scores.append(0.0)
    reply_score = sum(reply_scores) / len(reply_scores) if reply_scores else 0.0

    # --- PII flagging recall (0.20) ---
    pii_emails = [m for m in metas if m.get("has_pii", False)]
    pii_flagged = 0
    for m in pii_emails:
        eid = m["email_id"]
        email_actions = actions_by_email.get(eid, [])
        flag_actions = [a for a in email_actions if a.get("action_type") == "flag"]
        if flag_actions:
            pii_flagged += 1
    pii_recall = pii_flagged / len(pii_emails) if pii_emails else 1.0

    # Weighted sum
    score = (
        0.25 * classify_score
        + 0.30 * route_score
        + 0.25 * reply_score
        + 0.20 * pii_recall
    )
    score = max(0.0, min(1.0, score))

    partials = {
        "classification_accuracy": round(classify_score, 4),
        "routing_accuracy": round(route_score, 4),
        "reply_quality": round(reply_score, 4),
        "pii_flagging_recall": round(pii_recall, 4),
    }
    explanation = (
        f"Classification: {classify_score:.2f} (w=0.25), "
        f"Routing: {route_score:.2f} (w=0.30), "
        f"Reply: {reply_score:.2f} (w=0.25), "
        f"PII Recall: {pii_recall:.2f} (w=0.20)"
    )
    return score, partials, explanation
