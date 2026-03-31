"""Core Email Triage Environment implementing reset / step / state.

OpenEnv-compliant environment supporting 3 tasks of increasing difficulty.
Each call to reset() returns a completely fresh state with no mutation bleed.
"""

from __future__ import annotations

import copy
from typing import Dict, Any, List, Optional, Set

from .models import (
    Email,
    EmailMeta,
    EmailObservation,
    EmailAction,
    EmailReward,
)
from .tasks import get_task_config, generate_task_data, grade_episode


class EmailTriageEnv:
    """
    An OpenEnv-compliant environment for email triage.

    Agents observe an inbox of emails and must classify, prioritize,
    reply to, and route emails. The environment supports three tasks
    of increasing difficulty.
    """

    def __init__(self) -> None:
        self._task_id: Optional[str] = None
        self._emails: List[Email] = []
        self._metas: List[EmailMeta] = []
        self._meta_map: Dict[str, EmailMeta] = {}
        self._step_count: int = 0
        self._max_steps: int = 10
        self._email_queue: List[Email] = []
        self._current_email: Optional[Email] = None
        self._processed_ids: Set[str] = set()
        self._action_history: List[Dict[str, Any]] = []
        self._actions_per_email: Dict[str, List[str]] = {}
        self._done: bool = True
        self._cumulative_reward: float = 0.0

    @property
    def current_task_id(self) -> Optional[str]:
        return self._task_id

    def reset(self, task_id: str) -> EmailObservation:
        """Reset the environment for a new episode. Returns a completely fresh state."""
        config = get_task_config(task_id)
        emails, metas = generate_task_data(task_id)

        # Deep-copy to prevent mutation bleed
        self._task_id = task_id
        self._emails = copy.deepcopy(emails)
        self._metas = copy.deepcopy(metas)
        self._meta_map = {m.email_id: m for m in self._metas}
        self._step_count = 0
        self._max_steps = config["max_steps"]
        self._email_queue = list(self._emails)
        self._current_email = self._email_queue[0] if self._email_queue else None
        self._processed_ids = set()
        self._action_history = []
        self._actions_per_email = {}
        self._done = False
        self._cumulative_reward = 0.0

        return EmailObservation(
            inbox=self._emails,
            current_email=self._current_email,
            step_count=0,
            task_id=task_id,
            context={
                "task_name": config["name"],
                "difficulty": config["difficulty"],
                "total_emails": len(self._emails),
                "remaining_emails": len(self._email_queue),
                "max_steps": self._max_steps,
                "description": config["description"],
            },
        )

    def step(self, action: EmailAction) -> Dict[str, Any]:
        """
        Process one agent action and return observation, reward, done, info.

        Returns dict matching StepResponse schema:
          { observation, reward, done, info }
        """
        if self._done:
            return {
                "observation": EmailObservation(
                    inbox=self._emails,
                    current_email=None,
                    step_count=self._step_count,
                    task_id=self._task_id or "",
                    context={"message": "Episode already done"},
                ).model_dump(),
                "reward": 0.0,
                "done": True,
                "info": {"message": "Episode already done"},
            }

        self._step_count += 1
        current = self._current_email

        # Build action record (attach the email ID for graders)
        action_dict = action.model_dump()
        if current:
            action_dict["_email_id"] = current.id

        # --- Compute step reward ---
        reward, partials, explanation = self._compute_step_reward(action, current)
        self._cumulative_reward += reward

        # Record action
        self._action_history.append(action_dict)

        # Track per-email actions for loop detection
        if current:
            eid = current.id
            self._actions_per_email.setdefault(eid, [])
            self._actions_per_email[eid].append(action.action_type)

        # Advance queue: if action processes the current email, move to next
        if current and action.action_type in ("classify", "reply", "route", "archive", "flag"):
            self._processed_ids.add(current.id)
            # Remove from queue
            self._email_queue = [e for e in self._email_queue if e.id != current.id]
            self._current_email = self._email_queue[0] if self._email_queue else None

        # Check termination
        all_processed = len(self._email_queue) == 0
        at_max_steps = self._step_count >= self._max_steps
        self._done = all_processed or at_max_steps

        # Episode completion bonus
        info: Dict[str, Any] = {
            "step_reward": reward,
            "partial_credits": partials,
            "explanation": explanation,
        }

        if self._done:
            # Grade the full episode
            metas_dicts = [m.model_dump() for m in self._metas]
            ep_score, ep_partials, ep_explanation = grade_episode(
                self._task_id, self._action_history, metas_dicts
            )
            # Efficiency bonus
            efficiency_bonus = 0.0
            if self._step_count < self._max_steps * 0.6:
                efficiency_bonus = 0.1
            final_score = min(1.0, ep_score + efficiency_bonus)

            info["episode_score"] = round(final_score, 4)
            info["episode_breakdown"] = ep_partials
            info["episode_explanation"] = ep_explanation
            info["efficiency_bonus"] = efficiency_bonus
            info["steps_used"] = self._step_count
            info["max_steps"] = self._max_steps

        obs = EmailObservation(
            inbox=self._emails,
            current_email=self._current_email,
            step_count=self._step_count,
            task_id=self._task_id or "",
            context={
                "remaining_emails": len(self._email_queue),
                "processed_count": len(self._processed_ids),
                "total_emails": len(self._emails),
            },
        )

        return {
            "observation": obs.model_dump(),
            "reward": round(reward, 4),
            "done": self._done,
            "info": info,
        }

    def state(self) -> Dict[str, Any]:
        """Return the current environment state."""
        return {
            "task_id": self._task_id,
            "step_count": self._step_count,
            "max_steps": self._max_steps,
            "done": self._done,
            "total_emails": len(self._emails),
            "processed_count": len(self._processed_ids),
            "remaining_emails": len(self._email_queue),
            "cumulative_reward": round(self._cumulative_reward, 4),
            "action_count": len(self._action_history),
        }

    # ─── Step-Level Reward Shaping ───────────────────────────────────

    def _compute_step_reward(
        self,
        action: EmailAction,
        current_email: Optional[Email],
    ) -> tuple[float, Dict[str, float], str]:
        """
        Compute dense per-step reward with partial credit.

        Penalties:
          -0.15 for repeated action on same email (loop detection)
          -0.10 for archiving a truly urgent email
          -0.05 for reply_text under 10 chars when reply is required
        """
        if current_email is None:
            return 0.0, {}, "No current email to act on"

        eid = current_email.id
        meta = self._meta_map.get(eid)
        if meta is None:
            return 0.0, {}, f"No metadata for email {eid}"

        reward = 0.0
        partials: Dict[str, float] = {}
        explanations: list[str] = []

        # --- Loop detection penalty ---
        prev_actions = self._actions_per_email.get(eid, [])
        if action.action_type in prev_actions:
            reward -= 0.15
            partials["loop_penalty"] = -0.15
            explanations.append(f"Repeated {action.action_type} on {eid} (-0.15)")

        # --- Classify action ---
        if action.action_type == "classify":
            pred_label = (action.label or "").lower().strip()
            gold_label = meta.urgency
            if pred_label == gold_label:
                reward += 0.1
                partials["label_correct"] = 0.1
                explanations.append(f"Correct label '{pred_label}' (+0.1)")
            else:
                # Partial credit: at least they tried
                partials["label_correct"] = 0.0
                explanations.append(f"Wrong label: predicted '{pred_label}', gold '{gold_label}'")

        # --- Reply action ---
        elif action.action_type == "reply":
            reply_text = action.reply_text or ""
            gold_reply = meta.gold_reply or ""

            if meta.urgency == "urgent" and gold_reply:
                if not reply_text.strip():
                    reward -= 0.2
                    partials["empty_reply_penalty"] = -0.2
                    explanations.append("Empty reply on urgent email (-0.2)")
                elif len(reply_text.strip()) < 10:
                    reward -= 0.05
                    partials["short_reply_penalty"] = -0.05
                    explanations.append("Reply under 10 chars (-0.05)")
                else:
                    # Compute similarity
                    from .graders import _text_similarity, _tone_score
                    sim = _text_similarity(reply_text, gold_reply)
                    tone = _tone_score(reply_text)
                    reply_reward = 0.15 * sim + 0.05 * tone
                    reward += reply_reward
                    partials["reply_similarity"] = round(sim, 4)
                    partials["reply_tone"] = round(tone, 4)
                    explanations.append(
                        f"Reply similarity={sim:.2f}, tone={tone:.2f} (+{reply_reward:.3f})"
                    )
            elif meta.urgency != "urgent":
                # Reply to non-urgent: small positive signal
                reward += 0.02
                partials["voluntary_reply"] = 0.02
                explanations.append("Reply to non-urgent email (+0.02)")

        # --- Route action ---
        elif action.action_type == "route":
            pred_dept = (action.route_to or "").strip()
            gold_dept = meta.department
            if pred_dept.lower() == gold_dept.lower():
                reward += 0.1
                partials["route_correct"] = 0.1
                explanations.append(f"Correct routing to {gold_dept} (+0.1)")
            else:
                partials["route_correct"] = 0.0
                explanations.append(f"Wrong route: '{pred_dept}', should be '{gold_dept}'")

        # --- Archive action ---
        elif action.action_type == "archive":
            if meta.urgency == "urgent":
                reward -= 0.10
                partials["archive_urgent_penalty"] = -0.10
                explanations.append("Archived urgent email (-0.10)")
            else:
                reward += 0.03
                partials["archive_ok"] = 0.03
                explanations.append("Archived non-urgent email (+0.03)")

        # --- Flag action (PII) ---
        elif action.action_type == "flag":
            if meta.has_pii:
                reward += 0.1
                partials["pii_flag_correct"] = 0.1
                explanations.append("Correctly flagged PII email (+0.1)")
            else:
                reward += 0.01  # small positive for trying
                partials["pii_flag_false_positive"] = 0.01
                explanations.append("Flagged non-PII email (+0.01)")

        # Clamp reward
        reward = max(-1.0, min(1.0, reward))
        explanation = "; ".join(explanations) if explanations else "No signal"

        return round(reward, 4), partials, explanation
