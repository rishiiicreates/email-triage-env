"""Core Email Triage Environment implementing reset / step / state."""

import json
import os
from typing import Dict, Any, Tuple, List, Optional

from .models import Email, Observation, Action, Reward, StepResult
from .tasks import (
    TASK_DEFINITIONS,
    get_task_emails,
    compute_step_reward,
    grade_episode,
)


class EmailTriageEnv:
    """
    An OpenEnv-compliant environment for email triage.

    Agents observe an inbox of emails and must classify, prioritize,
    and optionally draft replies. The environment supports three tasks
    of increasing difficulty.
    """

    def __init__(self):
        self.current_task_id: Optional[str] = None
        self.emails: List[Email] = []
        self.step_count: int = 0
        self.max_steps: int = 15
        self.processed_email_ids: set = set()
        self.episode_log: List[Dict[str, Any]] = []
        self.done: bool = False
        self._state_data: Dict[str, Any] = {}

    def reset(self, task_id: str) -> Observation:
        """Reset the environment for a new episode with the given task."""
        if task_id not in TASK_DEFINITIONS:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Valid tasks: {list(TASK_DEFINITIONS.keys())}"
            )

        task_def = TASK_DEFINITIONS[task_id]
        raw_emails = get_task_emails(task_id)

        self.current_task_id = task_id
        self.emails = [Email(**e) for e in raw_emails]
        self.step_count = 0
        self.max_steps = task_def["max_steps"]
        self.processed_email_ids = set()
        self.episode_log = []
        self.done = False

        self._state_data = {
            "task_id": task_id,
            "task_name": task_def["name"],
            "difficulty": task_def["difficulty"],
            "total_emails": len(self.emails),
            "step_count": 0,
            "max_steps": self.max_steps,
            "processed_email_ids": [],
            "done": False,
        }

        return Observation(
            emails=self.emails,
            step_number=0,
            task_id=task_id,
            remaining_emails=len(self.emails),
        )

    def step(self, action: Action) -> StepResult:
        """
        Process one agent action and return the next observation + reward.
        """
        if self.done:
            return StepResult(
                observation=Observation(
                    emails=[],
                    step_number=self.step_count,
                    task_id=self.current_task_id or "",
                    remaining_emails=0,
                ),
                reward=Reward(value=0.0, done=True, info={"message": "Episode already done"}),
                done=True,
                info={"message": "Episode already done"},
            )

        self.step_count += 1

        # Record the action in the episode log
        action_dict = action.model_dump()
        self.episode_log.append(action_dict)
        self.processed_email_ids.add(action.email_id)

        # Compute step reward
        reward_value, reward_info = compute_step_reward(
            action_dict, self.current_task_id
        )

        # Check if the episode is done
        remaining = [e for e in self.emails if e.id not in self.processed_email_ids]
        self.done = (
            self.step_count >= self.max_steps or len(remaining) == 0
        )

        # If done, compute final episode grade and include it
        if self.done:
            final_score, grade_details = grade_episode(
                self.current_task_id, self.episode_log
            )
            reward_info["final_episode_score"] = final_score
            reward_info["grade_details"] = grade_details

        # Update state
        self._state_data.update({
            "step_count": self.step_count,
            "processed_email_ids": list(self.processed_email_ids),
            "done": self.done,
        })

        obs = Observation(
            emails=remaining if not self.done else [],
            step_number=self.step_count,
            task_id=self.current_task_id or "",
            remaining_emails=len(remaining),
        )

        reward = Reward(
            value=reward_value,
            done=self.done,
            info=reward_info,
        )

        return StepResult(
            observation=obs,
            reward=reward,
            done=self.done,
            info=reward_info,
        )

    def state(self) -> Dict[str, Any]:
        """Return the current environment state."""
        return {
            **self._state_data,
            "episode_log": self.episode_log,
        }
