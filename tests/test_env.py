"""Smoke tests for the Email Triage Environment.

Tests:
  - reset() returns clean state
  - step() returns correct structure
  - Grader output range [0.0, 1.0]
  - Done flag is set correctly
  - No mutation bleed between reset() calls
"""

import pytest
from env.environment import EmailTriageEnv
from env.models import EmailAction, EmailObservation
from env.tasks import TASK_REGISTRY, get_task_ids, generate_task_data
from env.graders import (
    grade_classify_basic,
    grade_triage_and_reply,
    grade_full_triage_pipeline,
)


@pytest.fixture
def env():
    """Fresh environment instance."""
    return EmailTriageEnv()


class TestReset:
    """Test reset() behavior."""

    def test_reset_returns_observation(self, env):
        obs = env.reset("classify_basic")
        assert isinstance(obs, EmailObservation)
        assert obs.task_id == "classify_basic"
        assert obs.step_count == 0
        assert len(obs.inbox) == 10
        assert obs.current_email is not None

    def test_reset_clears_state(self, env):
        """reset() must return a fresh, clean state."""
        env.reset("classify_basic")
        # Take a step
        action = EmailAction(action_type="classify", label="normal")
        env.step(action)
        # Reset again
        obs = env.reset("classify_basic")
        assert obs.step_count == 0
        assert len(obs.inbox) == 10
        state = env.state()
        assert state["step_count"] == 0
        assert state["processed_count"] == 0
        assert state["action_count"] == 0

    def test_reset_no_mutation_bleed(self, env):
        """Two consecutive resets should produce identical states."""
        obs1 = env.reset("classify_basic")
        inbox1 = [e.id for e in obs1.inbox]

        obs2 = env.reset("classify_basic")
        inbox2 = [e.id for e in obs2.inbox]

        assert inbox1 == inbox2

    def test_reset_invalid_task_raises(self, env):
        with pytest.raises(ValueError, match="Unknown task_id"):
            env.reset("nonexistent_task")

    @pytest.mark.parametrize("task_id", get_task_ids())
    def test_reset_all_tasks(self, env, task_id):
        obs = env.reset(task_id)
        assert obs.task_id == task_id
        assert obs.current_email is not None


class TestStep:
    """Test step() behavior."""

    def test_step_returns_correct_structure(self, env):
        env.reset("classify_basic")
        action = EmailAction(action_type="classify", label="urgent")
        result = env.step(action)

        assert "observation" in result
        assert "reward" in result
        assert "done" in result
        assert "info" in result
        assert isinstance(result["reward"], float)
        assert isinstance(result["done"], bool)

    def test_step_increments_count(self, env):
        env.reset("classify_basic")
        action = EmailAction(action_type="classify", label="normal")
        env.step(action)
        state = env.state()
        assert state["step_count"] == 1

    def test_step_advances_email(self, env):
        obs = env.reset("classify_basic")
        first_email_id = obs.current_email.id

        action = EmailAction(action_type="classify", label="normal")
        result = env.step(action)
        new_obs = result["observation"]

        # Current email should be different (next in queue)
        if new_obs.get("current_email"):
            assert new_obs["current_email"]["id"] != first_email_id

    def test_step_when_done_returns_zero_reward(self, env):
        env.reset("classify_basic")
        # Mark as done
        env._done = True
        action = EmailAction(action_type="classify", label="normal")
        result = env.step(action)
        assert result["done"] is True
        assert result["reward"] == 0.0

    def test_episode_completes(self, env):
        """Running through all emails should set done=True."""
        obs = env.reset("classify_basic")
        action = EmailAction(action_type="classify", label="normal")

        done = False
        steps = 0
        while not done and steps < 20:
            result = env.step(action)
            done = result["done"]
            steps += 1

        assert done is True
        assert steps <= 10  # max_steps for classify_basic


class TestGraders:
    """Test grader output ranges and properties."""

    def test_classify_grader_range(self):
        """Grader score must be in [0.0, 1.0]."""
        emails, metas = generate_task_data("classify_basic")
        metas_dicts = [m.model_dump() for m in metas]

        # All wrong
        actions = [
            {"action_type": "classify", "label": "spam", "_email_id": m["email_id"]}
            for m in metas_dicts
        ]
        score, _, _ = grade_classify_basic(actions, metas_dicts)
        assert 0.0 <= score <= 1.0

        # All correct
        actions = [
            {"action_type": "classify", "label": m["urgency"], "_email_id": m["email_id"]}
            for m in metas_dicts
        ]
        score, _, _ = grade_classify_basic(actions, metas_dicts)
        assert 0.0 <= score <= 1.0
        assert score == 1.0

    def test_triage_grader_range(self):
        emails, metas = generate_task_data("triage_and_reply")
        metas_dicts = [m.model_dump() for m in metas]

        # Empty actions
        score, _, _ = grade_triage_and_reply([], metas_dicts)
        assert 0.0 <= score <= 1.0

    def test_pipeline_grader_range(self):
        emails, metas = generate_task_data("full_triage_pipeline")
        metas_dicts = [m.model_dump() for m in metas]

        # Empty actions
        score, _, _ = grade_full_triage_pipeline([], metas_dicts)
        assert 0.0 <= score <= 1.0

    def test_grader_is_pure_function(self):
        """Same input → same output (deterministic)."""
        emails, metas = generate_task_data("classify_basic")
        metas_dicts = [m.model_dump() for m in metas]
        actions = [
            {"action_type": "classify", "label": "normal", "_email_id": m["email_id"]}
            for m in metas_dicts
        ]

        score1, p1, _ = grade_classify_basic(actions, metas_dicts)
        score2, p2, _ = grade_classify_basic(actions, metas_dicts)
        assert score1 == score2
        assert p1 == p2


class TestDoneFlag:
    """Test that done flag is set correctly."""

    def test_done_after_all_emails_processed(self, env):
        obs = env.reset("triage_and_reply")  # 5 emails
        action = EmailAction(action_type="classify", label="normal")

        for _ in range(5):
            result = env.step(action)

        assert result["done"] is True

    def test_done_at_max_steps(self, env):
        obs = env.reset("classify_basic")  # max_steps=10
        action = EmailAction(action_type="classify", label="normal")

        for _ in range(10):
            result = env.step(action)

        assert result["done"] is True


class TestRewardPenalties:
    """Test specific reward penalties."""

    def test_archive_urgent_penalty(self, env):
        """Archiving an urgent email should give negative reward."""
        obs = env.reset("classify_basic")
        # Find an urgent email
        emails, metas = generate_task_data("classify_basic")
        meta_map = {m.email_id: m for m in metas}

        env.reset("classify_basic")
        current = env._current_email
        meta = meta_map.get(current.id)

        if meta and meta.urgency == "urgent":
            action = EmailAction(action_type="archive")
            result = env.step(action)
            assert result["reward"] < 0

    def test_loop_detection_penalty(self, env):
        """Repeated action on same email should penalize."""
        env.reset("classify_basic")
        action = EmailAction(action_type="classify", label="normal")

        # Reset fresh and manually set up loop scenario
        env.reset("classify_basic")
        first_email = env._current_email
        assert first_email is not None

        # First action on this email
        env.step(action)

        # Force the same email back as current (simulate loop)
        env._current_email = first_email
        env._email_queue = [first_email] + env._email_queue
        env._done = False

        # Second action on same email should trigger loop detection
        second_result = env.step(action)
        partial_credits = second_result["info"].get("partial_credits", {})
        assert partial_credits.get("loop_penalty") == -0.15


class TestDataDeterminism:
    """Test that data generation is deterministic."""

    def test_classify_basic_deterministic(self):
        e1, m1 = generate_task_data("classify_basic")
        e2, m2 = generate_task_data("classify_basic")
        assert [e.id for e in e1] == [e.id for e in e2]
        assert [m.urgency for m in m1] == [m.urgency for m in m2]

    def test_triage_and_reply_deterministic(self):
        e1, m1 = generate_task_data("triage_and_reply")
        e2, m2 = generate_task_data("triage_and_reply")
        assert [e.id for e in e1] == [e.id for e in e2]

    def test_full_pipeline_deterministic(self):
        e1, m1 = generate_task_data("full_triage_pipeline")
        e2, m2 = generate_task_data("full_triage_pipeline")
        assert [e.id for e in e1] == [e.id for e in e2]
        assert [m.has_pii for m in m1] == [m.has_pii for m in m2]
