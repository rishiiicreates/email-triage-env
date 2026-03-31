"""
Baseline inference script for the Email Triage Agent Environment.

Uses HuggingFace Inference API via OpenAI-compatible client.
Runs all 3 tasks and prints a score summary table.

Environment variables:
  API_BASE_URL  = HuggingFace router URL (default: https://router.huggingface.co/v1)
  MODEL_NAME    = Model to use (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN      = HuggingFace API token (fallback: API_KEY)
  ENV_URL       = Environment server URL (default: http://localhost:7860)
"""

import os
import sys
import json
import time
import requests

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed. Run: pip install openai")
    sys.exit(1)

# ─── Configuration ───────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

MAX_STEPS_PER_EPISODE = 8
TEMPERATURE = 0.0

TASK_IDS = ["classify_basic", "triage_and_reply", "full_triage_pipeline"]

# ─── System Prompt ───────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert email triage agent. You process emails one at a time from a corporate inbox.

For EACH email, you must decide what actions to take and output them as a JSON array of action objects.
Each action object must have these fields:
- "action_type": one of "classify", "reply", "route", "archive", "flag"
- "label": (for classify) one of "urgent", "normal", "spam"
- "reply_text": (for reply) a professional reply text, minimum 10 characters
- "route_to": (for route) one of "Engineering", "Sales", "HR"
- "reasoning": brief explanation of your decision

CLASSIFICATION GUIDELINES:
- "urgent": Critical issues needing immediate action — outages, security incidents, expiring contracts, compliance issues
- "normal": Routine business — policy updates, meeting requests, code reviews, invoices
- "spam": Unsolicited/fraudulent — lotteries, fake deals, phishing (watch for spoofed domains like paypa1.com, c0mpany.com)

PII DETECTION:
- Flag emails containing SSNs (###-##-####), phone numbers, personal email addresses, bank account numbers, dates of birth

ROUTING GUIDELINES:
- Engineering: technical issues, code reviews, infrastructure, security
- Sales: client communications, contracts, invoices, deals
- HR: policy updates, onboarding, compliance, workplace issues

For each email, output ONLY a JSON array with the actions you take. Example for an urgent engineering email:
[
  {"action_type": "classify", "label": "urgent", "reasoning": "Production outage"},
  {"action_type": "reply", "reply_text": "Acknowledged. Investigating the issue immediately and will provide an update within 30 minutes.", "reasoning": "Urgent issue needs quick response"},
  {"action_type": "route", "route_to": "Engineering", "reasoning": "Technical infrastructure issue"}
]

For a simple spam email:
[
  {"action_type": "classify", "label": "spam", "reasoning": "Phishing attempt from spoofed domain"}
]

For an email with PII:
[
  {"action_type": "classify", "label": "normal", "reasoning": "HR onboarding document"},
  {"action_type": "flag", "reasoning": "Contains SSN and personal contact info"},
  {"action_type": "route", "route_to": "HR", "reasoning": "Employee onboarding"}
]

Output ONLY the JSON array. No markdown, no explanation outside the JSON."""


def parse_actions(response_text: str) -> list[dict]:
    """Parse the LLM response into a list of action dicts."""
    text = response_text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(
            lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        )
        text = text.strip()

    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return [result]
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        # Try to extract JSON array
        start = text.find("[")
        end = text.rfind("]") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        # Try single object
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return [json.loads(text[start:end])]
            except json.JSONDecodeError:
                pass

    # Fallback
    return [{"action_type": "classify", "label": "normal", "reasoning": "parse_error"}]


def run_task(client: OpenAI, task_id: str) -> float:
    """Run a single task episode and return the final episode score."""
    difficulty_map = {
        "classify_basic": "easy",
        "triage_and_reply": "medium",
        "full_triage_pipeline": "hard",
    }
    difficulty = difficulty_map.get(task_id, "?")
    print(f"\n{'━' * 60}")
    print(f"  Task: {task_id} ({difficulty})")
    print(f"{'━' * 60}")

    # Reset environment
    resp = requests.post(f"{ENV_URL}/reset", params={"task_id": task_id})
    if resp.status_code != 200:
        print(f"  ERROR: Reset failed — {resp.status_code}: {resp.text}")
        return 0.0
    reset_data = resp.json()
    observation = reset_data["observation"]
    inbox = observation["inbox"]
    current_email = observation.get("current_email")

    print(f"  Inbox size: {len(inbox)} emails")
    print(f"  Max steps: {observation['context'].get('max_steps', '?')}")

    step_count = 0
    total_step_reward = 0.0
    episode_score = 0.0

    while current_email and step_count < MAX_STEPS_PER_EPISODE:
        step_count += 1

        # Build email context for the LLM
        email_text = (
            f"Email ID: {current_email['id']}\n"
            f"From: {current_email['sender_name']} <{current_email['sender_email']}>\n"
            f"Department: {current_email['department']}\n"
            f"Subject: {current_email['subject']}\n"
            f"Body: {current_email['body']}\n"
            f"Timestamp: {current_email['timestamp']}"
        )

        prompt = (
            f"Process this email. Task: {task_id}. "
            f"Decide which actions to take and output as JSON array.\n\n{email_text}"
        )

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=800,
            )
            raw_response = response.choices[0].message.content or ""
            actions = parse_actions(raw_response)
        except Exception as e:
            print(f"  ⚠ LLM error on step {step_count}: {e}")
            actions = [
                {"action_type": "classify", "label": "normal", "reasoning": "LLM error fallback"}
            ]

        # Execute each action via the environment
        for action_dict in actions:
            if step_count > MAX_STEPS_PER_EPISODE:
                break

            # Build clean action payload
            payload = {
                "action_type": action_dict.get("action_type", "classify"),
                "label": action_dict.get("label"),
                "reply_text": action_dict.get("reply_text"),
                "route_to": action_dict.get("route_to"),
                "reasoning": action_dict.get("reasoning"),
            }

            step_resp = requests.post(f"{ENV_URL}/step", json=payload)
            if step_resp.status_code != 200:
                print(f"  ⚠ Step error: {step_resp.status_code}: {step_resp.text}")
                continue

            result = step_resp.json()
            reward = result.get("reward", 0.0)
            done = result.get("done", False)
            info = result.get("info", {})
            total_step_reward += reward

            print(
                f"  Step {step_count}: {current_email['id']} → "
                f"{payload['action_type']}"
                f"{' label=' + payload['label'] if payload.get('label') else ''}"
                f"{' route=' + payload['route_to'] if payload.get('route_to') else ''}"
                f"  reward={reward:+.3f}"
            )

            if done:
                episode_score = info.get("episode_score", 0.0)
                print(f"\n  ✓ Episode complete at step {step_count}")
                print(f"  Episode Score: {episode_score:.4f}")
                if info.get("episode_breakdown"):
                    for k, v in info["episode_breakdown"].items():
                        print(f"    {k}: {v:.4f}")
                if info.get("efficiency_bonus", 0) > 0:
                    print(f"    efficiency_bonus: +{info['efficiency_bonus']:.2f}")
                return episode_score

            # Update current email from observation
            new_obs = result.get("observation", {})
            current_email = new_obs.get("current_email")
            if current_email is None:
                # No more emails
                break

    # If we ran out of steps without done signal, get final state
    state_resp = requests.get(f"{ENV_URL}/state")
    if state_resp.status_code == 200:
        state = state_resp.json().get("state", {})
        print(f"\n  Ran out of steps. Steps used: {step_count}")
        print(f"  Cumulative reward: {total_step_reward:+.4f}")

    return episode_score


def main():
    """Run inference on all tasks and print summary."""
    if not HF_TOKEN:
        print("ERROR: No API token found.")
        print("Set HF_TOKEN or API_KEY environment variable.")
        print("  export HF_TOKEN=hf_...")
        sys.exit(1)

    # Verify server is running
    try:
        health = requests.get(f"{ENV_URL}/health", timeout=5)
        health.raise_for_status()
    except (requests.ConnectionError, requests.Timeout):
        print(f"ERROR: Cannot connect to environment at {ENV_URL}")
        print("Start the server first:")
        print("  uvicorn env.server:app --host 0.0.0.0 --port 7860")
        sys.exit(1)

    client = OpenAI(
        api_key=HF_TOKEN,
        base_url=API_BASE_URL,
    )

    print("=" * 60)
    print("  Email Triage Agent — Baseline Inference")
    print(f"  Model: {MODEL_NAME}")
    print(f"  API:   {API_BASE_URL}")
    print(f"  Env:   {ENV_URL}")
    print("=" * 60)

    start_time = time.time()
    scores = {}

    for task_id in TASK_IDS:
        scores[task_id] = run_task(client, task_id)

    elapsed = time.time() - start_time

    # ─── Summary Table ───────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  FINAL RESULTS")
    print(f"{'=' * 60}")

    difficulty_labels = {
        "classify_basic": "easy",
        "triage_and_reply": "medium",
        "full_triage_pipeline": "hard",
    }
    for task_id in TASK_IDS:
        diff = difficulty_labels.get(task_id, "?")
        print(f"  Task: {task_id} ({diff}):   score = {scores[task_id]:.2f}")

    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"  Average:                      {avg:.2f}")
    print(f"  Time elapsed:                 {elapsed:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
