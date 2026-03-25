"""
Baseline inference script for the Email Triage Agent Environment.

Uses OpenAI's gpt-4o-mini to run all three tasks and report scores.
Set the OPENAI_API_KEY environment variable before running.

Usage:
    export OPENAI_API_KEY=sk-...
    python baseline.py [--env-url http://localhost:7860]
"""

import os
import sys
import json
import argparse
import requests
from typing import Dict, Any, Optional

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed. Run: pip install openai")
    sys.exit(1)


ENV_URL = "http://localhost:7860"

SYSTEM_PROMPT = """You are an expert email triage agent. Your job is to process emails one at a time.

For each email, you must output a JSON object with exactly these fields:
- "email_id": the ID of the email you're processing
- "label": one of "urgent", "normal", "spam", or "reply-needed"
- "priority": integer from 1 (highest priority) to 5 (lowest priority)
- "draft_reply": a professional reply if the email is "urgent" or "reply-needed", otherwise null

Classification guidelines:
- "urgent": Critical issues requiring immediate action (outages, security incidents, P0 bugs)
- "reply-needed": Emails requiring a response but not time-critical (contract discussions, code reviews, requests)
- "normal": Informational emails (announcements, newsletters, FYI messages)
- "spam": Unsolicited, fraudulent, or junk emails

Output ONLY valid JSON. No explanation, no markdown, no extra text."""


def parse_action(response_text: str, fallback_email_id: str) -> Dict[str, Any]:
    """Parse the LLM response into an action dict."""
    text = response_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        text = text.strip()

    try:
        action = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: try to extract JSON from the response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                action = json.loads(text[start:end])
            except json.JSONDecodeError:
                action = {}
        else:
            action = {}

    # Ensure required fields
    return {
        "email_id": action.get("email_id", fallback_email_id),
        "label": action.get("label", "normal"),
        "priority": max(1, min(5, int(action.get("priority", 3)))),
        "draft_reply": action.get("draft_reply"),
    }


def run_task(client: OpenAI, task_id: str, env_url: str, verbose: bool = False) -> float:
    """Run a single task and return the final score."""
    print(f"\n{'='*60}")
    print(f"  Running: {task_id}")
    print(f"{'='*60}")

    # Reset environment
    resp = requests.post(f"{env_url}/reset", params={"task_id": task_id})
    resp.raise_for_status()
    obs = resp.json()

    emails = obs["emails"]
    step = 0
    total_reward = 0.0
    episode_log = []

    for email in emails:
        step += 1
        email_text = (
            f"Email ID: {email['id']}\n"
            f"From: {email['sender']}\n"
            f"Subject: {email['subject']}\n"
            f"Body: {email['body']}\n"
            f"Timestamp: {email['timestamp']}"
        )

        prompt = f"Process this email and return your classification as JSON:\n\n{email_text}"

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=500,
            )
            action = parse_action(
                response.choices[0].message.content, email["id"]
            )
        except Exception as e:
            print(f"  ⚠ LLM error on step {step}: {e}")
            action = {
                "email_id": email["id"],
                "label": "normal",
                "priority": 3,
                "draft_reply": None,
            }

        # Submit to environment
        step_resp = requests.post(f"{env_url}/step", json=action)
        step_resp.raise_for_status()
        result = step_resp.json()

        reward = result["reward"]["value"]
        total_reward += reward
        episode_log.append(action)

        if verbose:
            print(
                f"  Step {step}: {email['id']} → "
                f"label={action['label']}, priority={action['priority']}, "
                f"reward={reward:+.3f}"
            )

        if result["done"]:
            break

    # Get final grade from grader
    grade_resp = requests.post(
        f"{env_url}/grader",
        json={"task_id": task_id, "episode_log": episode_log},
    )
    grade_resp.raise_for_status()
    grade = grade_resp.json()

    print(f"\n  Final Score: {grade['score']:.4f}")
    print(f"  Details: {json.dumps(grade['details'], indent=2)}")
    print(f"  Total Step Rewards: {total_reward:+.4f}")

    return grade["score"]


def main():
    parser = argparse.ArgumentParser(description="Run baseline agent on Email Triage Env")
    parser.add_argument(
        "--env-url",
        default=ENV_URL,
        help=f"Environment server URL (default: {ENV_URL})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-step details",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["task_easy", "task_medium", "task_hard"],
        help="Which tasks to run",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Usage: export OPENAI_API_KEY=sk-... && python baseline.py")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # Verify server is running
    try:
        resp = requests.get(f"{args.env_url}/tasks")
        resp.raise_for_status()
    except requests.ConnectionError:
        print(f"Error: Cannot connect to environment at {args.env_url}")
        print("Start the server first: uvicorn server:app --port 7860")
        sys.exit(1)

    print("=" * 60)
    print("  Email Triage Agent — Baseline (gpt-4o-mini)")
    print("=" * 60)

    scores = {}
    for task_id in args.tasks:
        scores[task_id] = run_task(client, task_id, args.env_url, args.verbose)

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for task_id, score in scores.items():
        print(f"  {task_id}: {score:.4f}")
    print(f"  Average:  {sum(scores.values()) / len(scores):.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
