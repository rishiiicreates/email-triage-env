---
title: Email Triage Env
emoji: 📬
colorFrom: purple
colorTo: teal
sdk: docker
tags: [openenv]
pinned: false
---

# 📬 Email Triage Agent Environment

An **OpenEnv-compliant** environment where AI agents triage, classify, and respond to realistic email inboxes. Designed as a benchmark for evaluating agent decision-making on real-world productivity tasks.

## Why Email Triage?

Email triage is a universal, genuinely hard problem. It requires:
- **Classification** — distinguishing spam from urgent production alerts
- **Prioritization** — ranking emails by importance and time-sensitivity
- **Composition** — drafting contextually appropriate replies

The environment scales naturally from simple classification to complex multi-step reasoning, making it an ideal testbed for agent evaluation.

---

## 🔭 Observation Space

Each observation contains the current inbox state:

| Field | Type | Description |
|-------|------|-------------|
| `emails` | `List[Email]` | List of emails to process |
| `step_number` | `int` | Current step in the episode |
| `task_id` | `str` | Active task identifier |
| `remaining_emails` | `int` | Number of unprocessed emails |

Each **Email** has:

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique email identifier (e.g., `email_01`) |
| `sender` | `str` | Sender email address |
| `subject` | `str` | Email subject line |
| `body` | `str` | Full email body text |
| `timestamp` | `str` | ISO 8601 timestamp |

---

## 🎯 Action Space

Each action processes one email:

| Field | Type | Valid Values | Description |
|-------|------|-------------|-------------|
| `email_id` | `str` | Any email ID from observation | Which email to process |
| `label` | `str` | `"urgent"`, `"normal"`, `"spam"`, `"reply-needed"` | Classification label |
| `priority` | `int` | `1` (highest) to `5` (lowest) | Priority ranking |
| `draft_reply` | `str \| null` | Free-form text or null | Draft reply (required for `task_hard`) |

---

## 📋 Task Descriptions

### Task 1: Classify Urgency (`task_easy`)
**Difficulty:** Easy

Classify each of the 15 emails in the inbox as `urgent`, `normal`, `spam`, or `reply-needed`. The agent is scored solely on **exact-match accuracy** against pre-determined gold labels.

**Grader:** `score = correct_labels / total_emails`

---

### Task 2: Label and Prioritize Inbox (`task_medium`)
**Difficulty:** Medium

Classify each email AND assign a priority ranking (1–5). The agent is evaluated on both its label accuracy and how well its priority ordering matches the gold standard.

**Grader:** `score = 0.5 × label_accuracy + 0.5 × normalized_kendall_tau`

---

### Task 3: Triage and Draft Replies (`task_hard`)
**Difficulty:** Hard

Classify, prioritize, AND draft professional replies for emails labeled `urgent` or `reply-needed`. Reply quality is measured by keyword coverage against gold reply templates.

**Grader:** `score = 0.3 × label_accuracy + 0.2 × rank_correlation + 0.5 × reply_keyword_coverage`

---

## ⚡ Setup Instructions

### Prerequisites
- Python 3.11+
- Docker (for containerized deployment)
- OpenAI API key (for baseline only)

### Local Development

```bash
# Clone the repository
git clone <repo-url>
cd email-triage-env

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn server:app --host 0.0.0.0 --port 7860

# Test the endpoints
curl http://localhost:7860/tasks
curl -X POST "http://localhost:7860/reset?task_id=task_easy"
```

### Docker

```bash
docker build -t email-triage-env .
docker run -p 7860:7860 email-triage-env

# Verify
curl http://localhost:7860/tasks
```

### Run Baseline Agent

```bash
export OPENAI_API_KEY=sk-...
python baseline.py --verbose
```

---

## 📊 Baseline Scores (gpt-4o-mini)

| Task | Score | Details |
|------|-------|---------|
| `task_easy` | ~0.87 | Label accuracy |
| `task_medium` | ~0.72 | 50% label acc + 50% rank corr |
| `task_hard` | ~0.51 | 30% label + 20% rank + 50% reply |

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/tasks` | GET | List all available tasks with action schemas |
| `/reset?task_id=...` | POST | Reset environment for a new episode |
| `/step` | POST | Submit an action, get observation + reward |
| `/state` | GET | Get current environment state |
| `/grader` | POST | Grade a complete episode log |
| `/baseline` | GET | Baseline agent information |

---

## 📐 Reward Shaping

Per-step rewards provide learning signal during episodes:

| Condition | Reward |
|-----------|--------|
| Correct label | +0.3 |
| Correct priority (exact) | +0.2 |
| Close priority (±1) | +0.1 |
| Reply keyword coverage | +0.5 × coverage |
| Misclassify urgent as spam | −0.2 |
| Missing required reply | −0.1 |

---

## License

MIT
