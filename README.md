---
title: Email Triage Env
emoji: 📬
colorFrom: purple
colorTo: green
sdk: docker
tags: [openenv]
pinned: false
---

# 📬 Email Triage Agent Environment

An **OpenEnv-compliant** environment where AI agents triage, classify, reply to, route, and flag realistic email inboxes. Designed as a benchmark for evaluating agent decision-making on real-world productivity tasks.

## Why Email Triage?

Email triage is a universal, genuinely hard problem with immediate commercial value. It requires:
- **Classification** — distinguishing spam from urgent production alerts (and spoofed phishing)
- **Composition** — drafting contextually appropriate, professional replies
- **Routing** — sending emails to the correct department (Engineering / Sales / HR)
- **PII Detection** — flagging emails containing sensitive personal data

The environment scales naturally from simple classification to complex multi-step reasoning, making it an ideal testbed for agent evaluation.

---

## 🔭 Observation Space (`EmailObservation`)

| Field | Type | Description |
|-------|------|-------------|
| `inbox` | `List[Email]` | All emails in the inbox for this episode |
| `current_email` | `Email` | The email currently being processed |
| `step_count` | `int` | Current step in the episode |
| `task_id` | `str` | Active task identifier |
| `context` | `dict` | Extra metadata (remaining count, task info) |

Each **Email** has:

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique email identifier (e.g., `email_001`) |
| `sender_name` | `str` | Display name of sender |
| `sender_email` | `str` | Email address (may be spoofed) |
| `subject` | `str` | Email subject line |
| `body` | `str` | Full email body text (3-8 sentences) |
| `department` | `str` | Department tag: Engineering / Sales / HR |
| `timestamp` | `str` | ISO 8601 timestamp |

---

## 🎯 Action Space (`EmailAction`)

| Field | Type | Valid Values | Description |
|-------|------|-------------|-------------|
| `action_type` | `str` | `"classify"`, `"reply"`, `"route"`, `"archive"`, `"flag"` | What to do |
| `label` | `str?` | `"urgent"`, `"normal"`, `"spam"` | Classification label |
| `reply_text` | `str?` | Free-form text | Draft reply content |
| `route_to` | `str?` | `"Engineering"`, `"Sales"`, `"HR"` | Department to route to |
| `reasoning` | `str?` | Free-form text | Agent's reasoning (logged, not graded) |

---

## 📋 Tasks

### Task 1: `classify_basic` (Easy)
- **Inbox**: 10 emails (3 urgent, 4 normal, 3 spam)
- **Goal**: Classify each email as urgent / normal / spam
- **Grading**: Accuracy = correct_labels / total_emails (0.0–1.0)
- **Max Steps**: 10
- **Partial Reward**: +0.1 per correct classification

### Task 2: `triage_and_reply` (Medium)
- **Inbox**: 5 emails (2 urgent, 2 normal, 1 spam)
- **Goal**: Classify all AND write short replies to the 2 urgent emails
- **Grading** (weighted):
  - Correct urgency detection: 0.3
  - Reply relevance (cosine similarity vs reference): 0.4
  - Tone appropriateness (keyword heuristic): 0.3
  - Penalty: -0.2 for empty replies on urgent emails
- **Max Steps**: 15

### Task 3: `full_triage_pipeline` (Hard)
- **Inbox**: 15 emails across Engineering, Sales, HR departments
- **Goal**: Classify all, reply to urgent, route each to correct department, flag PII
- **Grading** (weighted):
  - Classification accuracy: 0.25
  - Routing accuracy: 0.30
  - Reply quality (top-3 urgent): 0.25
  - PII flagging recall: 0.20
- **Max Steps**: 25

---

## 📐 Reward Function Design

### Dense Step Rewards
Every step returns a signal. Partial credit for partially correct actions.

| Condition | Reward |
|-----------|--------|
| Correct classification | +0.10 |
| Correct routing | +0.10 |
| Correct PII flag | +0.10 |
| Reply quality (similarity + tone) | up to +0.20 |
| Archive non-urgent | +0.03 |
| Episode completion bonus (< 60% max steps) | +0.10 |

### Penalties
| Condition | Penalty |
|-----------|---------|
| Repeated action on same email (loop) | -0.15 |
| Archive urgent email | -0.10 |
| Empty reply on urgent email | -0.20 |
| Reply under 10 characters | -0.05 |

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check → `{"status": "ok"}` |
| `/reset?task_id=...` | POST | Reset for new episode → `{observation}` |
| `/step` | POST | Submit action → `{observation, reward, done, info}` |
| `/state` | GET | Current state → `{state}` |
| `/tasks` | GET | List available tasks |

---

## ⚡ Setup

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn env.server:app --host 0.0.0.0 --port 7860

# Test endpoints
curl http://localhost:7860/health
curl -X POST "http://localhost:7860/reset?task_id=classify_basic"
```

### Docker

```bash
docker build -t email-triage-env .
docker run -p 7860:7860 email-triage-env

curl http://localhost:7860/health
```

### Run Baseline Agent

```bash
export HF_TOKEN=hf_...
python inference.py
```

### Run Tests

```bash
pytest tests/ -v
```

---

## 📊 Synthetic Data

Emails are generated deterministically using seeded random generation. Categories include:

- **Password reset / security alerts** (some from spoofed domains like `paypa1.com`, `c0mpany.com`)
- **Meeting requests and calendar changes**
- **Client complaints and contract renewals**
- **Spam** (lottery scams, phishing, crypto schemes)
- **HR policy updates and compliance**
- **Code review requests and infrastructure alerts**
- **PII-containing emails** (SSNs, phone numbers, personal emails, bank details)

All data generation uses `random.seed(task_seed)` for full reproducibility.

---

## License

MIT
