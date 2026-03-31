"""FastAPI server for the Email Triage Agent Environment.

Endpoints:
  POST /reset  → { observation }
  POST /step   → { observation, reward, done, info }
  GET  /state  → { state }
  GET  /health → { status: "ok" }
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from env.models import (
    EmailAction,
    ResetResponse,
    StepResponse,
    StateResponse,
)
from env.environment import EmailTriageEnv
from env.tasks import TASK_REGISTRY, get_task_ids

app = FastAPI(
    title="Email Triage Agent Environment",
    description=(
        "An OpenEnv-compliant environment where agents triage, classify, "
        "reply to, route, and flag realistic email inboxes."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance
env = EmailTriageEnv()


# ─── Health ──────────────────────────────────────────────────────────


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/")
def root():
    """Landing page with environment info."""
    return {
        "name": "email-triage-env",
        "version": "1.0.0",
        "status": "ok",
        "description": "AI email triage: classify, reply, route, and flag emails",
        "endpoints": ["/health", "/reset", "/step", "/state", "/tasks"],
        "tasks": get_task_ids(),
    }


# ─── Core OpenEnv Endpoints ─────────────────────────────────────────


@app.post("/reset")
def reset(
    task_id: str = Query(
        "classify_basic",
        description="Task ID: classify_basic, triage_and_reply, or full_triage_pipeline",
    ),
):
    """
    Reset the environment for a new episode with the given task.

    Returns: { observation: EmailObservation }
    """
    try:
        observation = env.reset(task_id)
        return {"observation": observation.model_dump()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step(action: EmailAction):
    """
    Submit an action and receive the next observation + reward.

    Returns: { observation, reward: float, done: bool, info: dict }
    """
    if env.current_task_id is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call POST /reset first.",
        )
    try:
        result = env.step(action)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@app.get("/state")
def get_state():
    """
    Get the current environment state.

    Returns: { state: dict }
    """
    if env.current_task_id is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call POST /reset first.",
        )
    return {"state": env.state()}


# ─── Task Listing ───────────────────────────────────────────────────


@app.get("/tasks")
def list_tasks():
    """List all available tasks with their configuration."""
    tasks = []
    for task_id, config in TASK_REGISTRY.items():
        tasks.append(
            {
                "id": task_id,
                "name": config["name"],
                "difficulty": config["difficulty"],
                "description": config["description"],
                "max_steps": config["max_steps"],
            }
        )
    return {"tasks": tasks}


def run():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    run()
