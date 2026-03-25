"""FastAPI server for the Email Triage Agent Environment."""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from env import EmailTriageEnv
from env.models import Action, GraderRequest, GraderResponse, StepResult
from env.tasks import TASK_DEFINITIONS, grade_episode

app = FastAPI(
    title="Email Triage Agent Environment",
    description="An OpenEnv-compliant environment where agents triage, classify, and respond to realistic email inboxes.",
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


# ─── Endpoints ───────────────────────────────────────────────────────


@app.get("/")
def root():
    """Health check / landing page."""
    return {
        "name": "email-triage-env",
        "version": "1.0.0",
        "description": "An environment where agents triage, classify, and respond to realistic email inboxes.",
        "endpoints": ["/tasks", "/reset", "/step", "/state", "/grader"],
    }


@app.get("/tasks")
def list_tasks():
    """List all available tasks with their schemas."""
    tasks = []
    for task_id, task_def in TASK_DEFINITIONS.items():
        tasks.append({
            "id": task_id,
            "name": task_def["name"],
            "difficulty": task_def["difficulty"],
            "description": task_def["description"],
            "max_steps": task_def["max_steps"],
            "action_schema": Action.model_json_schema(),
        })
    return {"tasks": tasks}


@app.post("/reset")
def reset(task_id: str = Query(..., description="Task ID: task_easy, task_medium, or task_hard")):
    """Reset the environment for a new episode with the given task."""
    try:
        observation = env.reset(task_id)
        return observation.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step(action: Action):
    """Submit an action and receive the next observation + reward."""
    if env.current_task_id is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first.",
        )
    result = env.step(action)
    return result.model_dump()


@app.get("/state")
def get_state():
    """Get the current environment state."""
    if env.current_task_id is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first.",
        )
    return env.state()


@app.post("/grader")
def grader(request: GraderRequest):
    """
    Grade a complete episode log for a given task.
    Returns a score in [0.0, 1.0] and detailed breakdown.
    """
    try:
        episode_dicts = [a.model_dump() for a in request.episode_log]
        score, details = grade_episode(request.task_id, episode_dicts)
        return GraderResponse(score=score, details=details).model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/baseline")
def baseline_info():
    """
    Return information about running the baseline agent.
    The actual baseline is run via baseline.py script.
    """
    return {
        "description": "Run the baseline agent using: python baseline.py",
        "model": "gpt-4o-mini",
        "tasks": list(TASK_DEFINITIONS.keys()),
        "note": "Set OPENAI_API_KEY environment variable before running.",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
