from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - local fallback without network deps
    OpenAI = None

from warehouse_env.environment import WarehouseEnv
from warehouse_env.grader import Grader
from warehouse_env.models import TaskConfig


ROOT = Path(__file__).resolve().parent
TASK_FILES = [
    ROOT / "tasks" / "easy.json",
    ROOT / "tasks" / "medium.json",
    ROOT / "tasks" / "hard.json",
    ROOT / "tasks" / "peak_season.json",
]

# Submission checklist alignment:
# - defaults only for API_BASE_URL and MODEL_NAME
# - HF_TOKEN has no default
API_BASE_URL = os.getenv("API_BASE_URL", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")


def _safe_score(value: float) -> float:
    """Final output safety-net: always emit task scores inside (0, 1)."""
    return Grader._strict_unit_interval(float(value))


class HeuristicPolicy:
    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        active_order = observation.get("active_order")
        pending_orders = observation.get("pending_orders", [])
        inventory = observation.get("inventory", {})

        if active_order:
            remaining_time = active_order.get("remaining_time", 0)
            deadline = active_order.get("deadline", observation.get("time", 0))
            priority = active_order.get("priority", 0)
            if remaining_time > 1 and priority >= 3 and deadline - observation.get("time", 0) <= remaining_time:
                return {"type": "expedite_order", "order_id": active_order["id"]}
            return {"type": "wait"}

        best_order = None
        best_score = float("-inf")
        for order in pending_orders:
            if order.get("status") == "fulfilled":
                continue

            items = order.get("items", {})
            feasible = all(inventory.get(item, 0) >= quantity for item, quantity in items.items())
            urgency = max(0, order.get("deadline", 0) - observation.get("time", 0))
            score = order.get("priority", 0) * 3 - urgency - order.get("processing_time", 1)

            if feasible and score > best_score:
                best_score = score
                best_order = order

        if best_order is not None:
            return {"type": "start_order", "order_id": best_order["id"]}

        shortage_candidate = None
        shortage_amount = 0
        for order in pending_orders:
            for item, quantity in order.get("items", {}).items():
                shortage = max(0, quantity - inventory.get(item, 0))
                if shortage > shortage_amount:
                    shortage_amount = shortage
                    shortage_candidate = (item, shortage)

        if shortage_candidate is not None:
            item, quantity = shortage_candidate
            return {"type": "restock_item", "item": item, "quantity": max(1, quantity)}

        return {"type": "wait"}


class OpenAIPlanner:
    def __init__(self):
        api_key = HF_TOKEN
        base_url = API_BASE_URL
        self.model_name = MODEL_NAME
        self.client = None
        self.heuristic = HeuristicPolicy()

        if OpenAI is not None and api_key and base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        if self.client is None:
            return self.heuristic.act(observation)

        system_prompt = (
            "You are a warehouse operations agent. Choose one action in strict JSON. "
            "Prefer fulfilling urgent feasible orders, restocking only when necessary, and avoiding invalid actions."
        )

        user_prompt = json.dumps(
            {
                "observation": observation,
                "allowed_actions": [
                    {"type": "start_order", "order_id": 1},
                    {"type": "expedite_order", "order_id": 1},
                    {"type": "restock_item", "item": "A", "quantity": 1},
                    {"type": "wait"},
                ],
                "output_schema": {
                    "type": "string",
                    "order_id": "optional integer",
                    "item": "optional string",
                    "quantity": "optional integer",
                },
            },
            separators=(",", ":"),
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content or "{}"
            action = json.loads(content)
            if not isinstance(action, dict):
                raise ValueError("Model did not return a JSON object")
            return action
        except Exception:
            return self.heuristic.act(observation)


def load_task(task_path: Path) -> TaskConfig:
    with task_path.open("r", encoding="utf-8") as handle:
        return TaskConfig.model_validate(json.load(handle))


def print_log(prefix: str, payload: Dict[str, Any]):
    print(f"{prefix} {json.dumps(payload, separators=(',', ':'), ensure_ascii=True)}")


def run_task(task_path: Path, planner: OpenAIPlanner, grader: Grader) -> float:
    config = load_task(task_path)
    env = WarehouseEnv(config)
    observation = env.reset()
    total_reward = 0.0
    done = False
    step_index = 0

    print_log(
        "[START]",
        {
            "task_id": config.task_id,
            "task_name": config.name,
            "difficulty": config.difficulty,
            "model": planner.model_name,
            "max_steps": config.max_steps,
            "api_base_url": API_BASE_URL,
            "mode": "openai" if planner.client is not None else "heuristic",
        },
    )

    while not done and step_index < config.max_steps:
        action = planner.act(observation)
        observation, reward, done, info = env.step(action)
        total_reward += reward
        step_index += 1

        print_log(
            "[STEP]",
            {
                "task_id": config.task_id,
                "step": step_index,
                "action": action,
                "reward": round(reward, 6),
                "total_reward": round(total_reward, 6),
                "done": done,
                "completion_ratio": round(info["summary"]["completion_ratio"], 4),
            },
        )

    score = _safe_score(grader.score(env, total_reward))
    summary = env.summary()

    print_log(
        "[END]",
        {
            "task_id": config.task_id,
            "task_name": config.name,
            "difficulty": config.difficulty,
            "steps": step_index,
            "score": score,
            "total_reward": round(total_reward, 6),
            "completed_orders": summary["completed_orders"],
            "fulfilled_on_time": summary["fulfilled_on_time"],
            "fulfilled_late": summary["fulfilled_late"],
        },
    )

    return score


def main():
    run_all_tasks()


def run_all_tasks() -> Dict[str, Any]:
    planner = OpenAIPlanner()
    grader = Grader()

    scores: List[float] = []
    for task_path in TASK_FILES:
        scores.append(_safe_score(run_task(task_path, planner, grader)))

    average_score = _safe_score(sum(scores) / max(1, len(scores)))
    print_log(
        "[END]",
        {
            "task_id": "all-tasks",
            "task_name": "Overall Baseline",
            "difficulty": "mixed",
            "score": average_score,
            "task_scores": scores,
        },
    )

    return {
        "task_id": "all-tasks",
        "task_name": "Overall Baseline",
        "difficulty": "mixed",
        "score": average_score,
        "task_scores": scores,
    }


if __name__ == "__main__":
    main()