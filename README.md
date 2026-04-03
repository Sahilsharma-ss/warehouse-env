---
title: Warehouse Order Orchestrator
emoji: 📦
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
short_description: OpenEnv warehouse benchmark with graded planning tasks.
tags:
  - openenv
  - reinforcement-learning
  - logistics
  - warehouse
pinned: false
---

# Warehouse Order Orchestrator

Warehouse Order Orchestrator is a real-world style OpenEnv environment where an agent plays the role of a warehouse operations coordinator. The job is to triage incoming customer orders, decide when to start processing, restock scarce items, and prioritize urgent work before deadlines are missed.

```mermaid
flowchart LR
	A[reset()] --> B[Observation: inventory, pending orders, shipments]
	B --> C[Agent chooses action]
	C --> D[step(action)]
	D --> E[Reward shaping + deadline updates]
	E --> F[Next observation or done]
```

## Why this environment

This is designed to be useful for agent evaluation in a realistic operations setting. It captures the core planning loop humans use in warehouses: choose the next order, handle stock shortages, keep urgent work moving, and avoid wasting time on invalid actions.

## OpenEnv Interface

The environment implements the standard `reset()`, `step(action)`, and `state()` pattern through `warehouse_env.environment:WarehouseEnv`.

### Observation space

Each observation contains:

| Field | Meaning |
| --- | --- |
| `task_id` / `task_name` / `difficulty` | Task metadata |
| `time` / `max_steps` | Current episode clock |
| `inventory` | Available stock by item |
| `active_order` | Current order in progress, if any |
| `pending_orders` | Orders still awaiting fulfillment |
| `incoming_shipments` | Restocks scheduled to arrive later in the episode |
| `metrics` | Running progress ratios for agents and graders |

### Action space

The agent can choose one of four actions:

| Action | Required fields | Purpose |
| --- | --- | --- |
| `start_order` | `order_id` | Reserve inventory and begin processing an order |
| `expedite_order` | `order_id` | Reduce the remaining work on the active order |
| `restock_item` | `item`, `quantity` | Schedule replenishment for a missing SKU |
| `wait` | none | Advance time without new work |

## Tasks

Three graded tasks are included, with increasing difficulty and tighter planning pressure.

| Task file | Difficulty | Focus |
| --- | --- | --- |
| `tasks/easy.json` | Easy | Straightforward triage with ample stock and forgiving deadlines |
| `tasks/medium.json` | Medium | One restock decision plus tighter order timing |
| `tasks/hard.json` | Hard | Scarce inventory, urgent orders, and competing deadline pressure |

Each task also includes grader weights so the score is not just raw reward. The grader combines completion, timeliness, priority handling, action efficiency, and shaped reward into a normalized score in the range `0.0` to `1.0`.

## Reward Design

Rewards are shaped across the full trajectory:

| Signal | Effect |
| --- | --- |
| Starting an order | Small positive signal for making progress |
| Active processing | Per-step progress reward |
| On-time completion | Completion bonus plus timeliness bonus |
| Late completion | Lower completion value plus lateness penalty |
| Restocking | Small cost, plus delayed inventory arrival |
| Invalid actions | Clear penalty |
| Waiting | Small cost to discourage loops |

## Setup

### Local install

```bash
pip install -r requirements.txt
```

### Run the baseline

```bash
python inference.py
```

The script uses the OpenAI client when these environment variables are set:

- `API_BASE_URL`
- `MODEL_NAME`
- `OPENAI_API_KEY` or `HF_TOKEN`

If those are missing, it falls back to a deterministic heuristic policy so the repo still runs locally.

### Container build

```bash
docker build -t openenv-warehouse .
docker run --rm openenv-warehouse
```

## Deployment

This project is structured for Hugging Face Spaces as a containerized OpenEnv submission. The expected deployment path is:

1. Push the repo to Hugging Face.
2. Use the provided `Dockerfile`.
3. Verify the Space responds and the environment can be initialized from the `openenv.yaml` entrypoint.

## Baseline Scores

Run `python inference.py` to reproduce the baseline on all three tasks. The script prints per-task `[START]`, `[STEP]`, and `[END]` logs plus the final average score.

| Task | Baseline score |
| --- | --- |
| Easy | 0.9576 |
| Medium | 0.9081 |
| Hard | 0.7730 |
| Average | 0.8796 |

## Repository Layout

| Path | Purpose |
| --- | --- |
| `warehouse_env/environment.py` | Core OpenEnv environment |
| `warehouse_env/models.py` | Typed Pydantic models |
| `warehouse_env/grader.py` | Task-aware grader |
| `tasks/` | Three graded task configs |
| `inference.py` | Root baseline script |
| `Dockerfile` | Container entrypoint for deployment |

## Validation Checklist

- `openenv.yaml` includes the environment metadata and task list.
- The environment exposes `reset()`, `step()`, and `state()`.
- Three deterministic task configs are included.
- The baseline script is rooted at `inference.py`.
- The Docker image starts from a clean container and launches the baseline script.

