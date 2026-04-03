from __future__ import annotations

import html
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

from inference import run_all_tasks
from warehouse_env.environment import WarehouseEnv
from warehouse_env.models import TaskConfig


ROOT = Path(__file__).resolve().parent
TASK_FILES = [ROOT / "tasks" / "easy.json", ROOT / "tasks" / "medium.json", ROOT / "tasks" / "hard.json"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def render_page(report: Dict[str, Any], generated_at: str, task_ids: list[str], error: str = "") -> bytes:
    safe_error = html.escape(error)
    safe_json = html.escape(json.dumps(report, indent=2, ensure_ascii=True))

    tasks_joined = ", ".join(task_ids)

    body = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Warehouse Env Space</title>
  <style>
    :root {{
      --bg-1: #f7f4ec;
      --bg-2: #e8efe2;
      --ink: #1f2937;
      --accent: #0f766e;
      --panel: #ffffff;
      --warn: #b91c1c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      font-family: ui-serif, Georgia, Cambria, "Times New Roman", Times, serif;
      color: var(--ink);
      background: radial-gradient(circle at 15% 20%, var(--bg-2), transparent 45%),
                  radial-gradient(circle at 80% 10%, #d6e9ef, transparent 40%),
                  var(--bg-1);
      display: grid;
      place-items: center;
      padding: 24px;
    }}
    .card {{
      width: min(920px, 100%);
      background: color-mix(in srgb, var(--panel) 92%, #f3f4f6);
      border: 1px solid #d1d5db;
      border-radius: 16px;
      box-shadow: 0 10px 24px rgba(0, 0, 0, 0.08);
      overflow: hidden;
    }}
    .head {{
      padding: 20px 24px;
      border-bottom: 1px solid #e5e7eb;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
    }}
    h1 {{ margin: 0; font-size: clamp(1.2rem, 3vw, 1.8rem); }}
    .meta {{ font-size: 0.95rem; color: #4b5563; }}
    .actions {{ display: flex; gap: 10px; flex-wrap: wrap; }}
    a.btn {{
      text-decoration: none;
      border: 1px solid #cbd5e1;
      padding: 8px 12px;
      border-radius: 10px;
      color: var(--ink);
      background: #f8fafc;
      font-size: 0.9rem;
    }}
    a.btn.primary {{
      border-color: #0f766e;
      background: var(--accent);
      color: #ecfeff;
    }}
    .content {{ padding: 18px 24px 24px; }}
        .api {{
            margin: 0 0 14px;
            padding: 10px 12px;
            border-radius: 10px;
            border: 1px solid #a7f3d0;
            background: #ecfdf5;
            color: #065f46;
            font-size: 0.92rem;
            line-height: 1.4;
        }}
    .error {{
      margin: 0 0 14px;
      padding: 10px 12px;
      border-radius: 10px;
      color: #7f1d1d;
      background: #fef2f2;
      border: 1px solid #fecaca;
      display: {"block" if error else "none"};
    }}
    pre {{
      margin: 0;
      background: #111827;
      color: #e5e7eb;
      border-radius: 12px;
      padding: 14px;
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.4;
    }}
  </style>
</head>
<body>
  <main class=\"card\">
    <section class=\"head\">
      <div>
        <h1>Warehouse Order Orchestrator</h1>
        <div class=\"meta\">Latest evaluation at {html.escape(generated_at)}</div>
      </div>
      <nav class=\"actions\">
        <a class=\"btn primary\" href=\"/run\">Run evaluation</a>
        <a class=\"btn\" href=\"/health\">Health check</a>
      </nav>
    </section>
    <section class=\"content\">
            <div class=\"api\">
                OpenEnv API ready: GET /health, GET or POST /reset, POST /step, GET or POST /state.<br/>
                Available tasks: {html.escape(tasks_joined)}
            </div>
      <p class=\"error\">{safe_error}</p>
      <pre>{safe_json}</pre>
    </section>
  </main>
</body>
</html>
"""
    return body.encode("utf-8")


class AppState:
    def __init__(self):
        self.task_configs = self._load_task_configs()
        self.env: Optional[WarehouseEnv] = None
        self.report: Dict[str, Any] = {
            "task_id": "all-tasks",
            "task_name": "Overall Baseline",
            "difficulty": "mixed",
            "score": None,
            "task_scores": [],
        }
        self.generated_at = utc_now_iso()
        self.last_error = ""
        self.last_transition: Dict[str, Any] = {
            "observation": {},
            "reward": 0.0,
            "done": False,
            "info": {},
        }

    @staticmethod
    def _load_task_configs() -> Dict[str, TaskConfig]:
        configs: Dict[str, TaskConfig] = {}
        for task_file in TASK_FILES:
            with task_file.open("r", encoding="utf-8") as handle:
                config = TaskConfig.model_validate(json.load(handle))
            configs[config.task_id] = config
        return configs

    def reset_env(self, task_id: str | None = None) -> Dict[str, Any]:
        selected_id = task_id
        if selected_id is None:
            selected_id = next(iter(self.task_configs))
        if selected_id not in self.task_configs:
            raise ValueError(f"Unknown task_id: {selected_id}")

        self.env = WarehouseEnv(self.task_configs[selected_id])
        observation = self.env.reset()
        self.last_transition = {
            "observation": observation,
            "reward": 0.0,
            "done": False,
            "info": {"task_id": selected_id},
        }
        return observation

    def step_env(self, action: Dict[str, Any]) -> Dict[str, Any]:
        if self.env is None:
            self.reset_env(None)

        observation, reward, done, info = self.env.step(action)
        self.last_transition = {
            "observation": observation,
            "reward": reward,
            "done": done,
            "info": info,
        }
        return self.last_transition

    def state_env(self) -> Dict[str, Any]:
        if self.env is None:
            self.reset_env(None)
        return self.env.state()

    def run(self):
        try:
            self.report = run_all_tasks()
            self.generated_at = utc_now_iso()
            self.last_error = ""
        except Exception as exc:  # pragma: no cover
            self.generated_at = utc_now_iso()
            self.last_error = f"Evaluation failed: {exc}"


STATE = AppState()


class Handler(BaseHTTPRequestHandler):
    def _send_html(self, body: bytes, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: Dict[str, Any], status: int = 200):
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}

        raw = self.rfile.read(length)
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON body: {exc}")

        if not isinstance(parsed, dict):
            raise ValueError("JSON body must be an object")
        return parsed

    def _send_error_json(self, error: str, status: int = 400):
        self._send_json({"error": error}, status=status)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(
                {
                    "status": "ok",
                    "generated_at": STATE.generated_at,
                    "tasks": list(STATE.task_configs.keys()),
                }
            )
            return

        if self.path == "/reset":
            try:
                observation = STATE.reset_env(None)
                self._send_json({"observation": observation, "done": False})
            except Exception as exc:
                self._send_error_json(str(exc), status=500)
            return

        if self.path == "/state":
            try:
                self._send_json({"state": STATE.state_env()})
            except Exception as exc:
                self._send_error_json(str(exc), status=500)
            return

        if self.path == "/run":
            STATE.run()
            self.send_response(302)
            self.send_header("Location", "/")
            self.end_headers()
            return

        body = render_page(STATE.report, STATE.generated_at, list(STATE.task_configs.keys()), STATE.last_error)
        self._send_html(body)

    def do_POST(self):
        if self.path == "/reset":
            try:
                payload = self._read_json_body()
                task_id = payload.get("task_id")
                observation = STATE.reset_env(task_id)
                self._send_json({"observation": observation, "done": False})
            except Exception as exc:
                self._send_error_json(str(exc), status=400)
            return

        if self.path == "/step":
            try:
                payload = self._read_json_body()
                action = payload.get("action")
                if not isinstance(action, dict):
                    self._send_error_json("Request must include object field 'action'", status=400)
                    return
                self._send_json(STATE.step_env(action))
            except Exception as exc:
                self._send_error_json(str(exc), status=400)
            return

        if self.path == "/state":
            try:
                self._send_json({"state": STATE.state_env()})
            except Exception as exc:
                self._send_error_json(str(exc), status=400)
            return

        self._send_error_json("Not found", status=404)

    def log_message(self, format: str, *args):
        return


if __name__ == "__main__":
    STATE.run()
    port = int(os.getenv("PORT", "7860"))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"[INFO] Serving Warehouse Env app on port {port}")
    server.serve_forever()
