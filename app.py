from __future__ import annotations

import html
import json
import os
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict

from inference import run_all_tasks


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def render_page(report: Dict[str, Any], generated_at: str, error: str = "") -> bytes:
    safe_error = html.escape(error)
    safe_json = html.escape(json.dumps(report, indent=2, ensure_ascii=True))

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
        self.report: Dict[str, Any] = {
            "task_id": "all-tasks",
            "task_name": "Overall Baseline",
            "difficulty": "mixed",
            "score": None,
            "task_scores": [],
        }
        self.generated_at = utc_now_iso()
        self.last_error = ""

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

    def do_GET(self):
        if self.path == "/health":
            self._send_json({"status": "ok", "generated_at": STATE.generated_at})
            return

        if self.path == "/run":
            STATE.run()
            self.send_response(302)
            self.send_header("Location", "/")
            self.end_headers()
            return

        body = render_page(STATE.report, STATE.generated_at, STATE.last_error)
        self._send_html(body)

    def log_message(self, format: str, *args):
        return


if __name__ == "__main__":
    STATE.run()
    port = int(os.getenv("PORT", "7860"))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"[INFO] Serving Warehouse Env app on port {port}")
    server.serve_forever()
