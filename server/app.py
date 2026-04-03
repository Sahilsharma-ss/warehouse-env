from __future__ import annotations

import os
import sys
from pathlib import Path


# Ensure repository root is importable when launched via script entrypoint.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import Handler, STATE  # noqa: E402
from http.server import ThreadingHTTPServer  # noqa: E402


def main() -> None:
    STATE.run()
    port = int(os.getenv("PORT", "7860"))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"[INFO] Serving Warehouse Env app on port {port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
