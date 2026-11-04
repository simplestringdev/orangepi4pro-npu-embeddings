"""Minimal OpenAI-compatible embeddings endpoint on top of NpuEmbedder (standard library only).

  POST /v1/embeddings   {"input": "text" | ["a", "b"], "model": "anything"}
  GET  /v1/models

Binds to 127.0.0.1 by default. There is no authentication: do not expose it beyond localhost without a proxy.
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_ID = "paraphrase-multilingual-MiniLM-L12-v2"
MAX_BODY = 1 << 20  # 1 MiB
MAX_INPUTS = 256


def make_handler(embedder):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict) -> None:
            raw = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args) -> None:  # quiet
            pass

        def do_GET(self) -> None:
            if self.path.rstrip("/") == "/v1/models":
                self._send(200, {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]})
            else:
                self._send(404, {"error": {"message": "not found"}})

        def do_POST(self) -> None:
            if self.path.rstrip("/") != "/v1/embeddings":
                return self._send(404, {"error": {"message": "not found"}})
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if n <= 0 or n > MAX_BODY:
                    return self._send(413, {"error": {"message": "body missing or too large"}})
                req = json.loads(self.rfile.read(n))
                inp = req.get("input")
                texts = [inp] if isinstance(inp, str) else inp
                if not isinstance(texts, list) or not texts or len(texts) > MAX_INPUTS or not all(isinstance(t, str) for t in texts):
                    return self._send(400, {"error": {"message": f"'input' must be a string or a list of 1..{MAX_INPUTS} strings"}})
                vectors = embedder.encode(texts)
            except json.JSONDecodeError:
                return self._send(400, {"error": {"message": "invalid JSON"}})
            except Exception as e:  # NpuUnavailable and anything else: report, keep serving
                return self._send(503, {"error": {"message": str(e)}})
            self._send(
                200,
                {
                    "object": "list",
                    "model": req.get("model") or MODEL_ID,
                    "data": [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vectors)],
                    "usage": {"prompt_tokens": 0, "total_tokens": 0},
                },
            )

    return Handler


def serve(embedder, host: str = "127.0.0.1", port: int = 8089) -> None:
    ThreadingHTTPServer((host, port), make_handler(embedder)).serve_forever()
