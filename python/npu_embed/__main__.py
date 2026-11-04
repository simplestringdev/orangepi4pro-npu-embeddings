"""python -m npu_embed embed "text" ...   |   python -m npu_embed serve-http [--port 8089]"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .client import NpuEmbedder, NpuUnavailable
from .http_server import serve


def main() -> int:
    ap = argparse.ArgumentParser(prog="npu_embed")
    ap.add_argument("--model-dir", default=os.environ.get("NPU_EMBED_MODEL_DIR"), help="dir with tokenizer.json + model.safetensors")
    ap.add_argument("--socket", default=os.environ.get("NPU_EMBED_SOCKET"), help="npu_embed_server socket")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("embed", help="print embeddings as JSON")
    e.add_argument("texts", nargs="+")
    h = sub.add_parser("serve-http", help="OpenAI-compatible /v1/embeddings on localhost")
    h.add_argument("--host", default="127.0.0.1")
    h.add_argument("--port", type=int, default=8089)
    a = ap.parse_args()
    if not a.model_dir:
        ap.error("--model-dir (or NPU_EMBED_MODEL_DIR) is required")
    emb = NpuEmbedder(a.model_dir, a.socket)
    try:
        if a.cmd == "embed":
            print(json.dumps(emb.encode(a.texts)))
        else:
            print(f"listening on http://{a.host}:{a.port}/v1/embeddings", file=sys.stderr)
            serve(emb, a.host, a.port)
    except NpuUnavailable as ex:
        print(f"error: {ex}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
