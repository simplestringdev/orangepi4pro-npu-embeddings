"""OpenAI-compatible endpoint, with a fake embedder."""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from npu_embed.client import NpuUnavailable
from npu_embed.http_server import make_handler


class Fake:
    def encode(self, texts):
        return [[float(len(t)), 0.0] for t in texts]


class Down:
    def encode(self, texts):
        raise NpuUnavailable("server down")


def start(embedder):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(embedder))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def test_string_and_list_inputs():
    srv, url = start(Fake())
    s, body = post(f"{url}/v1/embeddings", {"input": "abc"})
    assert s == 200 and body["data"][0]["embedding"] == [3.0, 0.0]
    s, body = post(f"{url}/v1/embeddings", {"input": ["a", "bb"], "model": "x"})
    assert [d["index"] for d in body["data"]] == [0, 1] and body["model"] == "x"
    srv.shutdown()


@pytest.mark.parametrize("bad", [{}, {"input": []}, {"input": [1]}, {"input": 5}])
def test_bad_input_is_400(bad):
    srv, url = start(Fake())
    assert post(f"{url}/v1/embeddings", bad)[0] == 400
    srv.shutdown()


def test_npu_down_is_503_and_server_keeps_running():
    srv, url = start(Down())
    assert post(f"{url}/v1/embeddings", {"input": "x"})[0] == 503
    assert post(f"{url}/v1/embeddings", {"input": "x"})[0] == 503
    srv.shutdown()


def test_models_and_404():
    srv, url = start(Fake())
    with urllib.request.urlopen(f"{url}/v1/models", timeout=5) as r:
        assert json.load(r)["data"][0]["object"] == "model"
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen(f"{url}/nope", timeout=5)
    srv.shutdown()
