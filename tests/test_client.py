"""Protocol, pre/post-processing and error paths of NpuEmbedder. No NPU, torch or model download needed:
a fake server speaks the protocol of server/npu_embed_server.c and the tokenizer/tables are stubbed."""
from __future__ import annotations

import math
import shutil
import socket
import tempfile
import threading
from pathlib import Path

import numpy as np
import pytest

from npu_embed import NpuEmbedder, NpuUnavailable
from npu_embed.client import HEADER, INT16, MAGIC, MASK_NEG

SEQ, H = 128, 384
FP_IN, FP_OUT = 15, 12


class FakeServer(threading.Thread):
    def __init__(self, path: str, *, magic: int = MAGIC, value: float = 0.5):
        super().__init__(daemon=True)
        self.magic, self.value, self.frames = magic, value, []
        self.srv = socket.socket(socket.AF_UNIX)
        self.srv.bind(path)
        self.srv.listen(4)

    def run(self):
        while True:
            try:
                c, _ = self.srv.accept()
            except OSError:
                return
            i0, i1, o = SEQ * H * 2, SEQ * 2, SEQ * H * 2
            c.sendall(HEADER.pack(self.magic, i0, i1, o, INT16, FP_OUT, INT16, FP_IN))
            while True:
                buf = b""
                while len(buf) < i0 + i1:
                    chunk = c.recv(i0 + i1 - len(buf))
                    if not chunk:
                        break
                    buf += chunk
                if len(buf) < i0 + i1:
                    break
                self.frames.append(buf)
                h = np.zeros((SEQ, H), dtype=np.float32)
                h[:, 0] = self.value
                c.sendall(np.rint(h * 2**FP_OUT).astype("<i2").tobytes())
            c.close()


@pytest.fixture
def sockdir():
    d = tempfile.mkdtemp(dir="/tmp", prefix="npu-test-")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def make(sock: str) -> NpuEmbedder:
    e = NpuEmbedder("/nonexistent", Path(sock), timeout=5)
    e._loaded = True
    words = [len(t.split()) for t in ("a b c",)]  # placeholder to keep flake quiet
    del words

    class Tok:
        def encode(self, text):
            n = min(len(text.split()), SEQ)
            return type("E", (), {"ids": list(range(SEQ)), "attention_mask": [1] * n + [0] * (SEQ - n)})()

    e._tok = Tok()
    e._word = np.full((SEQ, H), 0.1, dtype=np.float32)
    e._pos = np.zeros((SEQ, H), dtype=np.float32)
    e._type0 = np.zeros(H, dtype=np.float32)
    return e


def test_roundtrip_is_normalized_and_pooled(sockdir):
    FakeServer(f"{sockdir}/s").start()
    v = make(f"{sockdir}/s").encode(["one two three"])[0]
    assert len(v) == 384
    assert math.isclose(math.sqrt(sum(x * x for x in v)), 1.0, rel_tol=1e-4)
    assert v[0] == pytest.approx(1.0, abs=1e-4)


def test_frame_is_q15_int16_with_additive_mask(sockdir):
    srv = FakeServer(f"{sockdir}/s")
    srv.start()
    make(f"{sockdir}/s").encode(["a b c"])
    frame = srv.frames[0]
    emb = np.frombuffer(frame[: SEQ * H * 2], dtype="<i2")
    assert emb[0] == round(0.1 * 2**FP_IN)
    mask = np.frombuffer(frame[SEQ * H * 2 :], dtype="<f2").astype(np.float32)
    assert (mask[:3] == 0).all() and (mask[3:] == MASK_NEG).all()


def test_many_texts_one_connection(sockdir):
    srv = FakeServer(f"{sockdir}/s")
    srv.start()
    out = make(f"{sockdir}/s").encode(["a", "b c", "d e f"])
    assert len(out) == 3 and len(srv.frames) == 3


def test_empty_input_returns_empty():
    assert NpuEmbedder("/nonexistent").encode([]) == []


def test_server_down_raises(sockdir):
    with pytest.raises(NpuUnavailable):
        make(f"{sockdir}/missing").encode(["x"])


def test_bad_magic_raises(sockdir):
    FakeServer(f"{sockdir}/s", magic=0xDEADBEEF).start()
    with pytest.raises(NpuUnavailable):
        make(f"{sockdir}/s").encode(["x"])


def test_missing_model_files_raise(sockdir):
    with pytest.raises(NpuUnavailable):
        NpuEmbedder(sockdir, Path(f"{sockdir}/s")).encode(["x"])


def test_header_is_eight_words():
    assert HEADER.size == 32
