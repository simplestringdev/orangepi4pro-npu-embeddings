"""Sentence embeddings from the NPU server (npu_embed_server) for paraphrase-multilingual-MiniLM-L12-v2.

The encoder runs on the NPU. Everything cheap stays on the CPU: tokenize, look up and sum the word/position/type
embeddings, mean pooling, L2 normalization. No torch needed (numpy + tokenizers only).
"""
from __future__ import annotations

import json
import os
import socket
import struct
import threading
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

MAGIC = 0x4555504E  # 'NPUE'
HEADER = struct.Struct("<IIIIIiIi")  # magic, bytes_in0, bytes_in1, bytes_out, fmt_out, fp_out, fmt_in0, fp_in0
FP32, FP16, UINT8, INT8, INT16 = 0, 1, 2, 3, 5  # VIP_BUFFER_FORMAT_*
MASK_NEG = -1e4  # additive attention mask for padded positions; fits in float16
HIDDEN = 384


class NpuUnavailable(RuntimeError):
    """Server not running, model files missing, or a reply that breaks the protocol."""


def default_socket() -> Path:
    run = os.environ.get("XDG_RUNTIME_DIR")
    return Path(run) / "npu-embed.sock" if run else Path(f"/tmp/npu-embed-{os.getuid()}.sock")


class NpuEmbedder:
    """`model_dir` = a directory with tokenizer.json and model.safetensors of the MiniLM checkpoint."""

    dim = HIDDEN

    def __init__(self, model_dir: str | Path, socket_path: str | Path | None = None, *, seq: int = 128, timeout: float = 30.0):
        self.model_dir = Path(model_dir)
        self.socket_path = Path(socket_path) if socket_path else default_socket()
        self.seq = seq
        self.timeout = timeout
        self._lock = threading.Lock()  # one NPU: one conversation at a time per embedder
        self._loaded = False

    # -- lazy load: tokenizer + embedding tables ------------------------------------------------------------
    def _load(self) -> None:
        if self._loaded:
            return
        tok_path, st_path = self.model_dir / "tokenizer.json", self.model_dir / "model.safetensors"
        if not tok_path.is_file() or not st_path.is_file():
            raise NpuUnavailable(f"tokenizer.json / model.safetensors not found in {self.model_dir}")
        tok = Tokenizer.from_file(str(tok_path))
        pad = tok.token_to_id("<pad>")
        tok.enable_truncation(max_length=self.seq)
        tok.enable_padding(length=self.seq, pad_id=pad if pad is not None else 0, pad_token="<pad>")
        self._tok = tok

        # safetensors = 8-byte header length + JSON header + raw data. The word table (250k rows) becomes a
        # memmap, so the kernel pages in only the rows actually used instead of copying ~384 MB.
        with open(st_path, "rb") as f:
            (hlen,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(hlen))
        base = 8 + hlen

        def table(suffix: str) -> np.ndarray:
            for name, meta in header.items():
                if name.endswith(suffix) and isinstance(meta, dict):
                    dtype = {"F32": "<f4", "F16": "<f2"}.get(meta["dtype"])
                    if dtype is None:
                        raise NpuUnavailable(f"unsupported dtype {meta['dtype']} for {name}")
                    return np.memmap(st_path, dtype=dtype, mode="r", offset=base + meta["data_offsets"][0], shape=tuple(meta["shape"]))
            raise NpuUnavailable(f"tensor ending in {suffix} not found in model.safetensors")

        self._word = table("embeddings.word_embeddings.weight")
        self._pos = np.array(table("embeddings.position_embeddings.weight")[: self.seq], dtype=np.float32)
        self._type0 = np.array(table("embeddings.token_type_embeddings.weight")[0], dtype=np.float32)
        self._loaded = True

    # -- one request ----------------------------------------------------------------------------------------
    def _frame(self, text: str, fmt_in0: int, fp_in0: int) -> tuple[bytes, np.ndarray]:
        enc = self._tok.encode(text)
        x = np.asarray(self._word[np.asarray(enc.ids)], dtype=np.float32) + self._pos + self._type0
        if fmt_in0 == INT16:
            a = np.clip(np.rint(x * float(2**fp_in0)), -32768, 32767).astype("<i2")
        elif fmt_in0 == FP16:
            a = x.astype("<f2")
        elif fmt_in0 == FP32:
            a = x.astype("<f4")
        else:
            raise NpuUnavailable(f"unsupported input format {fmt_in0}")
        attn = np.asarray(enc.attention_mask, dtype=np.float32)
        mask = np.where(attn == 1, 0.0, MASK_NEG).astype("<f2")
        return a.tobytes() + mask.tobytes(), attn

    def _pool(self, raw: bytes, fmt_out: int, fp_out: int, attn: np.ndarray) -> list[float]:
        if fmt_out == INT16:
            h = np.frombuffer(raw, dtype="<i2").astype(np.float32) / float(2**fp_out)
        elif fmt_out == FP16:
            h = np.frombuffer(raw, dtype="<f2").astype(np.float32)
        elif fmt_out == FP32:
            h = np.frombuffer(raw, dtype="<f4").copy()
        else:
            raise NpuUnavailable(f"unsupported output format {fmt_out}")
        h = h.reshape(self.seq, -1)
        v = (h * attn[:, None]).sum(axis=0) / max(float(attn.sum()), 1.0)
        n = float(np.linalg.norm(v)) or 1.0
        return (v / n).tolist()

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Unit-length embeddings, one per text. Raises NpuUnavailable if the NPU path cannot be used."""
        if not texts:
            return []
        with self._lock:
            self._load()
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(self.timeout)
            try:
                try:
                    s.connect(str(self.socket_path))
                except OSError as e:
                    raise NpuUnavailable(f"NPU server not reachable at {self.socket_path}: {e}") from e

                def read(n: int) -> bytes:
                    buf = bytearray()
                    while len(buf) < n:
                        chunk = s.recv(n - len(buf))
                        if not chunk:
                            raise NpuUnavailable("NPU server closed the connection")
                        buf += chunk
                    return bytes(buf)

                magic, b0, b1, b_out, fmt_out, fp_out, fmt_in0, fp_in0 = HEADER.unpack(read(HEADER.size))
                if magic != MAGIC:
                    raise NpuUnavailable("bad protocol header")
                out = []
                for text in texts:
                    frame, attn = self._frame(text, fmt_in0, fp_in0)
                    if len(frame) != b0 + b1:
                        raise NpuUnavailable(f"input size {len(frame)} != expected {b0 + b1} (is --seq right?)")
                    s.sendall(frame)
                    out.append(self._pool(read(b_out), fmt_out, fp_out, attn))
                return out
            except (socket.timeout, OSError) as e:
                raise NpuUnavailable(f"NPU communication failed: {e}") from e
            finally:
                s.close()
