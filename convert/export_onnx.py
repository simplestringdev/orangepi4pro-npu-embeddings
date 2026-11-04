#!/usr/bin/env python3
"""Export the MiniLM encoder as an NPU-friendly ONNX graph (fixed length, additive mask, tanh-GELU).

Why not export the Hugging Face model as is: its graph has IsNaN / Where / ConstantOfShape / Erf, which the
Acuity compiler or the VIP9000 either rejects or runs slowly. This wrapper re-implements the 12 encoder layers
with only Add / Mul / MatMul / Softmax / Tanh / LayerNorm (decomposed) / Reshape / Transpose.

The word-embedding table (250k rows, ~96M parameters) is NOT in the graph: the CPU looks rows up and feeds the
sum (word + position + token type) to the NPU. The graph starts at the embedding LayerNorm.

Needs: torch, transformers, onnx.    Output: <out-dir>/minilm_npu_<seq>.onnx
"""
import argparse
import collections
import os

import onnx
import torch
from transformers import AutoModel

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class Encoder(torch.nn.Module):
    """Inputs: emb [1, SEQ, H] (word+position+type already summed) and add_mask [1, 1, 1, SEQ] (0 or -1e4)."""

    def __init__(self, model):
        super().__init__()
        self.layers = model.encoder.layer
        self.ln = model.embeddings.LayerNorm

    def forward(self, emb, add_mask):
        h = self.ln(emb)
        for layer in self.layers:
            a = layer.attention.self
            b, t, _ = h.shape
            nh, dh = a.num_attention_heads, a.attention_head_size
            q = a.query(h).view(b, t, nh, dh).transpose(1, 2)
            k = a.key(h).view(b, t, nh, dh).transpose(1, 2)
            v = a.value(h).view(b, t, nh, dh).transpose(1, 2)
            scores = torch.matmul(q, k.transpose(-1, -2)) / (dh**0.5) + add_mask
            ctx = torch.matmul(torch.softmax(scores, dim=-1), v).transpose(1, 2).reshape(b, t, nh * dh)
            h = layer.attention.output.LayerNorm(layer.attention.output.dense(ctx) + h)
            h = layer.output.LayerNorm(layer.output.dense(layer.intermediate(h)) + h)
        return h


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="build")
    ap.add_argument("--seq", type=int, default=128, help="fixed sequence length (MiniLM max_seq_length is 128)")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model = AutoModel.from_pretrained(args.model, attn_implementation="eager").eval()
    for layer in model.encoder.layer:  # exact GELU uses Erf; the tanh approximation uses only Tanh/Mul/Add
        layer.intermediate.intermediate_act_fn = torch.nn.GELU(approximate="tanh")

    enc = Encoder(model).eval()
    emb = torch.randn(1, args.seq, model.config.hidden_size)
    mask = torch.zeros(1, 1, 1, args.seq)
    out = os.path.join(args.out_dir, f"minilm_npu_{args.seq}.onnx")
    torch.onnx.export(
        enc, (emb, mask), out, input_names=["emb", "add_mask"], output_names=["hidden"], opset_version=13, dynamo=False
    )
    ops = collections.Counter(n.op_type for n in onnx.load(out).graph.node)
    print("wrote", out, f"({os.path.getsize(out) / 1e6:.1f} MB)")
    print("operators:", dict(sorted(ops.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
