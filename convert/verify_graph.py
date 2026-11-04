#!/usr/bin/env python3
"""Check that the rewritten graph gives the same sentence embeddings as the original SentenceTransformer (FP32, CPU).

Run it BEFORE spending time in Acuity: if this is not ~1.0 the problem is the export, not the quantization.
Needs: torch, transformers, sentence-transformers.
"""
import argparse
import importlib.util
import os

import torch
from sentence_transformers import SentenceTransformer

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TEXTS = [
    "where is the project documentation",
    "onde fica a documentação do projeto",
    "The quick brown fox jumps over the lazy dog.",
    "bom dia, tudo bem?",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", type=int, default=128)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("export_onnx", os.path.join(here, "export_onnx.py"))
    exp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp)

    st = SentenceTransformer(MODEL, device="cpu")
    bert, tok = st[0].auto_model, st.tokenizer
    for layer in bert.encoder.layer:
        layer.intermediate.intermediate_act_fn = torch.nn.GELU(approximate="tanh")
    enc = exp.Encoder(bert).eval()
    emb = bert.embeddings

    def pool(h, m):
        m = m.unsqueeze(-1).float()
        return (h * m).sum(1) / m.sum(1).clamp(min=1)

    worst = 1.0
    for t in TEXTS:
        b = tok(t, padding="max_length", truncation=True, max_length=args.seq, return_tensors="pt")
        with torch.no_grad():
            ref = st.encode([t], convert_to_tensor=True, normalize_embeddings=True)
            x = (
                emb.word_embeddings(b["input_ids"])
                + emb.position_embeddings(torch.arange(args.seq)[None])
                + emb.token_type_embeddings(torch.zeros_like(b["input_ids"]))
            )
            mask = (1 - b["attention_mask"][:, None, None, :].float()) * -1e4
            out = pool(enc(x, mask), b["attention_mask"])
            out = torch.nn.functional.normalize(out, dim=-1)
        cos = float(torch.nn.functional.cosine_similarity(ref, out).item())
        worst = min(worst, cos)
        print(f"{cos:.5f}  {t}")
    print("worst cosine:", f"{worst:.5f}")


if __name__ == "__main__":
    main()
