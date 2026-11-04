#!/usr/bin/env python3
"""Build the calibration set for quantization: N pairs of (emb_XXX.npy, mask_XXX.npy) from real text.

int16 (dynamic fixed point) quantization picks its scale from these tensors, so use text that looks like what
you will embed. The repo ships `calibration_texts.txt` (generic multilingual paragraphs); pass your own with --texts.
A paragraph = block of lines separated by a blank line.

Needs: torch, transformers.     Output: <out-dir>/emb_000.npy, mask_000.npy, ... and <out-dir>/dataset.txt
"""
import argparse
import os
import re

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--texts", default=os.path.join(here, "calibration_texts.txt"))
    ap.add_argument("--out-dir", default="build/calib")
    ap.add_argument("-n", type=int, default=64)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", open(args.texts, encoding="utf-8").read())]
    paras = [p for p in paras if len(p) > 30][: args.n]
    os.makedirs(args.out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    emb = AutoModel.from_pretrained(args.model).embeddings
    lines = []
    for i, text in enumerate(paras):
        b = tok(text, padding="max_length", truncation=True, max_length=args.seq, return_tensors="pt")
        with torch.no_grad():
            x = (
                emb.word_embeddings(b["input_ids"])
                + emb.position_embeddings(torch.arange(args.seq)[None])
                + emb.token_type_embeddings(torch.zeros_like(b["input_ids"]))
            )
            m = (1 - b["attention_mask"][:, None, None, :].float()) * -1e4
        np.save(os.path.join(args.out_dir, f"emb_{i:03d}.npy"), x.numpy().astype(np.float32))
        np.save(os.path.join(args.out_dir, f"mask_{i:03d}.npy"), m.numpy().astype(np.float32))
        lines.append(f"calib/emb_{i:03d}.npy calib/mask_{i:03d}.npy\n")
    with open(os.path.join(os.path.dirname(args.out_dir.rstrip("/")) or ".", "dataset.txt"), "w") as f:
        f.writelines(lines)
    print(f"{len(paras)} calibration pairs in {args.out_dir} (+ dataset.txt next to it)")


if __name__ == "__main__":
    main()
