#!/usr/bin/env bash
# Download the two files of the Hugging Face checkpoint that the CPU side needs (tokenizer + embedding tables).
# usage: scripts/fetch_model.sh [dir]     (default: ~/.cache/npu-embed/minilm)
set -euo pipefail
dir=${1:-$HOME/.cache/npu-embed/minilm}
base=https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/resolve/main
mkdir -p "$dir"
for f in tokenizer.json model.safetensors; do
  [[ -s $dir/$f ]] || curl -fL --retry 3 -o "$dir/$f" "$base/$f"
done
ls -la "$dir"
echo "model dir: $dir"
