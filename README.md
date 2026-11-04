# Sentence embeddings on the Orange Pi 4 Pro NPU

Run the multilingual MiniLM sentence-embedding model (`paraphrase-multilingual-MiniLM-L12-v2`, 384 dimensions,
50+ languages) on the **NPU of the Orange Pi 4 Pro** (Allwinner A733, Vivante VIP9000, VIPLite runtime) instead of
its CPU. Useful for RAG / semantic search on the board without spending CPU that your other services need.

Measured on one board (details and caveats in [docs/benchmarks.md](docs/benchmarks.md)):

| | per text (128 tokens) | CPU time for 43 texts | cosine to the FP32 CPU model |
|---|---|---|---|
| CPU (sentence-transformers) | 190 – 420 ms | 24.2 s | 1.0 |
| **NPU, int16 (this repo)** | **~88 ms** | **0.28 s** | 0.9995 (min 0.999) |

Only **int16** is worth using: fp16 runs without acceleration (~2.6 s per text) and bf16 does not compile.

## How it works

```
text ──► tokenizer + word/position/type embedding sum (CPU, numpy) ──► npu_embed_server (C, VIPLite) ──► encoder on the NPU
                                                                                                          │
vector ◄── mean pooling + L2 normalize (CPU) ◄────────────────────────────────────────────────────────────┘
```

- `server/npu_embed_server.c` loads the compiled model (`.nb`) once and serves it on a Unix socket.
- `python/npu_embed` is the client (numpy + `tokenizers`, **no torch**), a CLI and an OpenAI-compatible
  `/v1/embeddings` endpoint using only the standard library.
- `convert/` is the recipe that turns the Hugging Face model into the `.nb` (needs an x86 PC and the Allwinner toolchain).

## What you need

- An Orange Pi 4 Pro with its Debian image (the NPU runtime ships with it): `ls /dev/vipcore`,
  `/usr/include/vip_lite.h`, `libVIPhal.so`, `libNBGlinker.so`. `gcc` and Python 3.10+.
- The compiled model `minilm_int16.nb`. **It is not in this repository** (see [NOTICE](NOTICE)): either use one
  you built, or build it with [convert/](#building-the-nb) below.

## Quick start (on the board)

```bash
git clone https://github.com/simplestringdev/orangepi4pro-npu-embeddings && cd orangepi4pro-npu-embeddings

server/build.sh                        # -> ~/.local/bin/npu_embed_server
scripts/fetch_model.sh                 # tokenizer.json + model.safetensors (~470 MB) -> ~/.cache/npu-embed/minilm
cp /path/to/minilm_int16.nb ~/.cache/npu-embed/

~/.local/bin/npu_embed_server ~/.cache/npu-embed/minilm_int16.nb "$XDG_RUNTIME_DIR/npu-embed.sock" &

pip install --user numpy tokenizers
export PYTHONPATH=$PWD/python NPU_EMBED_MODEL_DIR=~/.cache/npu-embed/minilm
python3 -m npu_embed embed "onde fica a documentação" "where is the documentation"
python3 -m npu_embed serve-http --port 8089        # POST /v1/embeddings, GET /v1/models, localhost only
```

From Python:

```python
import os
from npu_embed import NpuEmbedder

emb = NpuEmbedder(os.path.expanduser("~/.cache/npu-embed/minilm"))   # socket: $XDG_RUNTIME_DIR/npu-embed.sock
vectors = emb.encode(["bom dia", "good morning"])                    # list of unit-length 384-d vectors
```

To keep the server running across reboots see [systemd/npu-embed.service](systemd/npu-embed.service).

## Building the `.nb`

The compiler (**ACUITY**) only runs on x86-64 Linux inside Allwinner's Docker image
(`ubuntu-npu:v2.0.10.2` for the A733). Where to download it and how to load it is described in the
[Radxa ACUITY environment guide](https://docs.radxa.com/en/cubie/a7a/app-dev/npu-dev/cubie-acuity-env) (same NPU).
This repository does not redistribute it.

```bash
# 1. export the encoder as an NPU-friendly ONNX graph (any machine with torch)
pip install torch transformers sentence-transformers onnx
python convert/verify_graph.py                       # sanity: worst cosine vs the original should be ~1.0
python convert/export_onnx.py --out-dir work
python convert/make_calibration.py --out-dir work/calib      # ~47 generic paragraphs; pass --texts for text like yours

# 2. compile inside the Acuity container (x86 PC)
podman run -d --ipc=host --name acuity -v "$PWD/work:/workspace:Z" ubuntu-npu:v2.0.10.2 sleep infinity
podman cp convert/acuity_convert.sh acuity:/usr/local/bin/ && podman exec acuity chmod +x /usr/local/bin/acuity_convert.sh
podman exec acuity acuity_convert.sh /workspace int16
#    -> work/wksp/int16_nbg_unify/network_binary.nb   (~36 MB)  copy it to the board as minilm_int16.nb

# 3. optional: run it with the vendor tool and compare with the FP32 reference
python convert/make_inputs_dat.py work/calib work 4        # then /opt/vpm_run/vpm_run -s sample.txt
```

Gotchas that cost hours (all handled in the scripts):

- `pegasus import --input-size-list` must **not** contain the batch dimension, or the Transpose layers fail with
  "Dimension must be 5 but is 4".
- `bf16` export fails at graph verification ("configuration that cannot be supported"); use `int16`.
- `vip_prepare_network` must be called **before** `vip_set_input` on this driver (version 2.0.3.2).
- The int16 input uses fixed-point position 15, so inputs must stay within ±1 (true for MiniLM embedding sums; check
  `make_inputs_dat.py` output if you change the model).

## Limits and honesty

- Tested on a single Orange Pi 4 Pro running Debian 13 ("Orange Pi 1.1.0 Trixie", kernel 6.6.98-sun60iw2), VIPLite
  driver 2.0.3.2, ACUITY 6.30.22. A `.nb` built for one driver/toolchain version may not load on another.
- Accuracy was checked on 4 + 43 real texts against the FP32 model, not on a benchmark such as MTEB.
- Sequence length is fixed at 128 tokens (the model's own `max_seq_length`); a short text costs the same as a long one.
- One NPU: connections are served one at a time. The HTTP endpoint has no authentication; keep it on localhost.
- Other boards with the same chip (for example Radxa Cubie A7 series) should work in principle; untested.

## License

MIT for the code in this repository ([LICENSE](LICENSE)). Model and toolchain licenses: see [NOTICE](NOTICE).
