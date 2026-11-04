# Measurements

All numbers from one Orange Pi 4 Pro (4 GB, Debian 13, kernel 6.6.98-sun60iw2, VIPLite 2.0.3.2), November 2026.
The board was otherwise idle. These are small, informal measurements, not a benchmark suite.

## Speed

| path | per sequence | note |
|---|---|---|
| NPU int16, server only (`vpm_run`) | 80 – 81 ms | 36 MB NBG, 128 tokens fixed |
| NPU int16, end to end (tokenize + sum + NPU + pool) | ~88 ms | 43 real paragraphs, one connection |
| NPU fp16 | 2.6 s | no hardware acceleration for fp16 on this NPU |
| CPU torch, 1 – 8 threads | 346 – 426 ms | more threads did not help |
| CPU sentence-transformers (dynamic length, batch 16) | ~190 ms | shorter texts are cheaper on the CPU |

## CPU time spent by the Python process (43 texts)

| path | CPU seconds |
|---|---|
| NPU client | 0.28 |
| CPU model | 24.2 |

The kernel's load average shows ~3 while the NPU driver is loaded: three driver threads (`vip_*`) sit in
uninterruptible sleep (`D`), which the load average counts although they use no CPU. Look at `top` instead.

## Accuracy against the FP32 CPU model (cosine similarity of sentence embeddings)

| NBG | 4 calibration paragraphs | 43 texts + edge cases (empty, 1 word, very long) |
|---|---|---|
| int16 | 0.9994 – 0.9997 | mean 0.9995, min 0.9990 |
| fp16 | 0.99998 – 0.99999 | not measured |

Reproducibility: the `.nb` was rebuilt from scratch with the scripts in `convert/` (generic calibration text instead of
personal notes): same size (36,264,440 bytes), cosine to the CPU model on 43 texts min 0.9989, mean 0.99945.

The rewritten ONNX graph itself matches the original model at cosine 1.0000 in FP32 (`convert/verify_graph.py`),
so all the loss comes from int16 quantization.

Sanity check of semantics through the HTTP endpoint: "bom dia" vs "good morning" 0.94, "bom dia" vs "cafe" 0.45.
