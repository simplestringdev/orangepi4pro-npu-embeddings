#!/usr/bin/env python3
"""Turn calibration .npy pairs into the raw .dat inputs that /opt/vpm_run/vpm_run expects (to test a .nb on the board).

  int16 NBG: emb as int16 Q15 (x * 32768), mask as float16        fp16 NBG: emb and mask as float16
Run `nbinfo -in <file.nb>` (in the Acuity container) to see the exact input format and fixed-point position.

usage: make_inputs_dat.py <calib-dir> <out-dir> [count]
"""
import sys

import numpy as np

calib, out = sys.argv[1], sys.argv[2]
count = int(sys.argv[3]) if len(sys.argv) > 3 else 4
for k in range(count):
    e = np.load(f"{calib}/emb_{k:03d}.npy").reshape(128, 384).astype(np.float32)
    m = np.load(f"{calib}/mask_{k:03d}.npy").reshape(128).astype(np.float32)
    print(k, "emb range", round(float(e.min()), 3), round(float(e.max()), 3), "(int16 Q15 needs |x| < 1)")
    e.astype(np.float16).tofile(f"{out}/fp16_{k}_emb.dat")
    m.astype(np.float16).tofile(f"{out}/fp16_{k}_mask.dat")
    np.clip(np.round(e * 32768.0), -32768, 32767).astype(np.int16).tofile(f"{out}/int16_{k}_emb.dat")
    m.astype(np.float16).tofile(f"{out}/int16_{k}_mask.dat")
