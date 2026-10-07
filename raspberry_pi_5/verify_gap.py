#!/usr/bin/env python3
"""
Verify that TFLite conversion preserves the RUL head on the same
4750 Pronostia windows used by the server-side Keras evaluation
(RMSE = 0.2345).
"""

import os
import sys
import ctypes
import time
import numpy as np

# Flex delegate BEFORE importing TensorFlow Lite
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ctypes.CDLL(os.path.join(SCRIPT_DIR, "libtensorflowlite_flex.so"),
            mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL(os.path.join(SCRIPT_DIR, "libtensorflowlite.so"),
            mode=ctypes.RTLD_GLOBAL)

import tensorflow as tf
import joblib


def log(msg):
    print(msg, flush=True)


# ----------------------------------------------------------------
# 1. Load reference data
# ----------------------------------------------------------------
log("=" * 60)
log("STEP 1 — Loading reference data")
log("=" * 60)

feats   = np.load("reference_features.npy")
targets = np.load("reference_rul_targets.npy")
vib_2ch = np.load("reference_vib_pi.npy")

scaler = joblib.load("ews_scaler.pkl")
feats_scaled = ((feats - scaler.mean_) / scaler.scale_).astype(np.float32)

vib = np.zeros((len(vib_2ch), 1024, 4), dtype=np.float32)
vib[:, :, :2] = vib_2ch

log(f"  Features : {feats.shape} -> scaled {feats_scaled.shape}")
log(f"  Targets  : {targets.shape}, range {targets.min():.4f} to {targets.max():.4f}")
log(f"  Vibration: {vib.shape}")
log("")


def report(name, preds):
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    mae  = float(np.mean(np.abs(preds - targets)))
    ss_res = float(np.sum((targets - preds) ** 2))
    ss_tot = float(np.sum((targets - targets.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    log(f"{name:14s}  RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")
    return rmse, mae, r2


# ----------------------------------------------------------------
# 2. TFLite inference with progress
# ----------------------------------------------------------------
def run_tflite(model_path, name):
    log("")
    log("=" * 60)
    log(f"STEP 2 — Running {name} on 4750 windows")
    log("=" * 60)

    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    in_d  = interp.get_input_details()
    out_d = interp.get_output_details()

    vib_idx = [d['index'] for d in in_d  if list(d['shape']) == [1, 1024, 4]][0]
    ews_idx = [d['index'] for d in in_d  if list(d['shape']) == [1, 11]][0]

    rul_candidates = [d for d in out_d if list(d['shape']) == [1, 1]]
    rul_idx = None
    for d in rul_candidates:
        if 'StatefulPartitionedCall_1:4' in d['name']:
            rul_idx = d['index']
            break
    if rul_idx is None:
        rul_idx = rul_candidates[0]['index']

    log(f"  vib_idx={vib_idx}, ews_idx={ews_idx}, rul_idx={rul_idx}")

    n = len(feats)
    preds = np.zeros(n, dtype=np.float32)

    # Warm-up
    for _ in range(5):
        interp.set_tensor(vib_idx, vib[0:1])
        interp.set_tensor(ews_idx, feats_scaled[0:1])
        interp.invoke()

    t0 = time.time()
    log_every = 500
    for i in range(n):
        interp.set_tensor(vib_idx, vib[i:i+1])
        interp.set_tensor(ews_idx, feats_scaled[i:i+1])
        interp.invoke()
        preds[i] = interp.get_tensor(rul_idx).reshape(-1)[0]

        if (i + 1) % log_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate
            log(f"    [{i+1:5d}/{n}]  {rate:6.1f} w/s  ETA {eta:5.0f}s")

    elapsed = time.time() - t0
    log(f"  Done in {elapsed:.0f}s ({n/elapsed:.1f} w/s)")
    return report(name, preds), preds


# ----------------------------------------------------------------
# 3. Run all three
# ----------------------------------------------------------------
results = {}
for fname, name in [
    ("model_full_42_fp32.tflite", "TFLite FP32"),
    ("model_full_42_fp16.tflite", "TFLite FP16"),
    ("model_full_42_int8.tflite", "TFLite INT8"),
]:
    if not os.path.exists(fname):
        log(f"SKIP {name} — file not found: {fname}")
        continue
    (rmse, mae, r2), preds = run_tflite(fname, name)
    results[name] = preds

# ----------------------------------------------------------------
# 4. Summary
# ----------------------------------------------------------------
log("")
log("=" * 60)
log("SUMMARY")
log("=" * 60)
log("Keras FP32 (Kaggle, same 4750 windows): RMSE=0.2345  MAE=0.1926")
for name, preds in results.items():
    report(name, preds)

if len(results) >= 2:
    keys = list(results.keys())
    a, b = results[keys[0]], results[keys[1]]
    d = np.abs(a - b)
    log(f"\nMax abs diff ({keys[0]} vs {keys[1]}): {d.max():.6e}")

np.savez("pi_comparison.npz", **{k.replace(" ", "_"): v
                                  for k, v in results.items()},
         targets=targets)
log("\nSaved: pi_comparison.npz")
