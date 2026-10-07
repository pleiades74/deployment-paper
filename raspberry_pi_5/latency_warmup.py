#!/usr/bin/env python3
"""
latency_warmup.py — Warm-up + steady-state latency per model per dataset.
100 warm-up inferences, then 1000 timed inferences (interpreter.invoke only).
"""
import os, sys, time, ctypes
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

flex = os.path.join(SCRIPT_DIR, "libtensorflowlite_flex.so")
core = os.path.join(SCRIPT_DIR, "libtensorflowlite.so")
if os.path.exists(flex): ctypes.CDLL(flex, mode=ctypes.RTLD_GLOBAL)
if os.path.exists(core): ctypes.CDLL(core, mode=ctypes.RTLD_GLOBAL)

sys.path.insert(0, SCRIPT_DIR)
from deploy_mt_supinn import (
    DeployConfig, DatasetReader, load_scaler, ModelInference
)

import tensorflow as tf
from scipy.stats import skew, kurtosis

WARMUP = 100
TIMED  = 1000

MODELS = [
    ("full_fp32",  "model_full_42_fp32.tflite"),
    ("full_fp16",  "model_full_42_fp16.tflite"),
    ("full_int8",  "model_full_42_int8.tflite"),
    ("light_fp32", "model_lw_fp32.tflite"),
    ("light_fp16", "model_lw_fp16.tflite"),
    ("light_int8", "model_lw_int8.tflite"),
]

def ews_features(sig):
    sig = np.asarray(sig, dtype=np.float32).reshape(-1)
    eps = 1e-10
    rms = np.sqrt(np.mean(sig**2) + eps)
    peak = np.max(np.abs(sig))
    crest = peak / (rms + eps)
    std = np.std(sig)
    sk = skew(sig, bias=False, nan_policy='omit') if std > eps else 0.0
    ku = kurtosis(sig, fisher=True, bias=False, nan_policy='omit') if std > eps else -3.0
    p2p = np.ptp(sig)
    ma = np.mean(np.abs(sig)) + eps
    sf = rms / ma
    imp = peak / ma
    msa = np.mean(np.sqrt(np.abs(sig))) + eps
    cf = peak / (msa**2 + eps)
    ac = np.sum(sig[:-1]*sig[1:]) / (np.sum(sig**2)+eps) if len(sig) > 2 else 0.0
    return np.array([rms, peak, crest, std, sk, ku, p2p, sf, imp, cf, ac], dtype=np.float32)

def ews_multichannel(sig):
    if sig.ndim == 1:
        return ews_features(sig)
    feats = [ews_features(sig[:, c]) for c in range(sig.shape[1])]
    return np.mean(feats, axis=0)

def gather_windows(reader, dataset_type, n_needed):
    if dataset_type == 'cwru':
        files = reader.load_files(dataset_type)[:10]
    elif dataset_type == 'paderborn':
        files = reader.load_files(dataset_type)[:20]
    else:
        files = reader.load_files(dataset_type)[:50]

    windows = []
    for fp in files:
        if dataset_type == 'cwru':      data = reader.read_cwru(fp)
        elif dataset_type == 'paderborn': data = reader.read_paderborn(fp)
        else:                            data = reader.read_pronostia(fp)
        if data is None or len(data) < 1024: continue
        for s in range(0, len(data) - 1024, 1024):
            windows.append(data[s:s+1024])
            if len(windows) >= n_needed + WARMUP:
                return windows[:n_needed + WARMUP]

    # If we didn't collect enough (e.g. CWRU has only 6 files),
    # cycle through what we have until we reach the target count.
    if len(windows) == 0:
        print(f"  ⚠️ No windows collected for {dataset_type}")
        return windows

    target = n_needed + WARMUP
    print(f"  Only {len(windows)} windows available, cycling to {target}")
    while len(windows) < target:
        take = min(target - len(windows), len(windows))
        windows.extend(windows[:take])
    return windows[:target]

# Load scaler once
config = DeployConfig()
config.scaler_path = os.path.join(SCRIPT_DIR, "ews_scaler.pkl")
scaler = load_scaler(config.scaler_path)

all_results = []

for label, mfile in MODELS:
    mpath = os.path.join(SCRIPT_DIR, mfile)
    if not os.path.exists(mpath):
        print(f"⚠️ Skipping {label}: {mfile} not found")
        continue

    for dataset_type in ['cwru', 'paderborn', 'pronostia']:
        print(f"\n{'='*60}\n{label} × {dataset_type}\n{'='*60}")

        cfg = DeployConfig()
        cfg.window_size = 1024
        cfg.dataset_paths = {
            'cwru':      os.path.join(SCRIPT_DIR, "csv_files"),
            'paderborn': os.path.join(SCRIPT_DIR, "Paderborn"),
            'pronostia': os.path.join(SCRIPT_DIR, "FEMTOBearingDataSet"),
        }
        reader = DatasetReader(cfg)

        # Load model for warm-up windows
        from deploy_mt_supinn import ModelInference
        m = ModelInference(mpath, scaler)
        if not m.load(): continue

        windows = gather_windows(reader, dataset_type, TIMED)
        if len(windows) < WARMUP + 10:
            print(f"  ⚠️ Only {len(windows)} windows, skipping")
            continue

        # Pure invoke-only timing (fresh interpreter)
        interp = tf.lite.Interpreter(model_path=mpath)
        interp.allocate_tensors()
        inputs = interp.get_input_details()
        vib_idx = next(d['index'] for d in inputs if list(d['shape']) == [1, 1024, 4])
        ews_idx = next(d['index'] for d in inputs if list(d['shape']) == [1, 11])

        def prepare(window):
            padded = np.zeros((1024, 4), dtype=np.float32)
            if window.ndim == 2:
                n = min(window.shape[1], 4)
                padded[:, :n] = window[:, :n]
            else:
                padded[:, 0] = window
            if dataset_type == 'paderborn':
                v = window[:, 0] if window.ndim == 2 else window
                ews_raw = ews_features(v)
            else:
                ews_raw = ews_multichannel(window)
            ews = scaler.transform(ews_raw.reshape(1, -1)).astype(np.float32)
            return padded.reshape(1, 1024, 4).astype(np.float32), ews

        print(f"  Warming up ({WARMUP})...")
        for i in range(WARMUP):
            v, e = prepare(windows[i])
            interp.set_tensor(vib_idx, v)
            interp.set_tensor(ews_idx, e)
            interp.invoke()

        print(f"  Timing ({TIMED})...")
        times = []
        for i in range(WARMUP, WARMUP + TIMED):
            v, e = prepare(windows[i])
            interp.set_tensor(vib_idx, v)
            interp.set_tensor(ews_idx, e)
            t0 = time.perf_counter()
            interp.invoke()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        times = np.array(times)
        first20 = times[:20].mean()
        last20  = times[-20:].mean()

        r = {
            'model': label,
            'dataset': dataset_type,
            'n_timed': len(times),
            'mean_ms': float(times.mean()),
            'median_ms': float(np.median(times)),
            'p95_ms': float(np.percentile(times, 95)),
            'p99_ms': float(np.percentile(times, 99)),
            'std_ms': float(times.std()),
            'first20_ms': float(first20),
            'last20_ms': float(last20),
            'drift_ms': float(first20 - last20),
        }
        print(f"  mean={r['mean_ms']:.2f} ms  p95={r['p95_ms']:.2f}  "
              f"first20={first20:.2f}  last20={last20:.2f}")
        all_results.append(r)

out = os.path.join(SCRIPT_DIR, "results", "latency_warmup.csv")
os.makedirs(os.path.dirname(out), exist_ok=True)
pd.DataFrame(all_results).to_csv(out, index=False)
print(f"\n✅ Saved: {out}")
