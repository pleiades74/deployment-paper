#!/usr/bin/env python3
"""bench_onnx.py — Benchmark ONNX FP32 on Pi."""
import os, sys, time, ctypes
import numpy as np
import onnxruntime as ort

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

flex = os.path.join(SCRIPT_DIR, "libtensorflowlite_flex.so")
core = os.path.join(SCRIPT_DIR, "libtensorflowlite.so")
if os.path.exists(flex): ctypes.CDLL(flex, mode=ctypes.RTLD_GLOBAL)
if os.path.exists(core): ctypes.CDLL(core, mode=ctypes.RTLD_GLOBAL)

sys.path.insert(0, SCRIPT_DIR)
from deploy_mt_supinn import DeployConfig, DatasetReader, load_scaler, compute_ews_features, compute_ews_from_multichannel

MODEL = "mt_supinn_fp32.onnx"
WARMUP, TIMED = 100, 1000

print(f"Loading ONNX model: {MODEL}")
print(f"  Size: {os.path.getsize(MODEL)/1024:.2f} KB")

session = ort.InferenceSession(MODEL, providers=['CPUExecutionProvider'])
input_names = [i.name for i in session.get_inputs()]
print(f"  Inputs: {input_names}")

config = DeployConfig()
config.scaler_path = os.path.join(SCRIPT_DIR, "ews_scaler.pkl")
config.dataset_paths = {
    'cwru':      os.path.join(SCRIPT_DIR, "csv_files"),
    'paderborn': os.path.join(SCRIPT_DIR, "Paderborn"),
    'pronostia': os.path.join(SCRIPT_DIR, "FEMTOBearingDataSet"),
}
reader = DatasetReader(config)
scaler = load_scaler(config.scaler_path)

# Gather windows from all datasets
windows = []
for ds in ['cwru', 'paderborn', 'pronostia']:
    files = reader.load_files(ds)[:30]
    for fp in files:
        if ds == 'cwru':      data = reader.read_cwru(fp)
        elif ds == 'paderborn': data = reader.read_paderborn(fp)
        else:                    data = reader.read_pronostia(fp)
        if data is None or len(data) < 1024: continue
        for s in range(0, len(data) - 1024, 1024):
            windows.append((ds, data[s:s+1024]))

print(f"Collected {len(windows)} windows")

def prepare(ds, window):
    padded = np.zeros((1024, 4), dtype=np.float32)
    if window.ndim == 2:
        n = min(window.shape[1], 4)
        padded[:, :n] = window[:, :n]
    else:
        padded[:, 0] = window
    if ds == 'paderborn':
        v = window[:, 0] if window.ndim == 2 else window
        ews_raw = compute_ews_features(v)
    else:
        ews_raw = compute_ews_from_multichannel(window)
    ews = scaler.transform(ews_raw.reshape(1, -1)).astype(np.float32)
    return padded.reshape(1, 1024, 4).astype(np.float32), ews

# Warm-up
print("Warming up...")
for i in range(WARMUP):
    ds, w = windows[i]
    v, e = prepare(ds, w)
    session.run(None, {input_names[0]: v, input_names[1]: e})

# Timed
print(f"Timing {TIMED}...")
times = []
for i in range(WARMUP, WARMUP + TIMED):
    ds, w = windows[i]
    v, e = prepare(ds, w)
    t0 = time.perf_counter()
    session.run(None, {input_names[0]: v, input_names[1]: e})
    t1 = time.perf_counter()
    times.append((t1 - t0) * 1000)

times = np.array(times)
print(f"\nONNX FP32:")
print(f"  mean:    {times.mean():.2f} ms")
print(f"  median:  {np.median(times):.2f} ms")
print(f"  p95:     {np.percentile(times, 95):.2f} ms")
print(f"  first20: {times[:20].mean():.2f} ms")
print(f"  last20:  {times[-20:].mean():.2f} ms")

with open("onnx_bench.txt", "w") as f:
    f.write(f"ONNX FP32 benchmark\n")
    f.write(f"mean_ms: {times.mean():.4f}\n")
    f.write(f"median_ms: {np.median(times):.4f}\n")
    f.write(f"p95_ms: {np.percentile(times, 95):.4f}\n")
    f.write(f"first20_ms: {times[:20].mean():.4f}\n")
    f.write(f"last20_ms: {times[-20:].mean():.4f}\n")
    f.write(f"n_timed: {len(times)}\n")

print(f"\n✅ Saved: onnx_bench.txt")
