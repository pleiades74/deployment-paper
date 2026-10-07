#!/usr/bin/env python3
"""
disambiguate_rul.py v3 — spread across full bearing life.
"""
import os, ctypes
import numpy as np
import pandas as pd
import joblib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

flex = os.path.join(SCRIPT_DIR, "libtensorflowlite_flex.so")
core = os.path.join(SCRIPT_DIR, "libtensorflowlite.so")
if os.path.exists(flex): ctypes.CDLL(flex, mode=ctypes.RTLD_GLOBAL)
if os.path.exists(core): ctypes.CDLL(core, mode=ctypes.RTLD_GLOBAL)

import tensorflow as tf
from scipy.stats import skew, kurtosis

scaler = joblib.load("ews_scaler.pkl")
print(f"✅ Scaler loaded")

interp = tf.lite.Interpreter(model_path="model_full_42_fp32.tflite")
interp.allocate_tensors()

inputs = interp.get_input_details()
vib_idx = next(d['index'] for d in inputs if list(d['shape']) == [1, 1024, 4])
ews_idx = next(d['index'] for d in inputs if list(d['shape']) == [1, 11])

outputs = interp.get_output_details()
rul_cands = sorted([d for d in outputs if list(d['shape']) == [1, 1]],
                    key=lambda d: d['name'])
print(f"Two [1,1] outputs:")
for d in rul_cands:
    print(f"  name='{d['name']}'  index={d['index']}")

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

def ews_from_multichannel(sig):
    if sig.ndim == 1:
        return ews_features(sig)
    feats = [ews_features(sig[:, c]) for c in range(sig.shape[1])]
    return np.mean(feats, axis=0)

# ---- Load the FULL file list (all files, not just first 200) ----
bearing = "Bearing1_3"
folder = f"FEMTOBearingDataSet/Full_Test_Set/{bearing}"
all_files = sorted([f for f in os.listdir(folder) if f.startswith("acc_")])
print(f"\nBearing {bearing}: {len(all_files)} files total")

# Subsample 200 evenly spaced across the FULL range
n_samples = 200
indices = np.linspace(0, len(all_files)-1, n_samples).astype(int)
sampled_files = [all_files[i] for i in indices]

time_axis, out_a, out_b = [], [], []

for rank, (file_i, fname) in enumerate(zip(indices, sampled_files)):
    fp = os.path.join(folder, fname)
    with open(fp, 'r') as f:
        first = f.readline().strip()
    sep = ';' if ';' in first else ','
    df = pd.read_csv(fp, header=None, delimiter=sep)
    arr = df.values.astype(np.float32)
    if arr.shape[1] < 6:
        continue
    h, v = arr[:1024, 4], arr[:1024, 5]

    padded = np.zeros((1, 1024, 4), dtype=np.float32)
    padded[0, :, 0] = h
    padded[0, :, 1] = v

    raw_ews = ews_from_multichannel(np.stack([h, v], axis=1)).reshape(1, 11)
    ews = scaler.transform(raw_ews).astype(np.float32)

    interp.set_tensor(vib_idx, padded)
    interp.set_tensor(ews_idx, ews)
    interp.invoke()

    a = float(interp.get_tensor(rul_cands[0]['index']).flatten()[0])
    b = float(interp.get_tensor(rul_cands[1]['index']).flatten()[0])
    time_axis.append(file_i)  # use file index (progress through life)
    out_a.append(a)
    out_b.append(b)

# Report
print(f"\nOutput trends over FULL bearing life ({len(time_axis)} samples from index {time_axis[0]} to {time_axis[-1]}):")
print(f"  {rul_cands[0]['name']}: first={out_a[0]:.4f}  mid={out_a[len(out_a)//2]:.4f}  last={out_a[-1]:.4f}  min={min(out_a):.4f}  max={max(out_a):.4f}")
print(f"  {rul_cands[1]['name']}: first={out_b[0]:.4f}  mid={out_b[len(out_b)//2]:.4f}  last={out_b[-1]:.4f}  min={min(out_b):.4f}  max={max(out_b):.4f}")

from scipy.stats import linregress
slope_a, _, r_a, _, _ = linregress(time_axis, out_a)
slope_b, _, r_b, _, _ = linregress(time_axis, out_b)
print(f"\nTrend slopes (file index as x-axis):")
print(f"  {rul_cands[0]['name']}: slope={slope_a:+.6e}  R²={r_a**2:.3f}")
print(f"  {rul_cands[1]['name']}: slope={slope_b:+.6e}  R²={r_b**2:.3f}")

print(f"\nInterpretation:")
print(f"  Expected for RUL: negative slope (RUL decreases as bearing degrades)")
print(f"  Expected for shared_damage: positive slope (damage increases)")
print(f"")
if r_a**2 > 0.3 and slope_a < 0:
    print(f"  ✅ {rul_cands[0]['name']} decreases → pronostia_rul")
elif r_a**2 > 0.3 and slope_a > 0:
    print(f"  ✅ {rul_cands[0]['name']} increases → shared_damage")
if r_b**2 > 0.3 and slope_b < 0:
    print(f"  ✅ {rul_cands[1]['name']} decreases → pronostia_rul")
elif r_b**2 > 0.3 and slope_b > 0:
    print(f"  ✅ {rul_cands[1]['name']} increases → shared_damage")
if r_a**2 < 0.3 and r_b**2 < 0.3:
    print(f"  ⚠️ Neither output shows a trend across full bearing life.")
    print(f"     The model is NOT producing meaningful RUL predictions on TFLite.")
