#!/usr/bin/env python3
"""
power_monitor.py — Sample PMIC power rails + temperature during inference.
Runs 5 minutes of continuous inference while sampling power every 1 second.
"""
import os, sys, time, ctypes, threading, subprocess
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

flex = os.path.join(SCRIPT_DIR, "libtensorflowlite_flex.so")
core = os.path.join(SCRIPT_DIR, "libtensorflowlite.so")
if os.path.exists(flex): ctypes.CDLL(flex, mode=ctypes.RTLD_GLOBAL)
if os.path.exists(core): ctypes.CDLL(core, mode=ctypes.RTLD_GLOBAL)

sys.path.insert(0, SCRIPT_DIR)
from deploy_mt_supinn import DeployConfig, DatasetReader, load_scaler, ModelInference

MODEL = "model_full_42_int8.tflite"
DATASET = "pronostia"
DURATION_S = 300  # 5 minutes


def read_pmic():
    try:
        out = subprocess.check_output(["vcgencmd", "pmic_read_adc"]).decode()
        rails = {}
        for line in out.strip().split("\n"):
            parts = line.replace("=", " ").split()
            if len(parts) >= 3:
                name = parts[0]
                try:
                    val = float(parts[-1].rstrip("VA"))
                    rails[name] = val
                except ValueError:
                    pass
        return rails
    except Exception:
        return {}


def read_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return float(f.read()) / 1000.0
    except Exception:
        return np.nan


def read_throttled():
    try:
        out = subprocess.check_output(["vcgencmd", "get_throttled"]).decode().strip()
        return out
    except Exception:
        return "throttled=unknown"


def inference_worker(model, reader, dataset_type, stop_event):
    files = reader.load_files(dataset_type)[:200]
    count = 0
    while not stop_event.is_set():
        for fp in files:
            if stop_event.is_set(): break
            if dataset_type == 'cwru':      data = reader.read_cwru(fp)
            elif dataset_type == 'paderborn': data = reader.read_paderborn(fp)
            else:                            data = reader.read_pronostia(fp)
            if data is None or len(data) < 1024: continue
            for s in range(0, len(data) - 1024, 1024):
                if stop_event.is_set(): break
                model.predict(data[s:s+1024], dataset_type=dataset_type)
                count += 1
    return count


if __name__ == "__main__":
    config = DeployConfig()
    config.model_path = os.path.join(SCRIPT_DIR, MODEL)
    config.scaler_path = os.path.join(SCRIPT_DIR, "ews_scaler.pkl")
    config.dataset_paths = {
        'cwru':      os.path.join(SCRIPT_DIR, "csv_files"),
        'paderborn': os.path.join(SCRIPT_DIR, "Paderborn"),
        'pronostia': os.path.join(SCRIPT_DIR, "FEMTOBearingDataSet"),
    }
    config.flex_library_path = flex
    config.core_library_path = core

    reader = DatasetReader(config)
    scaler = load_scaler(config.scaler_path)
    model = ModelInference(config.model_path, scaler,
                           flex_library_path=flex, core_library_path=core)
    if not model.load():
        sys.exit(1)

    stop = threading.Event()
    worker = threading.Thread(target=inference_worker,
                              args=(model, reader, DATASET, stop),
                              daemon=True)
    worker.start()

    samples = []
    t_start = time.time()
    print(f"Sampling for {DURATION_S} seconds...")
    while time.time() - t_start < DURATION_S:
        t = time.time() - t_start
        rails = read_pmic()
        row = {
            'time_s': t,
            'temp_c': read_temp(),
            'throttled': read_throttled(),
            **rails
        }
        samples.append(row)
        time.sleep(1.0)

    stop.set()
    worker.join(timeout=5)

    df = pd.DataFrame(samples)
    out = os.path.join(SCRIPT_DIR, "results", "power_samples.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)

    print(f"\n✅ Saved {len(df)} samples to {out}")
    print(f"Columns: {df.columns.tolist()}")

    # Summary
    print("\nPower summary (mean over duration):")
    for col in df.columns:
        if col.endswith('_V') or col.endswith('_A'):
            print(f"  {col:40s} mean={df[col].mean():.4f}  max={df[col].max():.4f}")
