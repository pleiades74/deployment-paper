#!/usr/bin/env python3
"""
hardware_profile_v2.py — Hardware profiling with 4-core CPU and throttle detection.
Runs single-task models on each dataset.
"""
import os, sys, time, ctypes, threading, subprocess
import numpy as np
import pandas as pd
import psutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

flex = os.path.join(SCRIPT_DIR, "libtensorflowlite_flex.so")
core = os.path.join(SCRIPT_DIR, "libtensorflowlite.so")
if os.path.exists(flex): ctypes.CDLL(flex, mode=ctypes.RTLD_GLOBAL)
if os.path.exists(core): ctypes.CDLL(core, mode=ctypes.RTLD_GLOBAL)

sys.path.insert(0, SCRIPT_DIR)
from deploy_mt_supinn import DeployConfig, DatasetReader, load_scaler, ModelInference


class HardwareMonitor:
    def __init__(self, interval=0.5):
        self.interval = interval
        self.running = False
        self.cpu_total = []
        self.cpu_per_core = []
        self.ram = []
        self.temp = []
        self.throttled = []

    def _read_temp(self):
        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                return float(f.read()) / 1000.0
        except Exception:
            return np.nan

    def _read_throttled(self):
        try:
            out = subprocess.check_output(["vcgencmd", "get_throttled"]).decode().strip()
            val = int(out.split("=")[1], 16)
            return val != 0
        except Exception:
            return False

    def _loop(self):
        proc = psutil.Process(os.getpid())
        while self.running:
            per_core = psutil.cpu_percent(interval=None, percpu=True)
            self.cpu_per_core.append(per_core)
            self.cpu_total.append(sum(per_core))
            self.ram.append(proc.memory_info().rss / 1024 / 1024)
            self.temp.append(self._read_temp())
            self.throttled.append(self._read_throttled())
            time.sleep(self.interval)

    def start(self):
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self.running = False
        time.sleep(self.interval * 2)


def run_one(model, reader, dataset_type, n_windows=10000):
    if dataset_type == 'cwru':
        files = reader.load_files(dataset_type)[:100]
    elif dataset_type == 'paderborn':
        files = reader.load_files(dataset_type)[:50]
    else:
        files = reader.load_files(dataset_type)[:2000]

    if not files:
        print(f"❌ No files for {dataset_type}")
        return None

    monitor = HardwareMonitor(interval=0.5)
    monitor.start()

    latencies = []
    t_start = time.time()
    count = 0

    for fp in files:
        if dataset_type == 'cwru':      data = reader.read_cwru(fp)
        elif dataset_type == 'paderborn': data = reader.read_paderborn(fp)
        else:                            data = reader.read_pronostia(fp)
        if data is None or len(data) < 1024: continue
        for s in range(0, len(data) - 1024, 1024):
            window = data[s:s+1024]
            t0 = time.perf_counter()
            model.predict(window, dataset_type=dataset_type)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)
            count += 1
            if count >= n_windows: break
        if count >= n_windows: break

    elapsed = time.time() - t_start
    monitor.stop()

    cpu = np.array(monitor.cpu_total)
    ram = np.array(monitor.ram)
    temp = np.array([t for t in monitor.temp if not np.isnan(t)])
    per_core = np.array(monitor.cpu_per_core) if monitor.cpu_per_core else np.zeros((1,4))
    thr = np.array(monitor.throttled, dtype=bool)

    return {
        'dataset': dataset_type,
        'windows': count,
        'total_time_s': elapsed,
        'throughput_wps': count / elapsed if elapsed > 0 else np.nan,
        'latency_mean_ms': float(np.mean(latencies)) if latencies else np.nan,
        'latency_p95_ms': float(np.percentile(latencies, 95)) if latencies else np.nan,
        'cpu_total_mean_pct': float(np.mean(cpu)) if len(cpu) else np.nan,
        'cpu_core0_mean_pct': float(np.mean(per_core[:,0])) if per_core.shape[1] > 0 else np.nan,
        'cpu_core1_mean_pct': float(np.mean(per_core[:,1])) if per_core.shape[1] > 1 else np.nan,
        'cpu_core2_mean_pct': float(np.mean(per_core[:,2])) if per_core.shape[1] > 2 else np.nan,
        'cpu_core3_mean_pct': float(np.mean(per_core[:,3])) if per_core.shape[1] > 3 else np.nan,
        'cpu_single_core_max_pct': float(np.mean(np.max(per_core, axis=1))) if per_core.size else np.nan,
        'ram_mean_mb': float(np.mean(ram)) if len(ram) else np.nan,
        'ram_peak_mb': float(np.max(ram)) if len(ram) else np.nan,
        'temp_mean_c': float(np.mean(temp)) if len(temp) else np.nan,
        'temp_max_c': float(np.max(temp)) if len(temp) else np.nan,
        'throttled_fraction': float(np.mean(thr)) if len(thr) else 0.0,
    }


if __name__ == "__main__":
    MODELS = [
        ("single_cwru",  "single_task_cwru.tflite",       'cwru'),
        ("single_pader", "single_task_paderborn.tflite",  'paderborn'),
        ("single_prono", "single_task_pronostia.tflite",  'pronostia'),
    ]

    all_results = []
    for label, mfile, dataset_type in MODELS:
        mpath = os.path.join(SCRIPT_DIR, mfile)
        if not os.path.exists(mpath):
            print(f"⚠️ Missing: {mfile}")
            continue

        config = DeployConfig()
        config.model_path = mpath
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
                               flex_library_path=config.flex_library_path,
                               core_library_path=config.core_library_path)
        if not model.load():
            continue

        print(f"\n{'='*60}\n{label} × {dataset_type.upper()}\n{'='*60}")
        r = run_one(model, reader, dataset_type, n_windows=10000)
        if r is None:
            continue
        r['model'] = label
        all_results.append(r)
        print(f"  Windows:          {r['windows']}")
        print(f"  Latency mean:     {r['latency_mean_ms']:.2f} ms")
        print(f"  Throughput:       {r['throughput_wps']:.2f} w/s")
        print(f"  CPU total mean:   {r['cpu_total_mean_pct']:.1f}%")
        print(f"  CPU core 0-3:     {r['cpu_core0_mean_pct']:.1f} / {r['cpu_core1_mean_pct']:.1f} / {r['cpu_core2_mean_pct']:.1f} / {r['cpu_core3_mean_pct']:.1f}")
        print(f"  CPU single-core max mean: {r['cpu_single_core_max_pct']:.1f}%")
        print(f"  RAM peak:         {r['ram_peak_mb']:.1f} MB")
        print(f"  Temp max:         {r['temp_max_c']:.1f} °C")
        print(f"  Throttled:        {r['throttled_fraction']*100:.1f}% of samples")

    out = os.path.join(SCRIPT_DIR, "results", "hardware_profiling_singletask.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(all_results).to_csv(out, index=False)
    print(f"\n✅ Saved: {out}")
    print(f"   Rows: {len(all_results)}")
