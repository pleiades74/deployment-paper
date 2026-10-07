#!/usr/bin/env python3
"""op_histogram.py — Real op counts per model."""
import os, ctypes
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

flex = os.path.join(SCRIPT_DIR, "libtensorflowlite_flex.so")
core = os.path.join(SCRIPT_DIR, "libtensorflowlite.so")
if os.path.exists(flex): ctypes.CDLL(flex, mode=ctypes.RTLD_GLOBAL)
if os.path.exists(core): ctypes.CDLL(core, mode=ctypes.RTLD_GLOBAL)

import tensorflow as tf

MODELS = [
    "model_full_42_fp32.tflite",
    "model_full_42_fp16.tflite",
    "model_full_42_int8.tflite",
    "model_lw.tflite",         # adjust to actual light fp32 name
    "model_lw_int8.tflite",
    "single_task_cwru.tflite",
    "single_task_paderborn.tflite",
    "single_task_pronostia.tflite",
]

for fname in MODELS:
    if not os.path.exists(fname):
        print(f"⚠️ Missing: {fname}")
        continue
    print("=" * 72)
    print(f"{fname}  ({os.path.getsize(fname)/1024:.1f} KB)")
    print("=" * 72)
    try:
        interp = tf.lite.Interpreter(model_path=fname)
        interp.allocate_tensors()
        ops = interp._get_ops_details()
        counts = Counter(op['op_name'] for op in ops)
        print(f"Total ops: {len(ops)}")
        for name, count in counts.most_common():
            marker = ""
            if "Flex" in name or "Delegate" in name:
                marker = "  ← FALLBACK" 
            print(f"  {name:45s} {count:4d}{marker}")
        print()
    except Exception as e:
        print(f"  ❌ {e}\n")
