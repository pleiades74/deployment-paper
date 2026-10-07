#!/usr/bin/env python3
"""
check_ops.py — List ops and their types for each TFLite model.
Run on Raspberry Pi. Uses Flex delegate if available.
"""
import os
import ctypes
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# ---- Load Flex delegate BEFORE importing TF ----
flex = os.path.join(SCRIPT_DIR, "libtensorflowlite_flex.so")
core = os.path.join(SCRIPT_DIR, "libtensorflowlite.so")

if os.path.exists(flex):
    try:
        ctypes.CDLL(flex, mode=ctypes.RTLD_GLOBAL)
        print(f"✅ Loaded: {flex}")
    except Exception as e:
        print(f"⚠️ Could not load Flex: {e}")
else:
    print(f"⚠️ Flex library not found: {flex}")

if os.path.exists(core):
    try:
        ctypes.CDLL(core, mode=ctypes.RTLD_GLOBAL)
        print(f"✅ Loaded: {core}")
    except Exception as e:
        print(f"⚠️ Could not load core: {e}")

import tensorflow as tf
print(f"TensorFlow: {tf.__version__}")
print()

# ---- List all .tflite files ----
tflite_files = sorted([f for f in os.listdir(".") if f.endswith(".tflite")])
if not tflite_files:
    print("❌ No .tflite files found.")
    exit(1)

print(f"Found {len(tflite_files)} TFLite models:")
for f in tflite_files:
    print(f"  - {f}")
print()

# ---- Run op analysis on each ----
for fname in tflite_files:
    size_kb = os.path.getsize(fname) / 1024
    print("=" * 72)
    print(f"{fname}  ({size_kb:.2f} KB)")
    print("=" * 72)

    try:
        interp = tf.lite.Interpreter(model_path=fname)
        interp.allocate_tensors()
        ops = interp._get_ops_details()
        op_names = [op['op_name'] for op in ops]
        counts = Counter(op_names)

        print(f"Total ops: {len(ops)}")
        print("\nOp histogram (op_name : count):")
        for name, count in counts.most_common():
            print(f"  {name:40s} {count}")
        print()
    except Exception as e:
        print(f"  ❌ Failed: {e}\n")

    # Also try the experimental analyzer (may produce more detail)
    try:
        print("-" * 72)
        print(f"tf.lite.experimental.Analyzer output for {fname}:")
        print("-" * 72)
        tf.lite.experimental.Analyzer.analyze(model_path=fname)
        print()
    except Exception as e:
        print(f"  ⚠️ Analyzer failed (not critical): {e}\n")

print("=" * 72)
print("✅ Op analysis complete")
print("=" * 72)
