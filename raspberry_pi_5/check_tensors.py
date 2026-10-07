#!/usr/bin/env python3
"""
check_tensors.py — Print input/output tensor names for all TFLite models.
Run on Raspberry Pi. Uses Flex delegate if available.
"""
import os
import ctypes

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

# ---- List all .tflite files in this folder ----
tflite_files = sorted([f for f in os.listdir(".") if f.endswith(".tflite")])
if not tflite_files:
    print("❌ No .tflite files found in this directory.")
    exit(1)

print(f"Found {len(tflite_files)} TFLite models:")
for f in tflite_files:
    print(f"  - {f}")
print()

# ---- Inspect each model ----
for fname in tflite_files:
    size_kb = os.path.getsize(fname) / 1024
    print("=" * 72)
    print(f"{fname}  ({size_kb:.2f} KB)")
    print("=" * 72)
    try:
        interp = tf.lite.Interpreter(model_path=fname)
        interp.allocate_tensors()

        print("\nINPUTS:")
        for d in interp.get_input_details():
            print(f"  idx={d['index']:3d}  "
                  f"name='{d['name']}'  "
                  f"shape={list(d['shape'])}  "
                  f"dtype={d['dtype'].__name__}")

        print("\nOUTPUTS:")
        for d in interp.get_output_details():
            print(f"  idx={d['index']:3d}  "
                  f"name='{d['name']}'  "
                  f"shape={list(d['shape'])}  "
                  f"dtype={d['dtype'].__name__}")
        print()
    except Exception as e:
        print(f"  ❌ Failed to load: {e}\n")

print("=" * 72)
print("✅ Tensor inspection complete")
print("=" * 72)
