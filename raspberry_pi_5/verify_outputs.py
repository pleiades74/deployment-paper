#!/usr/bin/env python3
"""
verify_outputs.py — Determine which TFLite output corresponds to which head.
Uses fixed dummy inputs and prints all output values side by side.
"""
import os, ctypes
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

flex = os.path.join(SCRIPT_DIR, "libtensorflowlite_flex.so")
core = os.path.join(SCRIPT_DIR, "libtensorflowlite.so")
if os.path.exists(flex): ctypes.CDLL(flex, mode=ctypes.RTLD_GLOBAL)
if os.path.exists(core): ctypes.CDLL(core, mode=ctypes.RTLD_GLOBAL)

import tensorflow as tf

MODEL = "model_full_42_fp32.tflite"

interp = tf.lite.Interpreter(model_path=MODEL)
interp.allocate_tensors()

# Fixed dummy input
vib = np.zeros((1, 1024, 4), dtype=np.float32)
ews = np.zeros((1, 11), dtype=np.float32)

# Set inputs
input_details = interp.get_input_details()
for d in input_details:
    if list(d['shape']) == [1, 1024, 4]:
        interp.set_tensor(d['index'], vib)
    elif list(d['shape']) == [1, 11]:
        interp.set_tensor(d['index'], ews)

interp.invoke()

print(f"\nModel: {MODEL}")
print(f"Input: vib=all zeros, ews=all zeros")
print(f"{'tensor_name':35s} {'shape':15s} {'value(s)'}")
print("-" * 80)
for d in interp.get_output_details():
    out = interp.get_tensor(d['index'])
    flat = out.flatten()
    vals = ", ".join(f"{v:.4f}" for v in flat[:4])
    print(f"{d['name']:35s} {str(list(d['shape'])):15s} [{vals}]")

# Try another input: ews=1.0
ews2 = np.ones((1, 11), dtype=np.float32)
for d in input_details:
    if list(d['shape']) == [1, 11]:
        interp.set_tensor(d['index'], ews2)
interp.invoke()

print(f"\nInput: vib=all zeros, ews=all ones")
print(f"{'tensor_name':35s} {'shape':15s} {'value(s)'}")
print("-" * 80)
for d in interp.get_output_details():
    out = interp.get_tensor(d['index'])
    flat = out.flatten()
    vals = ", ".join(f"{v:.4f}" for v in flat[:4])
    print(f"{d['name']:35s} {str(list(d['shape'])):15s} [{vals}]")
