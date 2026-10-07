#!/usr/bin/env python3
# ============================================================
# MT-SUPINN DEPLOYMENT - FULL DATASET SUPPORT (FIXED v3)
# ============================================================
#
# This deployment supports:
#   - CWRU: 3-channel (DE, FE, BA) -> padded to 4 channels
#   - Paderborn: 4-channel (vibration, current1, current2, temp)
#   - Pronostia: 2-channel (horizontal, vertical)
#
# CRITICAL FIXES APPLIED:
#   1. EWS for Paderborn uses VIBRATION ONLY (matches training)
#   2. EWS for CWRU/Pronostia uses ALL channels (matches training)
#   3. Pronostia padding uses 2 channels (matches training)
#   4. Output tensor identification by name
#   5. NEW: Pronostia predictions now include 'bearing' column
#      and a per-bearing 'window_start' offset that matches the
#      file-by-file ground truth used for RMSE/MAE evaluation.
#
# Run: python3 deploy_mt_supinn.py --dataset pronostia
# ============================================================

import os
import sys
import glob
import time
import ctypes
import argparse
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import scipy.io as sio

# Try to import TensorFlow
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
    print(f"✅ TensorFlow version: {tf.__version__}")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("❌ TensorFlow not available. Please install: pip3 install tensorflow")
    sys.exit(1)

# Try to import joblib for scaler loading
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    print("⚠️ joblib not available. Install: pip3 install joblib")
    sys.exit(1)


# ============================================================
# PRONOSTIA BEARING HELPER  (NEW)
# ============================================================

def extract_pronostia_bearing(file_path: str) -> str:
    """
    Extract the Pronostia bearing name from a file path.

    Pronostia files are stored as:
        .../Full_Test_Set/Bearing1_3/acc_00001.csv
        .../Learning_set/Bearing1_1/acc_00001.csv

    The bearing name is the parent directory of the file.
    """
    return os.path.basename(os.path.dirname(file_path))


# ============================================================
# CONFIGURATION
# ============================================================

class DeployConfig:
    """Configuration for deployment"""
    
    def __init__(self):
        self.window_size = 1024
        self.ews_features = 11
        self.model_path = "mt_supinn_lightweight.tflite"
        self.scaler_path = "ews_scaler.pkl"
        
        # Custom TFLite library paths
        self.flex_library_path = "libtensorflowlite_flex.so"
        self.core_library_path = "libtensorflowlite.so"
        
        self.threshold_warning = 0.10
        self.threshold_critical = 0.05
        
        self.results_dir = "results"
        self.alert_file = os.path.join(self.results_dir, "alerts.log")
        
        self.dataset_paths = {
            'cwru': "csv_files/",
            'paderborn': "Paderborn/",
            'pronostia': "FEMTOBearingDataSet/Full_Test_Set/"
        }
        
        os.makedirs(self.results_dir, exist_ok=True)
    
    def get_dataset_path(self, dataset_type: str) -> str:
        return self.dataset_paths.get(dataset_type, "")


# ============================================================
# CUSTOM TFLITE LIBRARY LOADER
# ============================================================

def load_custom_tflite_libraries(flex_path: str, core_path: str) -> bool:
    """Load custom TensorFlow Lite libraries with Flex delegate support."""
    print("\n📚 Loading custom TensorFlow Lite libraries...")
    
    flex_path = os.path.abspath(flex_path)
    core_path = os.path.abspath(core_path)
    
    if not os.path.exists(flex_path):
        print(f"❌ Flex library not found: {flex_path}")
        print("   Please copy libtensorflowlite_flex.so to this directory.")
        return False
    
    if not os.path.exists(core_path):
        print(f"❌ Core library not found: {core_path}")
        print("   Please copy libtensorflowlite.so to this directory.")
        return False
    
    try:
        print(f"   Loading: {flex_path}")
        flex_lib = ctypes.CDLL(flex_path, mode=ctypes.RTLD_GLOBAL)
        
        print(f"   Loading: {core_path}")
        core_lib = ctypes.CDLL(core_path, mode=ctypes.RTLD_GLOBAL)
        
        print("✅ Custom TensorFlow Lite libraries loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load custom libraries: {e}")
        return False


# ============================================================
# EWS FEATURE EXTRACTION
# ============================================================

from scipy.stats import skew, kurtosis

def compute_ews_features(signal: np.ndarray) -> np.ndarray:
    """Compute 11 EWS features. Matches training FeatureExtractor.ews_features."""
    eps = 1e-10
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)

    if len(signal) == 0:
        return np.zeros(11, dtype=np.float32)

    rms = np.sqrt(np.mean(signal ** 2) + eps)
    peak = np.max(np.abs(signal))
    crest = peak / (rms + eps)
    std_val = np.std(signal)

    if np.std(signal) < eps:
        skew_val, kurt_val = 0.0, -3.0
    else:
        skew_val = skew(signal, bias=False, nan_policy='omit')
        kurt_val = kurtosis(signal, fisher=True, bias=False, nan_policy='omit')

    peak_to_peak = np.ptp(signal)
    mean_abs = np.mean(np.abs(signal)) + eps
    shape_factor = rms / mean_abs
    impulse_factor = peak / mean_abs
    mean_sqrt_abs = np.mean(np.sqrt(np.abs(signal))) + eps
    clearance_factor = peak / (mean_sqrt_abs ** 2 + eps)

    if len(signal) > 2:
        autocorr = np.sum(signal[:-1] * signal[1:]) / (np.sum(signal ** 2) + eps)
    else:
        autocorr = 0.0

    return np.array([
        rms, peak, crest, std_val, skew_val, kurt_val,
        peak_to_peak, shape_factor, impulse_factor,
        clearance_factor, autocorr
    ], dtype=np.float32)


def compute_ews_from_multichannel(signal: np.ndarray) -> np.ndarray:
    """Compute EWS features from multi-channel signal."""
    if signal.ndim == 1:
        return compute_ews_features(signal)
    
    n_channels = signal.shape[1]
    all_feats = []
    for ch in range(n_channels):
        all_feats.append(compute_ews_features(signal[:, ch]))
    
    return np.mean(all_feats, axis=0)


# ============================================================
# SCALER LOADING
# ============================================================

def load_scaler(scaler_path: str):
    """Load the EWS scaler from training"""
    if not JOBLIB_AVAILABLE:
        print("❌ joblib not available.")
        return None
    
    if not os.path.exists(scaler_path):
        print(f"❌ Scaler file not found: {scaler_path}")
        return None
    
    try:
        scaler = joblib.load(scaler_path)
        print(f"✅ Loaded scaler from: {scaler_path}")
        print(f"   Mean shape: {scaler.mean_.shape}")
        print(f"   Std shape:  {scaler.scale_.shape}")
        return scaler
    except Exception as e:
        print(f"❌ Failed to load scaler: {e}")
        return None


# ============================================================
# DATASET READER
# ============================================================

class DatasetReader:
    def __init__(self, config: DeployConfig):
        self.config = config
    
    def load_files(self, dataset_type: str) -> List[str]:
        data_path = self.config.get_dataset_path(dataset_type)
        
        if not data_path or not os.path.exists(data_path):
            print(f"⚠️ Dataset path not found: {data_path}")
            return []
        
        if dataset_type == "paderborn":
            pattern = os.path.join(data_path, "**", "*.mat")
            files = glob.glob(pattern, recursive=True)
        else:
            pattern = os.path.join(data_path, "**", "*.csv")
            files = glob.glob(pattern, recursive=True)
        
        files.sort()
        print(f"  Found {len(files)} files in {dataset_type}")
        return files
    
    def read_cwru(self, file_path: str) -> Optional[np.ndarray]:
        try:
            df = pd.read_csv(file_path)
            cols = df.columns.tolist()
            
            de_col = next((c for c in cols if 'DE' in c.upper()), None)
            fe_col = next((c for c in cols if 'FE' in c.upper()), None)
            ba_col = next((c for c in cols if 'BA' in c.upper()), None)
            
            if de_col is None or fe_col is None:
                raise ValueError(f"Missing DE or FE channel in {file_path}")
            
            de = pd.to_numeric(df[de_col], errors='coerce').values.astype(np.float32)
            fe = pd.to_numeric(df[fe_col], errors='coerce').values.astype(np.float32)
            
            if ba_col is not None:
                ba = pd.to_numeric(df[ba_col], errors='coerce').values.astype(np.float32)
                min_len = min(len(de), len(fe), len(ba))
                data = np.stack([de[:min_len], fe[:min_len], ba[:min_len]], axis=1)
            else:
                min_len = min(len(de), len(fe))
                data = np.stack([de[:min_len], fe[:min_len]], axis=1)
            
            return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as e:
            print(f"    Error reading CWRU: {e}")
            return None
    
    def read_pronostia(self, file_path: str) -> Optional[np.ndarray]:
        try:
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
            
            delimiter = ';' if ';' in first_line else ','
            df = pd.read_csv(file_path, header=None, delimiter=delimiter)
            data = df.values.astype(np.float32)
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            
            if data.shape[1] >= 6:
                vib_h = data[:, 4]
                vib_v = data[:, 5]
                signal = np.stack([vib_h, vib_v], axis=1)
            elif data.shape[1] >= 2:
                signal = np.stack([data[:, 0], data[:, 1]], axis=1)
            else:
                signal = np.stack([data[:, 0], data[:, 0]], axis=1)
            
            return np.asarray(signal, dtype=np.float32)
        except Exception as e:
            print(f"    Error reading Pronostia: {e}")
            return None
    
    def read_paderborn(self, file_path: str) -> Optional[np.ndarray]:
        try:
            mat = sio.loadmat(file_path)
            
            var_name = None
            for key in mat.keys():
                if key.startswith('N') and not key.startswith('__'):
                    var_name = key
                    break
            
            if var_name is None:
                print(f"    WARNING: no Paderborn data variable found.")
                return None
            
            data_struct = mat[var_name]
            if data_struct.shape == (1, 1):
                data_struct = data_struct[0, 0]
            
            vibration = None
            current_1 = None
            current_2 = None
            temperature = None
            
            if hasattr(data_struct, 'dtype') and 'Y' in data_struct.dtype.names:
                Y = data_struct['Y']
                n_items = Y.shape[1] if Y.ndim == 2 else Y.size
                
                for i in range(n_items):
                    try:
                        item = Y[0, i] if Y.ndim == 2 else Y[i]
                    except Exception:
                        continue
                    
                    if not hasattr(item, 'dtype') or not item.dtype.names:
                        continue
                    
                    name = None
                    if 'Name' in item.dtype.names:
                        try:
                            raw_name = item['Name']
                            while isinstance(raw_name, np.ndarray) and raw_name.size == 1:
                                raw_name = raw_name.item()
                            name = str(raw_name).strip()
                        except Exception:
                            name = None
                    
                    if 'Data' not in item.dtype.names:
                        continue
                    
                    try:
                        data_arr = item['Data']
                        if data_arr.size == 0:
                            continue
                        signal = np.asarray(data_arr, dtype=np.float32).reshape(-1)
                    except Exception:
                        continue
                    
                    if name == 'vibration_1':
                        vibration = signal
                    elif name == 'phase_current_1':
                        current_1 = signal
                    elif name == 'phase_current_2':
                        current_2 = signal
                    elif name == 'temp_2_bearing_module':
                        temperature = signal
            
            if vibration is None:
                if hasattr(data_struct, 'dtype') and 'X' in data_struct.dtype.names:
                    X = data_struct['X']
                    n_items = X.shape[1] if X.ndim == 2 else X.size
                    for i in range(n_items):
                        try:
                            item = X[0, i] if X.ndim == 2 else X[i]
                        except Exception:
                            continue
                        if not hasattr(item, 'dtype') or not item.dtype.names:
                            continue
                        if 'Data' not in item.dtype.names:
                            continue
                        try:
                            data_arr = item['Data']
                            if data_arr.size == 0:
                                continue
                            vibration = np.asarray(data_arr, dtype=np.float32).reshape(-1)
                            break
                        except Exception:
                            continue
            
            if vibration is None:
                print(f"    WARNING: vibration_1 not found in {os.path.basename(file_path)}")
                return None
            
            vibration = np.asarray(vibration, dtype=np.float32)
            vib_len = len(vibration)
            
            if vib_len == 0:
                return None
            
            def align_channel(channel, target_length):
                if channel is None or len(channel) == 0:
                    return np.zeros(target_length, dtype=np.float32)
                channel = np.asarray(channel, dtype=np.float32).reshape(-1)
                if len(channel) == target_length:
                    return channel
                if len(channel) == 1:
                    return np.full(target_length, channel[0], dtype=np.float32)
                return np.interp(
                    np.linspace(0.0, 1.0, target_length),
                    np.linspace(0.0, 1.0, len(channel)),
                    channel
                ).astype(np.float32)
            
            current_1 = align_channel(current_1, vib_len)
            current_2 = align_channel(current_2, vib_len)
            temperature = align_channel(temperature, vib_len)
            
            data = np.stack([vibration, current_1, current_2, temperature], axis=1)
            return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        except Exception as e:
            print(f"    Error reading Paderborn {os.path.basename(file_path)}: {e}")
            return None


# ============================================================
# MODEL INFERENCE
# ============================================================

class ModelInference:
    def __init__(self, model_path: str, scaler=None, flex_library_path=None, core_library_path=None):
        self.model_path = model_path
        self.scaler = scaler
        self.flex_library_path = flex_library_path
        self.core_library_path = core_library_path
        self.interpreter = None
        
        self.vib_input_idx = None
        self.ews_input_idx = None
        
        self.cwru_output_idx = None
        self.paderborn_output_idx = None
        self.rul_output_idx = None
        
        self.inference_times = []
    
    def load(self) -> bool:
        if not TENSORFLOW_AVAILABLE:
            print("❌ TensorFlow not available")
            return False
        
        if not os.path.exists(self.model_path):
            print(f"❌ Model not found: {self.model_path}")
            return False
        
        print(f"\n📦 Loading model: {self.model_path}")
        
        if self.flex_library_path and self.core_library_path:
            if not load_custom_tflite_libraries(self.flex_library_path, self.core_library_path):
                print("⚠️ Custom library loading failed. Trying standard interpreter...")
        
        try:
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f"   {len(self.input_details)} inputs, {len(self.output_details)} outputs")
        
        self._print_tensor_details()
        self._identify_inputs()
        self._identify_outputs()
        
        return True
    
    def _print_tensor_details(self):
        print("\n📥 Input Tensors:")
        for i, d in enumerate(self.input_details):
            print(f"  [{i}] idx={d['index']}, name='{d['name']}', shape={d['shape']}, dtype={d['dtype']}")
        
        print("\n📤 Output Tensors:")
        for i, d in enumerate(self.output_details):
            print(f"  [{i}] idx={d['index']}, name='{d['name']}', shape={d['shape']}, dtype={d['dtype']}")
    
    def _identify_inputs(self):
        for detail in self.input_details:
            shape = list(detail['shape'])
            if shape == [1, 1024, 4]:
                self.vib_input_idx = detail['index']
                print(f"  ✅ Vibration input: index={self.vib_input_idx}")
            elif shape == [1, 11]:
                self.ews_input_idx = detail['index']
                print(f"  ✅ EWS input: index={self.ews_input_idx}")
        
        if self.vib_input_idx is None:
            raise RuntimeError("Could not find vibration input [1,1024,4]")
        if self.ews_input_idx is None:
            raise RuntimeError("Could not find EWS input [1,11]")
    
    def _identify_outputs(self):
        print("\n📤 Identifying outputs...")
        
        for detail in self.output_details:
            name = detail['name'].lower()
            if 'cwru' in name:
                self.cwru_output_idx = detail['index']
                print(f"  ✅ CWRU: idx={self.cwru_output_idx} (name='{detail['name']}')")
            elif 'paderborn' in name:
                self.paderborn_output_idx = detail['index']
                print(f"  ✅ Paderborn: idx={self.paderborn_output_idx} (name='{detail['name']}')")
            elif 'rul' in name or 'pronostia' in name:
                self.rul_output_idx = detail['index']
                print(f"  ✅ RUL: idx={self.rul_output_idx} (name='{detail['name']}')")
        
        if self.cwru_output_idx is None:
            for detail in self.output_details:
                if list(detail['shape']) == [1, 4]:
                    self.cwru_output_idx = detail['index']
                    print(f"  ⚠️ CWRU fallback: idx={self.cwru_output_idx} (shape [1,4])")
                    break
        
        if self.paderborn_output_idx is None:
            for detail in self.output_details:
                if list(detail['shape']) == [1, 3]:
                    self.paderborn_output_idx = detail['index']
                    print(f"  ⚠️ Paderborn fallback: idx={self.paderborn_output_idx} (shape [1,3])")
        
        if self.rul_output_idx is None:
            print("  🔍 Searching for RUL tensor by name pattern...")
            for detail in self.output_details:
                if list(detail['shape']) == [1, 1]:
                    if 'StatefulPartitionedCall_1:4' in detail['name']:
                        self.rul_output_idx = detail['index']
                        print(f"  ✅ RUL: idx={self.rul_output_idx} (name='{detail['name']}')")
                        break
            
            if self.rul_output_idx is None:
                candidates = []
                for detail in self.output_details:
                    if list(detail['shape']) == [1, 1]:
                        candidates.append((detail['index'], detail['name']))
                        print(f"  ⚠️ Candidate: idx={detail['index']}, name='{detail['name']}'")
                if candidates:
                    candidates.sort(key=lambda x: x[0])
                    self.rul_output_idx = candidates[0][0]
                    print(f"  ⚠️ RUL fallback: idx={self.rul_output_idx} (first [1,1])")
        
        has_cwru_head = self.cwru_output_idx is not None
        has_pader_head = self.paderborn_output_idx is not None
        has_rul_head = self.rul_output_idx is not None

        if not (has_cwru_head or has_pader_head or has_rul_head):
            raise RuntimeError(
                "Model has no recognized output head. "
                "Expected at least one of: cwru_class, paderborn_class, pronostia_rul."
            )

        print(f"\n  Heads present in this model: "
              f"CWRU={has_cwru_head}, Paderborn={has_pader_head}, RUL={has_rul_head}")
        
        print("\n📊 Final Output Mapping:")
        print(f"  CWRU:      idx={self.cwru_output_idx}")
        print(f"  Paderborn: idx={self.paderborn_output_idx}")
        print(f"  RUL:       idx={self.rul_output_idx}")
    
    def _prepare_vibration_input(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=np.float32)
        
        if data.ndim == 1:
            if len(data) != 1024:
                raise ValueError(f"Expected 1024 samples, got {len(data)}")
            padded = np.zeros((1024, 4), dtype=np.float32)
            padded[:, 0] = data
            padded[:, 1] = data
            return padded.reshape(1, 1024, 4)
        
        if data.ndim == 2:
            if data.shape[0] != 1024:
                raise ValueError(f"Expected 1024 samples, got {data.shape[0]}")
            padded = np.zeros((1024, 4), dtype=np.float32)
            n_channels = min(data.shape[1], 4)
            padded[:, :n_channels] = data[:, :n_channels]
            return padded.reshape(1, 1024, 4)
        
        raise ValueError(f"Unsupported input shape: {data.shape}")
    
    def predict(self, window: np.ndarray, dataset_type: str = None) -> Dict[str, Any]:
        if self.interpreter is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        window = np.asarray(window, dtype=np.float32)
        
        if dataset_type == "paderborn":
            vibration = window[:, 0] if window.ndim == 2 else window
            ews_features = compute_ews_features(vibration).reshape(1, 11)
        else:
            ews_features = compute_ews_from_multichannel(window).reshape(1, 11)
        
        if self.scaler is None:
            raise RuntimeError("Scaler is required but was not loaded.")
        
        try:
            ews_features = ews_features.astype(np.float32)
            ews_features = self.scaler.transform(ews_features)
            ews_features = ews_features.astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Scaler transform failed: {e}")
        
        vib_input = self._prepare_vibration_input(window)
        
        self.interpreter.set_tensor(self.vib_input_idx, vib_input)
        self.interpreter.set_tensor(self.ews_input_idx, ews_features)
        
        start_time = time.perf_counter()
        self.interpreter.invoke()
        inference_time_ms = (time.perf_counter() - start_time) * 1000.0
        self.inference_times.append(inference_time_ms)
        
        result = {
            'cwru_class': None,
            'cwru_probabilities': None,
            'paderborn_class': None,
            'paderborn_probabilities': None,
            'rul': None,
            'inference_time_ms': inference_time_ms
        }
        
        if self.cwru_output_idx is not None:
            raw = self.interpreter.get_tensor(self.cwru_output_idx)
            probs = np.asarray(raw, dtype=np.float32).reshape(-1)
            result['cwru_probabilities'] = probs
            result['cwru_class'] = int(np.argmax(probs))
        
        if self.paderborn_output_idx is not None:
            raw = self.interpreter.get_tensor(self.paderborn_output_idx)
            probs = np.asarray(raw, dtype=np.float32).reshape(-1)
            result['paderborn_probabilities'] = probs
            result['paderborn_class'] = int(np.argmax(probs))
        
        if self.rul_output_idx is not None:
            raw = self.interpreter.get_tensor(self.rul_output_idx)
            result['rul'] = float(np.asarray(raw, dtype=np.float32).reshape(-1)[0])
        
        return result
    
    def get_average_inference_time(self) -> float:
        if not self.inference_times:
            return 0.0
        return np.mean(self.inference_times)


# ============================================================
# ALERT CHECKING
# ============================================================

def check_alerts(rul_value: float, warning_threshold: float = 0.10, 
                 critical_threshold: float = 0.05) -> Tuple[str, str]:
    if rul_value < critical_threshold:
        return "RED", "IMMEDIATE SHUTDOWN REQUIRED"
    elif rul_value < warning_threshold:
        return "ORANGE", "Plan maintenance within 24 hours"
    elif rul_value < warning_threshold * 2:
        return "YELLOW", "Schedule inspection"
    else:
        return "GREEN", "Continue monitoring"


# ============================================================
# PROCESSING FUNCTIONS
# ============================================================

def process_dataset(files: List[str], reader: DatasetReader, model: ModelInference,
                    config: DeployConfig, dataset_type: str, result_file: str,
                    max_files: Optional[int] = None, max_windows: Optional[int] = None):
    """Process a dataset and save predictions."""
    
    if max_files is not None:
        files_to_process = files[:max_files]
    else:
        files_to_process = files
    
    print(f"\n🔄 Processing {len(files_to_process)} files...")
    
    # ============================================================
    # NEW: per-bearing window offset for Pronostia
    # ============================================================
    # Each bearing's window_start restarts at 0 and increments by the
    # window stride (1024). This matches the file-by-file ground truth.
    bearing_window_offset = {}
    # ============================================================
    
    total_windows = 0
    successful_predictions = 0
    failed_predictions = 0
    all_cwru_predictions = []
    all_paderborn_predictions = []
    all_rul_values = []
    
    for file_idx, file_path in enumerate(files_to_process):
        filename = os.path.basename(file_path)
        print(f"\n  [{file_idx+1}/{len(files_to_process)}] {filename}")
        
        try:
            if dataset_type == "cwru":
                data = reader.read_cwru(file_path)
            elif dataset_type == "pronostia":
                data = reader.read_pronostia(file_path)
            elif dataset_type == "paderborn":
                data = reader.read_paderborn(file_path)
            else:
                continue
        except Exception as e:
            print(f"    ❌ Error reading file: {e}")
            failed_predictions += 1
            continue
        
        if data is None or len(data) == 0:
            print("    ⚠️ Empty data, skipping")
            continue
        
        window_size = config.window_size
        file_windows = 0
        
        for start in range(0, len(data) - window_size + 1, window_size):
            if max_windows is not None and file_windows >= max_windows:
                break
            
            window = data[start:start + window_size]
            
            try:
                result = model.predict(window, dataset_type=dataset_type)
                successful_predictions += 1
                total_windows += 1
                file_windows += 1
                
                if dataset_type == "cwru":
                    all_cwru_predictions.append({
                        'file': filename,
                        'start': start,
                        'class': result['cwru_class'],
                        'confidence': result['cwru_probabilities'][result['cwru_class']],
                        'probabilities': result['cwru_probabilities'].tolist(),
                        'time_ms': result['inference_time_ms']
                    })
                    print(f"    CWRU: class={result['cwru_class']}, conf={result['cwru_probabilities'][result['cwru_class']]:.4f}")
                    
                elif dataset_type == "paderborn":
                    all_paderborn_predictions.append({
                        'file': filename,
                        'start': start,
                        'class': result['paderborn_class'],
                        'confidence': result['paderborn_probabilities'][result['paderborn_class']],
                        'probabilities': result['paderborn_probabilities'].tolist(),
                        'time_ms': result['inference_time_ms']
                    })
                    print(f"    Paderborn: class={result['paderborn_class']}, conf={result['paderborn_probabilities'][result['paderborn_class']]:.4f}")
                    
                elif dataset_type == "pronostia":
                    rul_value = result['rul']
                    
                    # ============================================================
                    # NEW: record bearing + per-bearing window_start
                    # ============================================================
                    bearing_name = extract_pronostia_bearing(file_path)
                    
                    if bearing_name not in bearing_window_offset:
                        bearing_window_offset[bearing_name] = 0
                    global_window_start = bearing_window_offset[bearing_name]
                    bearing_window_offset[bearing_name] += window_size
                    # ============================================================
                    
                    all_rul_values.append({
                        'bearing': bearing_name,          # NEW
                        'file': filename,
                        'window_start': global_window_start,  # NEW (was 'start')
                        'rul': rul_value,
                        'time_ms': result['inference_time_ms']
                    })
                    
                    alert_level, action = check_alerts(
                        rul_value,
                        config.threshold_warning,
                        config.threshold_critical
                    )
                    
                    if alert_level != "GREEN":
                        with open(config.alert_file, 'a') as f:
                            f.write(f"{datetime.now().isoformat()}, {bearing_name}, "
                                   f"{filename}, {global_window_start}, {alert_level}, "
                                   f"{rul_value:.6f}, {action}\n")
                        print(f"    ⚠️ ALERT [{alert_level}] RUL={rul_value:.6f}")
                    else:
                        print(f"    RUL: {rul_value:.6f} ({result['inference_time_ms']:.2f}ms)")
                
            except Exception as e:
                failed_predictions += 1
                print(f"    ❌ Prediction failed at window {start}: {e}")
                continue
        
        print(f"    Processed {file_windows} windows")
    
    # Save results
    print("\n💾 Saving results...")
    
    if dataset_type == "cwru" and all_cwru_predictions:
        df = pd.DataFrame(all_cwru_predictions)
        df.to_csv(result_file, index=False)
        print(f"  ✅ Saved {len(df)} CWRU predictions")
    
    elif dataset_type == "paderborn" and all_paderborn_predictions:
        df = pd.DataFrame(all_paderborn_predictions)
        df.to_csv(result_file, index=False)
        print(f"  ✅ Saved {len(df)} Paderborn predictions")
    
    elif dataset_type == "pronostia" and all_rul_values:
        df = pd.DataFrame(all_rul_values)
        df.to_csv(result_file, index=False)
        print(f"  ✅ Saved {len(df)} Pronostia predictions")
        print(f"  ✅ Columns: {df.columns.tolist()}")
        # Per-bearing summary
        print(f"\n  📊 Windows per bearing:")
        for b, g in df.groupby('bearing'):
            print(f"     {b}: {len(g)} windows")
    
    # Summary
    print("\n" + "=" * 70)
    print(f"{dataset_type.upper()} PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Files processed: {len(files_to_process)}")
    print(f"Windows processed: {total_windows}")
    print(f"Successful predictions: {successful_predictions}")
    print(f"Failed predictions: {failed_predictions}")
    print(f"Results saved to: {result_file}")
    
    if model.inference_times:
        avg_time = model.get_average_inference_time()
        print(f"Average inference time: {avg_time:.2f}ms")
        if avg_time > 0:
            print(f"Frames per second: {1000/avg_time:.1f} FPS")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("MT-SUPINN DEPLOYMENT - FULL DATASET SUPPORT (FIXED v3)")
    print("=" * 70)
    
    parser = argparse.ArgumentParser(description="MT-SUPINN Deployment")
    parser.add_argument('--model', type=str, default="mt_supinn_lightweight.tflite",
                        help='Path to TFLite model')
    parser.add_argument('--scaler', type=str, default="ews_scaler.pkl",
                        help='Path to scaler file')
    parser.add_argument('--dataset', type=str, choices=['cwru', 'paderborn', 'pronostia'],
                        help='Dataset to process (if not provided, shows menu)')
    parser.add_argument('--output-dir', type=str, default="results",
                        help='Output directory for results')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Maximum number of files to process')
    parser.add_argument('--max-windows', type=int, default=None,
                        help='Maximum windows per file')
    args = parser.parse_args()
    
    config = DeployConfig()
    config.model_path = args.model
    config.scaler_path = args.scaler
    config.results_dir = args.output_dir
    os.makedirs(config.results_dir, exist_ok=True)
    
    print("\n📊 Loading EWS scaler...")
    scaler = load_scaler(config.scaler_path)
    if scaler is None:
        print("❌ Scaler is required. Exiting.")
        return
    
    model = ModelInference(
        config.model_path,
        scaler,
        flex_library_path=config.flex_library_path,
        core_library_path=config.core_library_path
    )
    if not model.load():
        print("❌ Failed to load model. Exiting.")
        return
    
    reader = DatasetReader(config)
    
    if args.dataset:
        dataset_type = args.dataset
    else:
        print("\n📂 Available datasets:")
        print("   1. CWRU")
        print("   2. Paderborn")
        print("   3. PRONOSTIA")
        choice = input("\nSelect dataset (1-3): ").strip()
        dataset_map = {"1": "cwru", "2": "paderborn", "3": "pronostia"}
        dataset_type = dataset_map.get(choice)
        
        if dataset_type is None:
            print("❌ Invalid selection.")
            return
    
    result_file = os.path.join(config.results_dir, f"{dataset_type}_predictions.csv")
    
    print(f"\n📂 Loading {dataset_type} files...")
    files = reader.load_files(dataset_type)
    
    if not files:
        print(f"❌ No files found for {dataset_type}")
        return
    
    process_dataset(
        files=files,
        reader=reader,
        model=model,
        config=config,
        dataset_type=dataset_type,
        result_file=result_file,
        max_files=args.max_files,
        max_windows=args.max_windows
    )
    
    print("\n✅ Deployment complete!")


if __name__ == "__main__":
    main()
