import os
import re
import glob
import numpy as np
import scipy.io as sio
from collections import Counter, defaultdict

# ============================================================
# PADERBORN DATASET DIAGNOSTIC
# ============================================================

DATASET_ROOT = "Paderborn"

WINDOW_SIZE = 1024
STRIDE = 1024
MAX_WINDOWS_PER_FILE = 50
TOTAL_FILES_TO_LOAD = 150

print("=" * 80)
print("PADERBORN DATASET DIAGNOSTIC")
print("=" * 80)

# ============================================================
# 1. FIND ALL MAT FILES
# ============================================================

files = glob.glob(
    os.path.join(DATASET_ROOT, "**", "*.mat"),
    recursive=True
)

print(f"\nTotal .mat files found: {len(files)}")

if len(files) == 0:
    print("ERROR: No .mat files found.")
    print(f"Check dataset path: {DATASET_ROOT}")
    raise SystemExit


# ============================================================
# 2. EXTRACT FAULT CODE FROM FILENAME
# ============================================================

def get_fault_code(filepath):
    filename = os.path.basename(filepath)

    # Paderborn filenames typically contain:
    # N09_M07_F10_KA01_10.mat
    #
    # Extract K001, KA01, KI01, KB23, etc.

    match = re.search(
        r'_(K\d{3}|KA\d{2}|KI\d{2}|KB\d{2})_',
        filename.upper()
    )

    if match:
        return match.group(1)

    # fallback: search anywhere
    match = re.search(
        r'(K\d{3}|KA\d{2}|KI\d{2}|KB\d{2})',
        filename.upper()
    )

    if match:
        return match.group(1)

    return "UNKNOWN"


# ============================================================
# 3. CURRENT CODE'S CLASSIFICATION
# ============================================================

def current_loader_class(filepath):

    path_lower = filepath.lower()

    if "healthy" in path_lower:
        return "Normal"

    elif "artificial" in path_lower:
        if "ka" in path_lower:
            return "OuterRace"
        else:
            return "InnerRace"

    elif "time" in path_lower:
        if "ki" in path_lower:
            return "InnerRace"
        else:
            return "OuterRace"

    return "Normal"


# ============================================================
# 4. MORE EXPLICIT FAULT-CODE CLASSIFICATION
# ============================================================

def explicit_fault_category(fault_code):

    # Healthy bearings:
    if fault_code.startswith("K0"):
        return "Healthy"

    # Artificially damaged bearings
    if fault_code.startswith("KA"):
        return "Artificial - KA"

    if fault_code.startswith("KI"):
        return "Artificial - KI"

    if fault_code.startswith("KB"):
        return "Artificial - KB"

    return "UNKNOWN"


# ============================================================
# 5. COUNT FILES BY FAULT CODE
# ============================================================

fault_files = defaultdict(list)

for f in files:
    fault = get_fault_code(f)
    fault_files[fault].append(f)

print("\n" + "=" * 80)
print("FILES PER FAULT CODE")
print("=" * 80)

for fault in sorted(fault_files):
    print(f"{fault:10s} {len(fault_files[fault]):5d}")

print("-" * 80)
print("Unique fault codes:", len(fault_files))
print("Total files:", sum(len(v) for v in fault_files.values()))


# ============================================================
# 6. COUNT BY TOP-LEVEL DIRECTORY
# ============================================================

top_level_counts = Counter()

for f in files:

    rel = os.path.relpath(f, DATASET_ROOT)
    parts = rel.split(os.sep)

    if len(parts) > 0:
        top_level_counts[parts[0].lower()] += 1
    else:
        top_level_counts["UNKNOWN"] += 1

print("\n" + "=" * 80)
print("FILES BY TOP-LEVEL DIRECTORY")
print("=" * 80)

for category, count in sorted(top_level_counts.items()):
    print(f"{category:15s} {count:5d}")


# ============================================================
# 7. CURRENT CLASS ASSIGNMENT BY FAULT
# ============================================================

print("\n" + "=" * 80)
print("CURRENT LOADER CLASSIFICATION")
print("=" * 80)

fault_class_map = {}

for fault in sorted(fault_files):

    classes = Counter(
        current_loader_class(f)
        for f in fault_files[fault]
    )

    fault_class_map[fault] = classes

    print(
        f"{fault:10s} -> "
        + ", ".join(
            f"{cls}: {count}"
            for cls, count in classes.items()
        )
    )


# ============================================================
# 8. FLAG SUSPICIOUS CLASSIFICATIONS
# ============================================================

print("\n" + "=" * 80)
print("POTENTIAL CLASSIFICATION PROBLEMS")
print("=" * 80)

problems_found = False

for fault in sorted(fault_files):

    assigned = set(fault_class_map[fault].keys())

    # K001 etc. should be Normal
    if fault.startswith("K0") and assigned != {"Normal"}:
        print(
            f"WARNING: {fault} is a healthy fault code "
            f"but current loader gives {assigned}"
        )
        problems_found = True

    # KI should be InnerRace
    if fault.startswith("KI") and assigned != {"InnerRace"}:
        print(
            f"WARNING: {fault} is a KI fault but "
            f"current loader gives {assigned}"
        )
        problems_found = True

    # KA should be OuterRace
    if fault.startswith("KA") and assigned != {"OuterRace"}:
        print(
            f"WARNING: {fault} is a KA fault but "
            f"current loader gives {assigned}"
        )
        problems_found = True

    # KB is especially important
    if fault.startswith("KB"):
        print(
            f"IMPORTANT: {fault} is classified by the current code as "
            f"{assigned}. This needs to be verified."
        )
        problems_found = True

if not problems_found:
    print("No obvious classification problems detected.")


# ============================================================
# 9. CLASS DISTRIBUTION
# ============================================================

class_counter = Counter()

for f in files:
    cls = current_loader_class(f)
    class_counter[cls] += 1

print("\n" + "=" * 80)
print("CURRENT CLASS DISTRIBUTION")
print("=" * 80)

total = len(files)

for cls in ["Normal", "InnerRace", "OuterRace"]:

    count = class_counter[cls]

    percentage = (
        100.0 * count / total
        if total > 0 else 0
    )

    print(
        f"{cls:12s}: "
        f"{count:5d} files "
        f"({percentage:6.2f}%)"
    )


# ============================================================
# 10. DIRECTORY DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("FAULT CODE -> DIRECTORY")
print("=" * 80)

for fault in sorted(fault_files):

    dirs = Counter()

    for f in fault_files[fault]:

        rel = os.path.relpath(f, DATASET_ROOT)
        parts = rel.split(os.sep)

        if len(parts) > 1:
            dirs[parts[0]] += 1

    print(
        f"{fault:10s}: "
        + ", ".join(
            f"{d}={n}"
            for d, n in dirs.items()
        )
    )


# ============================================================
# 11. WINDOW COUNTS
# ============================================================

print("\n" + "=" * 80)
print("WINDOW ANALYSIS")
print("=" * 80)

print(f"Window size: {WINDOW_SIZE}")
print(f"Stride:      {STRIDE}")
print(f"Max windows/file: {MAX_WINDOWS_PER_FILE}")

print(
    "\nNOTE: Window counts below require reading each MAT file."
)


# ============================================================
# 12. MAT FILE SIGNAL INSPECTION
# ============================================================

def extract_signals(mat_path):

    try:

        mat = sio.loadmat(mat_path)

        var_name = None

        for key in mat.keys():

            if key.startswith("N") and not key.startswith("__"):

                var_name = key
                break

        if var_name is None:
            return None

        data_struct = mat[var_name]

        if data_struct.shape == (1, 1):
            data_struct = data_struct[0, 0]

        signals = {
            "vibration": None,
            "current_1": None,
            "current_2": None,
            "temperature": None
        }

        if (
            hasattr(data_struct, "dtype")
            and data_struct.dtype.names
            and "Y" in data_struct.dtype.names
        ):

            Y = data_struct["Y"]

            if Y.shape == (1, 7):

                for i in range(Y.shape[1]):

                    item = Y[0, i]

                    try:
                        name = item["Name"][0]
                    except Exception:
                        continue

                    try:
                        data_arr = item["Data"]
                    except Exception:
                        continue

                    if data_arr is None or data_arr.size == 0:
                        continue

                    signal = data_arr.flatten().astype(np.float32)

                    if name == "vibration_1":
                        signals["vibration"] = signal

                    elif name == "phase_current_1":
                        signals["current_1"] = signal

                    elif name == "phase_current_2":
                        signals["current_2"] = signal

                    elif name == "temp_2_bearing_module":
                        signals["temperature"] = signal

        return signals

    except Exception as e:

        return None


# ============================================================
# 13. SAMPLE EVERY FAULT CODE
# ============================================================

print("\n" + "=" * 80)
print("SIGNAL INSPECTION: ONE FILE PER FAULT")
print("=" * 80)

sample_signal_info = {}

for fault in sorted(fault_files):

    f = fault_files[fault][0]

    signals = extract_signals(f)

    print(f"\n{fault}: {os.path.basename(f)}")

    if signals is None:
        print("  ERROR: Could not read MAT structure.")
        continue

    sample_signal_info[fault] = signals

    for name, signal in signals.items():

        if signal is None:
            print(f"  {name:15s}: MISSING")

        else:
            print(
                f"  {name:15s}: "
                f"length={len(signal):8d}, "
                f"mean={np.mean(signal): .6f}, "
                f"std={np.std(signal): .6f}"
            )


# ============================================================
# 14. CHECK CHANNEL LENGTH CONSISTENCY
# ============================================================

print("\n" + "=" * 80)
print("CHANNEL LENGTH CONSISTENCY")
print("=" * 80)

length_problems = []

for fault in sorted(fault_files):

    f = fault_files[fault][0]

    signals = extract_signals(f)

    if signals is None:
        continue

    lengths = {
        name: len(signal)
        for name, signal in signals.items()
        if signal is not None
    }

    unique_lengths = set(lengths.values())

    if len(unique_lengths) > 1:

        print(
            f"WARNING {fault}: channel lengths differ -> "
            f"{lengths}"
        )

        length_problems.append((fault, lengths))


if not length_problems:
    print("No channel-length inconsistencies found in sampled files.")


# ============================================================
# 15. CALCULATE WINDOW COUNTS FOR ALL FILES
# ============================================================

window_stats = []

print("\nScanning all files for window counts...")

for index, f in enumerate(files, start=1):

    fault = get_fault_code(f)

    signals = extract_signals(f)

    if signals is None or signals["vibration"] is None:
        continue

    vib_len = len(signals["vibration"])

    if vib_len < WINDOW_SIZE:

        total_windows = 0

    else:

        total_windows = (
            (vib_len - WINDOW_SIZE)
            // STRIDE
            + 1
        )

    selected_windows = min(
        MAX_WINDOWS_PER_FILE,
        total_windows
    )

    window_stats.append({
        "file": f,
        "fault": fault,
        "class": current_loader_class(f),
        "length": vib_len,
        "total_windows": total_windows,
        "selected_windows": selected_windows
    })

    if index % 100 == 0:
        print(f"  Processed {index}/{len(files)} files...")


# ============================================================
# 16. WINDOW DISTRIBUTION BY FAULT
# ============================================================

print("\n" + "=" * 80)
print("WINDOW DISTRIBUTION BY FAULT")
print("=" * 80)

fault_window_stats = defaultdict(lambda: {
    "files": 0,
    "total_windows": 0,
    "selected_windows": 0
})

for item in window_stats:

    fault = item["fault"]

    fault_window_stats[fault]["files"] += 1
    fault_window_stats[fault]["total_windows"] += item["total_windows"]
    fault_window_stats[fault]["selected_windows"] += item["selected_windows"]


print(
    f"{'Fault':10s}"
    f"{'Files':>8s}"
    f"{'All windows':>15s}"
    f"{'Selected':>15s}"
)

print("-" * 55)

for fault in sorted(fault_window_stats):

    s = fault_window_stats[fault]

    print(
        f"{fault:10s}"
        f"{s['files']:8d}"
        f"{s['total_windows']:15d}"
        f"{s['selected_windows']:15d}"
    )


# ============================================================
# 17. WINDOW DISTRIBUTION BY CLASS
# ============================================================

class_window_stats = defaultdict(lambda: {
    "files": 0,
    "total_windows": 0,
    "selected_windows": 0
})

for item in window_stats:

    cls = item["class"]

    class_window_stats[cls]["files"] += 1
    class_window_stats[cls]["total_windows"] += item["total_windows"]
    class_window_stats[cls]["selected_windows"] += item["selected_windows"]


print("\n" + "=" * 80)
print("WINDOW DISTRIBUTION BY CLASS")
print("=" * 80)

for cls in ["Normal", "InnerRace", "OuterRace"]:

    s = class_window_stats[cls]

    print(
        f"{cls:12s}: "
        f"{s['files']:4d} files, "
        f"{s['total_windows']:8d} possible windows, "
        f"{s['selected_windows']:8d} selected windows"
    )


# ============================================================
# 18. SIMULATE YOUR load_balanced_subset(total_files=150)
# ============================================================

print("\n" + "=" * 80)
print("SIMULATION OF load_balanced_subset(total_files=150)")
print("=" * 80)

healthy_files = []
artificial_files = []
time_files = []

for f in files:

    path_lower = f.lower()

    if "healthy" in path_lower:
        healthy_files.append(f)

    elif "artificial" in path_lower:
        artificial_files.append(f)

    elif "time" in path_lower:
        time_files.append(f)


print(f"Healthy files:    {len(healthy_files)}")
print(f"Artificial files: {len(artificial_files)}")
print(f"Time files:       {len(time_files)}")

files_per_category = TOTAL_FILES_TO_LOAD // 3
remaining = TOTAL_FILES_TO_LOAD % 3

print(
    f"\nRequested total: {TOTAL_FILES_TO_LOAD}"
)
print(
    f"Base files/category: {files_per_category}"
)
print(
    f"Remaining: {remaining}"
)

selected_simulation = []

categories = [
    ("healthy", healthy_files),
    ("artificial", artificial_files),
    ("time", time_files)
]

for i, (category, category_files) in enumerate(categories):

    n_to_take = (
        files_per_category
        + (1 if i < remaining else 0)
    )

    n_to_take = min(
        n_to_take,
        len(category_files)
    )

    print(
        f"{category:12s}: requested={n_to_take:3d}, "
        f"available={len(category_files):4d}"
    )

    if n_to_take > 0:

        selected_simulation.extend(
            category_files[:n_to_take]
        )


print(
    f"\nACTUALLY SELECTED: "
    f"{len(selected_simulation)} files"
)


# ============================================================
# 19. CLASS DISTRIBUTION OF SIMULATED 150 FILES
# ============================================================

sim_classes = Counter(
    current_loader_class(f)
    for f in selected_simulation
)

print("\nClass distribution of those files:")

for cls in ["Normal", "InnerRace", "OuterRace"]:

    print(
        f"  {cls:12s}: "
        f"{sim_classes[cls]:4d}"
    )


# ============================================================
# 20. ESTIMATED WINDOWS FROM SIMULATED DATA
# ============================================================

sim_windows = Counter()

for f in selected_simulation:

    fault = get_fault_code(f)
    cls = current_loader_class(f)

    matching = [
        x for x in window_stats
        if x["file"] == f
    ]

    if matching:

        selected = matching[0]["selected_windows"]

        sim_windows[cls] += selected


print("\nSelected windows from simulated subset:")

for cls in ["Normal", "InnerRace", "OuterRace"]:

    print(
        f"  {cls:12s}: "
        f"{sim_windows[cls]:6d}"
    )


# ============================================================
# 21. IMPORTANT: CHECK EWS IMPLEMENTATION
# ============================================================

print("\n" + "=" * 80)
print("EWS FEATURE CHECK")
print("=" * 80)

print(
    """
Your loader currently contains:

    feats = self.extractor.ews_features(vib_win)

This means EWS features are calculated from vibration ONLY.

The four-channel input is:

    [vibration, current_1, current_2, temperature]

but the EWS extractor receives only:

    vibration

Therefore, if your training pipeline was intended to calculate
EWS features from all four channels, the current loader does NOT
do that.

This diagnostic does not change that code; it only reports it.
"""
)


# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL DIAGNOSTIC SUMMARY")
print("=" * 80)

print(f"\nTotal files: {len(files)}")
print(f"Unique fault codes: {len(fault_files)}")

print("\nFiles by current class:")

for cls in ["Normal", "InnerRace", "OuterRace"]:

    count = class_counter[cls]

    percentage = (
        100 * count / len(files)
        if files else 0
    )

    print(
        f"  {cls:12s}: "
        f"{count:5d} "
        f"({percentage:6.2f}%)"
    )

print("\nFiles by directory:")

for category, count in sorted(top_level_counts.items()):

    print(
        f"  {category:12s}: {count:5d}"
    )

print("\nWindow totals:")

for cls in ["Normal", "InnerRace", "OuterRace"]:

    s = class_window_stats[cls]

    print(
        f"  {cls:12s}: "
        f"{s['total_windows']:8d} possible / "
        f"{s['selected_windows']:8d} selected"
    )

print("\n150-file loader simulation:")
print(f"  Selected files: {len(selected_simulation)}")

for cls in ["Normal", "InnerRace", "OuterRace"]:

    print(
        f"  {cls:12s}: "
        f"{sim_classes[cls]:4d} files"
    )

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)
