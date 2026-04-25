# Python Parallel SEISM Edge Evaluation – Benchmark Documentation

**File:** `edge_eval/evaluate_edges_seism_matlab_identical_parallel.py`
**Benchmark script:** `benchmark/scripts/benchmark_python_parallel.py`

> This document covers the **parallel Python** edge evaluation benchmark.
> It should be read alongside the serial Python documentation in
> `edge_eval/README_evaluate_edges_seism_matlab_identical.md`.

---

## Table of Contents

1. [Purpose](#1-purpose)
2. [Implementation Overview](#2-implementation-overview)
3. [Correctness / Equivalence](#3-correctness--equivalence)
4. [Fair-Comparison Methodology](#4-fair-comparison-methodology)
5. [Benchmark Commands](#5-benchmark-commands)
6. [Suggested Benchmark Procedure](#6-suggested-benchmark-procedure)
7. [Remaining Unavoidable Differences](#7-remaining-unavoidable-differences)
8. [Output Format Reference](#8-output-format-reference)

---

## 1. Purpose

### 1.1 What already existed

This project already had **serial benchmarking** for both evaluation environments:

| Environment | Script | Benchmark harness |
|---|---|---|
| **MATLAB (serial)** | `evaluation/seism/` (original) | `benchmark/scripts/benchmark_matlab.m` |
| **Python (serial)** | `edge_eval/evaluate_edges_seism_matlab_identical.py` | `benchmark/scripts/benchmark_python.py` |

### 1.2 What this work adds

This work adds **parallel benchmarking for Python** only:

| Environment | Script | Benchmark harness |
|---|---|---|
| **Python (parallel)** | `edge_eval/evaluate_edges_seism_matlab_identical_parallel.py` | `benchmark/scripts/benchmark_python_parallel.py` |

> **Note:** MATLAB parallel benchmarking (using `parfor`) already exists and is
> not part of this task.  This document covers the Python side only.

### 1.3 Main goal

Provide a **fair runtime comparison** between the MATLAB parallel implementation
and the Python parallel implementation, using the same datasets, the same
predictions, and the same evaluation algorithm, measured on the same machine.

---

## 2. Implementation Overview

### 2.1 Base script

The parallel script is derived **directly** from:

```
edge_eval/evaluate_edges_seism_matlab_identical.py
```

It imports all evaluation logic from that module:

```python
from edge_eval.evaluate_edges_seism_matlab_identical import (
    read_edge_map,
    load_gt_from_segmentation,
    compute_maxdist,
    evaluate_single_image,      # the core per-image evaluation
    compute_ods_matlab,
    compute_ois_matlab,
    compute_ap_matlab,
    ORTOOLS_AVAILABLE,
    USE_CSA_MATCHING,
)
```

No algorithm code is redefined. The parallel script is a thin orchestration
layer around the unchanged serial evaluation functions.

### 2.2 What was changed

Only **one structural change** was made relative to the serial script:

| Component | Serial | Parallel |
|---|---|---|
| Per-image loop | `for pred_file in pbar:` (sequential) | `ProcessPoolExecutor` + `as_completed` |
| Worker function | (inline loop body) | `_worker_evaluate_image()` (top-level) |
| Result aggregation | In-loop accumulation | Ordered post-collection + accumulation |
| Progress bar | tqdm inside the loop | tqdm in the main process |
| New CLI arg | — | `--workers N` |

The serial loop is preserved **verbatim as a comment block** inside
`evaluate_edge_maps_parallel()` for easy diff-based verification.

### 2.3 The serial section that was replaced

In the original script, the serial section is the `for` loop
(lines ~812–860 of `evaluate_edges_seism_matlab_identical.py`):

```python
# SERIAL (replaced):
pbar = tqdm(pred_files, desc="[MATLAB-IDENTICAL]", unit="img")
for pred_file in pbar:
    ...
    img_result = evaluate_single_image(pred, gt, thresholds, max_dist, ...)
    per_image_results.append(img_result)
    total_cntP += img_result['cntP']
    ...
```

### 2.4 Parallelisation strategy

**Strategy:** `concurrent.futures.ProcessPoolExecutor` with the `'spawn'`
multiprocessing context.

**Why processes, not threads?**

- CPython's Global Interpreter Lock (GIL) prevents true thread-level
  parallelism for CPU-bound code.
- Each image evaluation is CPU-intensive (99 thresholds × bipartite matching).
- OR-Tools (the CSA matching library) and NumPy release the GIL but their
  Python wrapper overhead still blocks threads.
- Spawning separate OS processes bypasses the GIL entirely.

**Why `'spawn'` context?**

- OR-Tools' C extensions and NumPy's internal state are not safe to fork
  on macOS and Windows.
- `'spawn'` creates a fresh interpreter in each worker, avoiding
  fork-safety issues at the cost of a slightly higher startup overhead.
- On Linux, `'fork'` is the system default but `'spawn'` is explicitly set
  here for cross-platform safety.

**Worker scheduling:**

```
Main process
  │
  ├── submit image_0 → worker_0
  ├── submit image_1 → worker_1
  ├── ...
  └── submit image_N → worker_K
           │
           └── Each worker runs evaluate_single_image()
                 (99 thresholds, thinning, CSA/Hungarian matching)
                 Returns: img_result dict
Main process
  └── Collects results via as_completed()
      Stores at original index → preserves order
      Accumulates cntP/sumP/cntR/sumR in order
      Computes ODS/OIS/AP (identical to serial)
```

**Chunking:** Each worker handles exactly one image (no manual chunking).
The OS process pool naturally load-balances across available CPUs.

### 2.5 Key constraints and assumptions

- Worker count defaults to `os.cpu_count()`. Set `--workers N` for control.
- The `thresholds` array is serialised as a Python list for pickling.
- `tqdm_write_func` is set to `None` inside workers (child processes cannot
  write to the parent's tqdm bar).
- `debug_timing=False` is forced in workers to avoid interleaved output.
- The `--debug_timing` flag of the serial script is **not available** in the
  parallel version (per-image stage logs would be interleaved and misleading).

---

## 3. Correctness / Equivalence

### 3.1 What is preserved

| Property | Preserved? | Notes |
|---|---|---|
| 99 SEISM thresholds | ✅ | `linspace(1/100, 99/100, 99)` |
| Morphological thinning | ✅ | `skimage.morphology.thin` |
| GT loading / seg2bmap | ✅ | Imported unchanged |
| CSA / Hungarian matching | ✅ | Imported unchanged |
| ODS / OIS / AP formulas | ✅ | Imported unchanged |
| Accumulation order | ✅ | `ordered_results` preserves input order |
| Output JSON schema | ✅ | Same keys + `n_workers` and `mode` extra |

### 3.2 What could cause tiny numerical differences

- **Float accumulation order:** Since workers finish in non-deterministic
  order but results are collected into `ordered_results[original_idx]` and
  then accumulated sequentially in input order, the accumulation order is
  **identical to the serial version**.  No floating-point differences are
  expected.
- If any difference >1e-6 is observed in ODS/OIS/AP between serial and
  parallel runs on the same image list, it indicates a bug.

### 3.3 How to verify

```bash
# Run serial
python edge_eval/evaluate_edges_seism_matlab_identical.py \
    --pred_dir outputs/PASCALContext/resnet18/single_task/edge/results/edge \
    --seg_dir  PASCAL_MT/pascal-context/trainval \
    --image_list_file benchmark/image_lists/small_10.txt \
    --out results/verify_serial_10.json

# Run parallel (same list)
python edge_eval/evaluate_edges_seism_matlab_identical_parallel.py \
    --pred_dir outputs/PASCALContext/resnet18/single_task/edge/results/edge \
    --seg_dir  PASCAL_MT/pascal-context/trainval \
    --image_list_file benchmark/image_lists/small_10.txt \
    --workers 4 \
    --out results/verify_parallel_10.json

# Compare ODS/OIS/AP
python - <<'EOF'
import json
s = json.load(open('results/verify_serial_10.json'))
p = json.load(open('results/verify_parallel_10.json'))
for k in ('ODS','OIS','AP'):
    diff = abs(s[k] - p[k])
    status = "✅ OK" if diff < 1e-6 else f"❌ DIFF={diff:.2e}"
    print(f"{k}: serial={s[k]:.6f}  parallel={p[k]:.6f}  {status}")
EOF
```

---

## 4. Fair-Comparison Methodology

This section documents how the Python parallel benchmark is designed to be
**fair** against the MATLAB parallel benchmark.

### 4.1 What is kept identical

| Factor | MATLAB | Python | Guaranteed equal? |
|---|---|---|---|
| Machine | Same host | Same host | ✅ Must be enforced |
| Dataset subset | `benchmark/image_lists/*.txt` | `--image_list_file` same `.txt` | ✅ |
| Prediction files | `pred_dir` PNG files | `--pred_dir` same directory | ✅ |
| GT files | `.mat` files in `trainval/` | `--seg_dir` same directory | ✅ |
| Number of images | N images | N images | ✅ |
| Threshold set | `linspace(0.01, 0.99, 99)` | Same formula | ✅ |
| Thinning | `bwmorph('thin', Inf)` | `skimage.morphology.thin` | ✅ (verified) |
| Matching tolerance | `0.0075 × diagonal` | `0.0075 × diagonal` | ✅ |
| Worker count | `parpool(N)` | `--workers N` | ✅ Must set same N |

### 4.2 Warm-up philosophy

- **MATLAB:** The first benchmark run often includes JIT compilation and MEX
  loading overhead. Recommended: run 1 warm-up + 3 timed runs; discard warm-up.
- **Python:** The first run includes worker process spawning, module import in
  each worker, and OR-Tools initialisation. Recommended: same 1 warm-up + 3
  timed runs.
- Both sides: use `--runs 3` (or equivalent) and report the mean ± std.

### 4.3 Timing scope

| Side | What is timed |
|---|---|
| MATLAB `parfor` | `tic`/`toc` around the `parfor` loop |
| Python parallel | `eval_time_seconds` field in output JSON (wraps `ProcessPoolExecutor` block) |
| Python `total_wall_s` | Includes pool startup/teardown |

> **Recommendation:** Compare `eval_time_seconds` (Python) vs. MATLAB's
> `parfor` wall time for the fairest comparison.  Also report `total_wall_s`
> for completeness.

### 4.4 Same subset definitions

The three canonical subsets used in the benchmark are defined by:

| Subset | File | N |
|---|---|---|
| Small | `benchmark/image_lists/small_10.txt` | 10 |
| Medium | `benchmark/image_lists/medium_100.txt` | 100 |
| Full | `benchmark/image_lists/full_5105.txt` | 5105 |

To ensure MATLAB uses the **exact same subset**, convert the `.txt` file to a
MATLAB cell array before the MATLAB run:

```matlab
% In MATLAB — load the same image list used by Python
fid = fopen('benchmark/image_lists/small_10.txt', 'r');
lines = textscan(fid, '%s', 'Delimiter', '\n');
fclose(fid);
image_names = lines{1};  % cell array of stems
% Pass image_names to your parfor evaluation loop
```

### 4.5 Same benchmark conditions

- Close all other CPU-intensive applications before benchmarking.
- Ensure both MATLAB and Python runs use the same number of workers.
- On a machine with `C` physical cores, set `--workers C` (Python) and
  `parpool(C)` (MATLAB).
- Do not mix hyper-threaded and physical core counts between the two sides.
- Use wall-clock time (not CPU time) for both sides.

---

## 5. Benchmark Commands

### 5.1 Python parallel — n=10

```bash
python edge_eval/evaluate_edges_seism_matlab_identical_parallel.py \
    --pred_dir        outputs/PASCALContext/resnet18/single_task/edge/results/edge \
    --seg_dir         PASCAL_MT/pascal-context/trainval \
    --image_list_file benchmark/image_lists/small_10.txt \
    --workers         4 \
    --out             results/parallel_eval_10.json
```

### 5.2 Python parallel — n=100

```bash
python edge_eval/evaluate_edges_seism_matlab_identical_parallel.py \
    --pred_dir        outputs/PASCALContext/resnet18/single_task/edge/results/edge \
    --seg_dir         PASCAL_MT/pascal-context/trainval \
    --image_list_file benchmark/image_lists/medium_100.txt \
    --workers         4 \
    --out             results/parallel_eval_100.json
```

### 5.3 Python parallel — n=5105

```bash
python edge_eval/evaluate_edges_seism_matlab_identical_parallel.py \
    --pred_dir        outputs/PASCALContext/resnet18/single_task/edge/results/edge \
    --seg_dir         PASCAL_MT/pascal-context/trainval \
    --image_list_file benchmark/image_lists/full_5105.txt \
    --workers         4 \
    --out             results/parallel_eval_5105.json
```

### 5.4 Using the benchmark harness (multiple runs + summary stats)

```bash
# n=10, 4 workers, 3 runs
python benchmark/scripts/benchmark_python_parallel.py \
    --image_list benchmark/image_lists/small_10.txt \
    --pred_dir   outputs/PASCALContext/resnet18/single_task/edge/results/edge \
    --gt_dir     PASCAL_MT/pascal-context/trainval \
    --workers    4 \
    --runs       3 \
    --output     benchmark/logs/python_parallel/small_10.json

# n=100, 4 workers, 3 runs
python benchmark/scripts/benchmark_python_parallel.py \
    --image_list benchmark/image_lists/medium_100.txt \
    --pred_dir   outputs/PASCALContext/resnet18/single_task/edge/results/edge \
    --gt_dir     PASCAL_MT/pascal-context/trainval \
    --workers    4 \
    --runs       3 \
    --output     benchmark/logs/python_parallel/medium_100.json

# n=5105, 4 workers, 1 run (long — expect hours)
python benchmark/scripts/benchmark_python_parallel.py \
    --image_list benchmark/image_lists/full_5105.txt \
    --pred_dir   outputs/PASCALContext/resnet18/single_task/edge/results/edge \
    --gt_dir     PASCAL_MT/pascal-context/trainval \
    --workers    4 \
    --runs       1 \
    --output     benchmark/logs/python_parallel/full_5105.json
```

### 5.5 How to select image subsets

The image list files in `benchmark/image_lists/` are plain text files, one
stem per line (no extension).  Pass the appropriate file via
`--image_list_file` (direct evaluator) or `--image_list` (benchmark harness).

To create a custom subset of M images:

```bash
head -M benchmark/image_lists/full_5105.txt > benchmark/image_lists/custom_M.txt
```

### 5.6 Where outputs are written

| Output | Location |
|---|---|
| JSON results (direct evaluator) | Path given to `--out` |
| JSON benchmark summary | Path given to `--output` (harness) |
| Log output | stdout / redirect with `tee` |

Example with logging:

```bash
python benchmark/scripts/benchmark_python_parallel.py \
    --image_list benchmark/image_lists/medium_100.txt \
    --workers 4 --runs 3 \
    --output benchmark/logs/python_parallel/medium_100.json \
    2>&1 | tee benchmark/logs/python_parallel/medium_100.log
```

---

## 6. Suggested Benchmark Procedure

Follow these steps for a rigorous, reproducible benchmark campaign.

### Step 1 — Prepare subset lists

Verify the three canonical lists exist and contain the expected counts:

```bash
wc -l benchmark/image_lists/small_10.txt   # → 10
wc -l benchmark/image_lists/medium_100.txt  # → 100
wc -l benchmark/image_lists/full_5105.txt   # → 5105
```

If you need to create them from scratch, see `benchmark/README.md`.

### Step 2 — Verify same inputs on both sides

```bash
# Check prediction files exist for all images in the list
while IFS= read -r stem; do
    f="outputs/PASCALContext/resnet18/single_task/edge/results/edge/${stem}.png"
    [ -f "$f" ] || echo "MISSING: $f"
done < benchmark/image_lists/medium_100.txt

# Check GT .mat files exist for all images
while IFS= read -r stem; do
    f="PASCAL_MT/pascal-context/trainval/${stem}.mat"
    [ -f "$f" ] || echo "MISSING: $f"
done < benchmark/image_lists/medium_100.txt
```

### Step 3 — Verify serial vs. parallel output match (correctness check)

Run both serial and parallel on n=10 and compare ODS/OIS/AP (see §3.3).
All values should match to at least 1e-6.

### Step 4 — Run warm-up

Run one warm-up iteration (not timed) to let the OS cache files and load
Python modules / OR-Tools into each worker:

```bash
python edge_eval/evaluate_edges_seism_matlab_identical_parallel.py \
    --pred_dir outputs/PASCALContext/resnet18/single_task/edge/results/edge \
    --seg_dir  PASCAL_MT/pascal-context/trainval \
    --image_list_file benchmark/image_lists/small_10.txt \
    --workers 4
```

Discard this run's timing.

### Step 5 — Run timed benchmark

Use the benchmark harness (`--runs 3` recommended):

```bash
python benchmark/scripts/benchmark_python_parallel.py \
    --image_list benchmark/image_lists/medium_100.txt \
    --workers 4 --runs 3 \
    --output benchmark/logs/python_parallel/medium_100.json \
    2>&1 | tee benchmark/logs/python_parallel/medium_100.log
```

Repeat for n=10, n=100, and n=5105 as needed.

### Step 6 — Save logs and results

Results JSON and log files are written to `benchmark/logs/python_parallel/`.
Archive both for your thesis.

### Step 7 — Compare outputs

```python
import json

serial  = json.load(open('results/verify_serial_100.json'))
parallel = json.load(open('results/parallel_eval_100.json'))

print(f"Serial   ODS={serial['ODS']*100:.2f}  OIS={serial['OIS']*100:.2f}  AP={serial['AP']*100:.2f}")
print(f"Parallel ODS={parallel['ODS']*100:.2f}  OIS={parallel['OIS']*100:.2f}  AP={parallel['AP']*100:.2f}")
print(f"Serial time:   {serial['eval_time_seconds']:.1f} s")
print(f"Parallel time: {parallel['eval_time_seconds']:.1f} s")
```

### Step 8 — Report results

Include in your thesis:
- Mean ± std runtime for each subset (n=10, 100, 5105)
- Worker count used
- ODS/OIS/AP to confirm correctness
- Machine specs (CPU model, core count, RAM)
- Python version and OR-Tools version

---

## 7. Remaining Unavoidable Differences

Even with identical data and algorithm parameters, some differences between
MATLAB and Python runtime environments are inherent:

| Factor | MATLAB | Python | Impact |
|---|---|---|---|
| **Matching engine** | MEX-compiled C++ CSA | OR-Tools Python bindings (C++ under the hood) | Algorithmic equivalent but different constant factors |
| **Thinning** | `bwmorph('thin', Inf)` — proprietary | `skimage.morphology.thin` — verified equivalent | Numerically identical (verified) |
| **JIT compilation** | MATLAB JIT (always on) | CPython (no JIT); NumPy uses C/Fortran internally | Python may be slower for pure-Python loops |
| **Worker startup** | `parpool` stays alive between runs | `ProcessPoolExecutor` restarts between benchmark calls | Python has higher cold-start overhead |
| **Memory model** | Shared memory (copy-on-write) | Each process has own memory; data pickled | Python may use more RAM with many workers |
| **Scheduler** | MATLAB's built-in task scheduler | OS scheduler + Python futures | Both are non-deterministic across runs |
| **Filesystem cache** | OS page cache (same) | OS page cache (same) | Equal after warm-up |
| **OR-Tools version** | N/A | Depends on installed version | Pin version for reproducibility |

### Pinning OR-Tools version

```bash
pip show ortools | grep Version
# Record this in your thesis.  Recommended: ortools >= 9.8
```

### Python version

```bash
python --version
# Record this in your thesis.
```

---

## 8. Output Format Reference

### Direct evaluator (`evaluate_edges_seism_matlab_identical_parallel.py`)

The output JSON is compatible with the serial script's format, with two
additional fields:

```json
{
  "ODS": 0.6615,
  "ODS_precision": 0.6720,
  "ODS_recall": 0.6513,
  "ODS_threshold_idx": 54,
  "OIS": 0.6781,
  "OIS_precision": 0.6890,
  "OIS_recall": 0.6675,
  "AP": 0.5402,
  "n_images": 100,
  "n_skipped": 0,
  "n_thresholds": 99,
  "n_workers": 4,
  "eval_time_seconds": 95.3,
  "per_image_f1": [...],
  "thresholds": [0.01, 0.02, "..."],
  "mean_precision": [...],
  "mean_recall": [...],
  "mean_fmeas": [...],
  "matching": "csa_ortools",
  "ap_method": "interpolation_101_points_matlab_identical",
  "ortools_available": true,
  "mode": "parallel"
}
```

### Benchmark harness (`benchmark_python_parallel.py`)

```json
{
  "mode": "parallel",
  "image_list": "benchmark/image_lists/medium_100.txt",
  "n_images": 100,
  "n_runs": 3,
  "n_workers": 4,
  "n_thresholds": 99,
  "timestamp": "2026-04-23T14:00:00",
  "runs": [
    {
      "run_id": 1,
      "total_wall_s": 102.1,
      "eval_time_s": 95.3,
      "pool_overhead_s": 6.8,
      "n_workers": 4,
      "metrics": { "ODS": 0.6615, "OIS": 0.6781, "AP": 0.5402, "n_images": 100 }
    }
  ],
  "summary": {
    "mean_total_wall_s": 101.5,
    "std_total_wall_s": 0.9,
    "min_total_wall_s": 100.7,
    "max_total_wall_s": 102.5,
    "mean_eval_s": 94.8
  }
}
```

---

**Last updated:** 2026-04-23
**Author:** Parallel benchmark extension for bachelor thesis runtime comparison
