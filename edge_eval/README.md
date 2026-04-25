# Edge Evaluation — `edge_eval/`

This folder contains the Python SEISM-compatible edge evaluation pipeline used
to benchmark predicted edge maps against PASCAL Context ground-truth.

---

## Files

| File | Description |
|---|---|
| `evaluate_edges_seism_matlab_identical.py` | **Serial evaluator.** Single source of truth for all metric functions (ODS, OIS, AP). MATLAB-identical. |
| `evaluate_edges_seism_matlab_identical_parallel.py` | **Parallel evaluator.** Runs the same per-image evaluation in parallel using `ProcessPoolExecutor`. All metric functions are imported from the serial file. |

---

## Metrics

Both scripts report:

- **ODS** -- Optimal Dataset Scale: threshold that maximises global F1
- **OIS** -- Optimal Image Scale: per-image optimal threshold, then pooled P/R
- **AP**  -- Average Precision: 101-point interpolation of the P/R curve, MATLAB-compatible

---

## How to Run

### Serial (recommended for correctness checks)

```bash
python edge_eval/evaluate_edges_seism_matlab_identical.py \
    --pred_dir  <path/to/predicted/edge/pngs> \
    --seg_dir   <path/to/pascal-context/trainval> \
    --image_list_file benchmark/image_lists/small_10.txt \
    --out results/serial_eval_10.json
```

### Parallel (recommended for large-scale evaluation)

```bash
python edge_eval/evaluate_edges_seism_matlab_identical_parallel.py \
    --pred_dir  <path/to/predicted/edge/pngs> \
    --seg_dir   <path/to/pascal-context/trainval> \
    --image_list_file benchmark/image_lists/small_10.txt \
    --workers 4 \
    --out results/parallel_eval_10.json
```

Replace `small_10.txt` with `medium_100.txt` or `full_5105.txt` for larger subsets.
Use `--workers N` to set the number of parallel processes (default: all CPU cores).

---

## Parallelisation Design

Only the outer image loop is parallelised. Per-image evaluation (`evaluate_single_image`)
and all metric aggregation functions (`compute_ods_matlab`, `compute_ois_matlab`,
`compute_ap_matlab`) are imported unchanged from the serial script. No algorithm
logic is duplicated. Results are collected in original image order to guarantee
deterministic accumulation identical to the serial run.

---

## Output

Both scripts write a JSON file (when `--out` is given) containing:
`ODS`, `OIS`, `AP`, full P/R curves (99 points), per-image F1, thresholds,
evaluation time, and matching algorithm used.

---

## Requirements

- Python >= 3.8
- `numpy`, `scipy`, `scikit-image`, `tqdm`
- OR-Tools (`ortools`) for CSA bipartite matching (falls back to Hungarian if unavailable)
- PASCAL Context `.mat` ground-truth files (`LabelMap` field)
- Predicted edge maps as float PNG files (values in `[0, 1]`)
