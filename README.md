# mtl-edge-eval

Pure-Python SEISM edge evaluation for multi-task learning.
Replicates the MATLAB `correspondPixels` MEX kernel using OR-Tools CSA matching.

Bachelor thesis project, 2026.

---

## What the script does

`edge_eval/evaluate_edges_seism_matlab_identical.py` evaluates predicted edge
maps against PASCAL Context ground-truth boundaries and reports the standard
SEISM metrics: **ODS**, **OIS**, and **AP**.

It is a drop-in Python replacement for the original MATLAB evaluation path
(`evaluation/eval_edge.py` -> SEISM `pr_curves_base.m`), with every algorithmic
choice made to reproduce MATLAB results exactly:

| Component | MATLAB (SEISM) | This script |
|-----------|---------------|-------------|
| Thresholds | `linspace(1/100, 99/100, 99)` | identical |
| GT thinning | `bwmorph(E,'thin',Inf)` | `skimage.morphology.thin` |
| Pred thinning | `bwmorph(E,'thin',Inf)` | `skimage.morphology.thin` |
| Matching tolerance | `maxDist = 0.0075 * diagonal` | identical constant |
| Matching kernel | `correspondPixels` MEX (C++) | OR-Tools CSA (Python) |
| ODS / OIS / AP | `general_ods/ois/ap.m` | identical formulas |

## Dependencies

```
pip install scikit-image scipy ortools numpy Pillow
```

| Package | Role |
|---------|------|
| `scikit-image` | Morphological thinning |
| `scipy` | Hungarian matching fallback + .mat file loading |
| `ortools` | OR-Tools CSA primary matching (strongly recommended) |
| `numpy` | Array operations |
| `Pillow` | PNG loading |

Python 3.8+ required.

---

## Inputs

| Argument | Required | Description |
|----------|----------|-------------|
| `--pred_dir` | yes | Directory of predicted edge maps as PNG files. Stems must match GT .mat stems. |
| `--seg_dir` | yes | Directory of GT .mat files (PASCAL Context, LabelMap field). Typical path: `PASCAL_MT/pascal-context/trainval/` |
| `--out` | no | Path to write the JSON result file. Recommended for reproducibility. |
| `--image_list_file` | no | Text file of image stems to evaluate, one per line, no extension. |
| `--image_list` | no | Comma-separated image stems for small ad-hoc subsets. |
| `--max_images` | no | Limit to first N images (for quick tests). |

---

## Outputs

**JSON file** (written to `--out`), containing:

- `ODS`, `OIS`, `AP`
- `thresholds` -- 99 threshold values used
- `precision`, `recall`, `fmeasure` -- full 99-point P/R/F curve
- `per_image_f1` -- per-image best F1 (used by OIS)
- `timing` -- total runtime, matching vs. thinning breakdown
- `matching_algorithm` -- `"CSA (OR-Tools)"` or `"Hungarian"`

---

## Example commands

### 1 -- Single-image sanity check (~7 s)

```bash
python edge_eval/evaluate_edges_seism_matlab_identical.py \
    --pred_dir  outputs/HRNet/edge \
    --seg_dir   PASCAL_MT/pascal-context/trainval \
    --image_list "2008_000002" \
    --out        results/edge_eval_1img.json
```

### 2 -- 10-image subset benchmark (~66 s)

```bash
python edge_eval/evaluate_edges_seism_matlab_identical.py \
    --pred_dir         outputs/HRNet/edge \
    --seg_dir          PASCAL_MT/pascal-context/trainval \
    --image_list_file  benchmark/image_lists/small_10.txt \
    --out              results/edge_eval_10.json
```

### 3 -- 100-image subset benchmark (~655 s, thesis result)

```bash
python edge_eval/evaluate_edges_seism_matlab_identical.py \
    --pred_dir         outputs/HRNet/edge \
    --seg_dir          PASCAL_MT/pascal-context/trainval \
    --image_list_file  benchmark/image_lists/medium_100.txt \
    --out              results/edge_eval_100.json
```

For the full argument list, run:

```bash
python edge_eval/evaluate_edges_seism_matlab_identical.py --help
```

---

## Relation to the MATLAB benchmark

This script replicates the core kernel of the MATLAB SEISM evaluation pipeline,
omitting only production-infrastructure overhead (rsync, temp files, SEISM framework
boilerplate). The omitted parts would make MATLAB *slower*, so the speedup comparison
is conservative. The Python implementation is ~2.2x faster because OR-Tools CSA
outperforms `correspondPixels` MEX for the bipartite matching step, which accounts
for ~98% of MATLAB's total runtime.

---

## License

MIT -- see [LICENSE](LICENSE).
