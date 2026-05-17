# mtl-edge-eval

Python-native SEISM-compatible edge evaluation for multi-task learning.

Bachelor thesis project, 2026.

---

## What the script does

`edge_eval/evaluate_edges_seism_matlab_identical.py` evaluates predicted edge
maps against PASCAL Context ground-truth boundaries and reports the standard
SEISM metrics: **ODS**, **OIS**, and **AP**.

It provides a Python-native alternative to the original MATLAB evaluation path
(`evaluation/eval_edge.py` -> SEISM `pr_curves_base.m`). The implementation
follows the MATLAB/SEISM reference in the main algorithmic choices and was
validated to closely reproduce the MATLAB results:

| Component | MATLAB (SEISM) | This script |
|-----------|---------------|-------------|
| Thresholds | `linspace(1/100, 99/100, 99)` | same threshold grid |
| GT thinning | `bwmorph(E,'thin',Inf)` | `skimage.morphology.thin` |
| Pred thinning | `bwmorph(E,'thin',Inf)` | `skimage.morphology.thin` |
| Matching tolerance | `maxDist = 0.0075 * diagonal` | same tolerance constant |
| Matching kernel | `correspondPixels` MEX (C++) | OR-Tools min-cost flow / Hungarian fallback |
| ODS / OIS / AP | `general_ods/ois/ap.m` | MATLAB-compatible aggregation |

## Dependencies

```
pip install scikit-image scipy ortools numpy Pillow tqdm
```

| Package | Role |
|---------|------|
| `scikit-image` | Morphological thinning |
| `scipy` | Hungarian matching fallback + .mat file loading |
| `ortools` | OR-Tools CSA primary matching (strongly recommended) |
| `numpy` | Array operations |
| `Pillow` | PNG loading |
| `tqdm` | Progress bars |

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
- `timing` -- total runtime
- `matching_algorithm` -- `"CSA (OR-Tools)"` or `"Hungarian"`

---

## Example commands

### 1 -- Single-image sanity check

```bash
python edge_eval/evaluate_edges_seism_matlab_identical.py \
    --pred_dir outputs/PASCALContext/resnet18/single_task/edge/results/edge \
    --seg_dir  PASCAL_MT/pascal-context/trainval \
    --image_list_file image_lists/single_2008_000002.txt \
    --out results/edge_eval_1img.json
```

### 2 -- 10-image subset 

```bash
python edge_eval/evaluate_edges_seism_matlab_identical.py \
    --pred_dir         outputs/PASCALContext/resnet18/single_task/edge/results/edge \
    --seg_dir          PASCAL_MT/pascal-context/trainval \
    --image_list_file  image_lists/small_10.txt \
    --out              results/edge_eval_10.json
```

### 3 -- 100-image subset 

```bash
python edge_eval/evaluate_edges_seism_matlab_identical.py \
    --pred_dir         outputs/PASCALContext/resnet18/single_task/edge/results/edge \
    --seg_dir          PASCAL_MT/pascal-context/trainval \
    --image_list_file  image_lists/medium_100.txt \
    --out              results/edge_eval_100.json
```

### 4 -- Full dataset: 5,105 images

```bash
python edge_eval/evaluate_edges_seism_matlab_identical.py \
    --pred_dir         outputs/PASCALContext/resnet18/single_task/edge/results/edge \
    --seg_dir          PASCAL_MT/pascal-context/trainval \
    --image_list_file  image_lists/full_5105.txt \
    --out              results/edge_eval_full.json
```



---

## License

MIT -- see [LICENSE](LICENSE).
