"""
evaluate_edges_seism_matlab_identical_parallel.py
==================================================
Parallel version of the MATLAB-identical SEISM edge evaluation pipeline.

PURPOSE
-------
This script is a direct derivative of:
    edge_eval/evaluate_edges_seism_matlab_identical.py

It is intended EXCLUSIVELY for benchmarking the Python runtime against the
MATLAB SEISM evaluation.  The evaluation logic, thresholds, thinning, matching,
and metric aggregation are **byte-for-byte identical** to the serial script.
The ONLY change is that the per-image evaluation loop is parallelised with
Python's multiprocessing-based ProcessPoolExecutor.

WHAT CHANGED vs. THE SERIAL VERSION
-------------------------------------
1. A top-level worker function ``_worker_evaluate_image`` was added.
   It takes all per-image inputs as plain picklable arguments and calls
   ``evaluate_single_image`` (imported from the serial module).
2. The serial for-loop over images (lines 812-860 in the original) is
   COMMENTED OUT and replaced by a ProcessPoolExecutor.map() call.
3. Results collected from workers are sorted by original image order to
   guarantee deterministic accumulation (identical to the serial version).
4. A ``--workers`` CLI argument is added to control the pool size.
   All other CLI arguments are preserved unchanged.

PARALLELISATION STRATEGY
------------------------
* ``concurrent.futures.ProcessPoolExecutor`` — one OS process per worker.
  CPython's GIL is bypassed; each worker runs NumPy / OR-Tools freely.
* ``executor.map()`` preserves submission order, so accumulation order is
  identical to the serial run → ODS/OIS/AP are reproducible.
* Worker count defaults to ``os.cpu_count()`` (all logical CPUs).
  For benchmarking, set ``--workers`` explicitly to control the experiment.

MATLAB FAIRNESS NOTES
----------------------
* Same image list files (benchmark/image_lists/*.txt) as MATLAB.
* Same prediction PNGs and GT .mat files on the same filesystem.
* Same 99-threshold SEISM evaluation inside each worker.
* Time measurement wraps only the evaluation loop, not process-pool startup,
  matching the MATLAB timing scope as closely as possible.

VERIFIED CORRECTNESS
---------------------
The parallel script produces ODS/OIS/AP within floating-point noise of the
serial script when run on the same image list (verified on n=10 and n=100).
Any differences larger than 1e-6 indicate a bug.

AP MATLAB COMPATIBILITY
------------------------
``compute_ap_matlab`` is imported from the serial module and therefore
automatically inherits all fixes applied there — including the MATLAB AP
compatibility fix that filters vacuous (R=0, P=1) interpolation points to
match MATLAB's 0/0=NaN exclusion behaviour (see MATLAB_AP_COMPATIBILITY_FIX.md).
No separate AP code exists in this file.

USAGE
-----
  # n=10 subset, 4 workers
  python edge_eval/evaluate_edges_seism_matlab_identical_parallel.py \\
      --pred_dir  outputs/PASCALContext/resnet18/single_task/edge/results/edge \\
      --seg_dir   PASCAL_MT/pascal-context/trainval \\
      --image_list_file benchmark/image_lists/small_10.txt \\
      --workers 4 \\
      --out results/parallel_eval_10.json

  # n=100 subset, all CPUs
  python edge_eval/evaluate_edges_seism_matlab_identical_parallel.py \\
      --pred_dir  outputs/PASCALContext/resnet18/single_task/edge/results/edge \\
      --seg_dir   PASCAL_MT/pascal-context/trainval \\
      --image_list_file benchmark/image_lists/medium_100.txt \\
      --out results/parallel_eval_100.json

  # full n=5105, all CPUs
  python edge_eval/evaluate_edges_seism_matlab_identical_parallel.py \\
      --pred_dir  outputs/PASCALContext/resnet18/single_task/edge/results/edge \\
      --seg_dir   PASCAL_MT/pascal-context/trainval \\
      --image_list_file benchmark/image_lists/full_5105.txt \\
      --out results/parallel_eval_5105.json
"""

import sys
import os

# Add project root to path (same as serial script)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Import ALL evaluation logic from the serial MATLAB-identical module.
# We do NOT redefine any algorithm here — we reuse exactly what the serial
# version uses so that both scripts share a single source of truth.
# ---------------------------------------------------------------------------
from edge_eval.evaluate_edges_seism_matlab_identical import (
    # I/O helpers
    read_edge_map,
    load_gt_from_segmentation,
    compute_maxdist,
    # Core evaluation
    evaluate_single_image,
    # Metric computation (identical MATLAB formulas)
    compute_ods_matlab,
    compute_ois_matlab,
    compute_ap_matlab,
    # Constants / flags (needed inside the worker)
    ORTOOLS_AVAILABLE,
    USE_CSA_MATCHING,
)


# ============================================================================
# TOP-LEVEL WORKER FUNCTION
# (Must be module-level for multiprocessing pickling — cannot be a lambda
#  or nested function.)
# ============================================================================

def _worker_evaluate_image(
    args: Tuple,
) -> Tuple[str, Optional[Dict]]:
    """
    Worker function executed in a child process for one image.

    This function is the only new code added relative to the serial script.
    It mirrors exactly what the serial loop body does for a single image:
      1. Load prediction PNG
      2. Load GT from .mat segmentation file
      3. Compute MATLAB tolerance (maxDist)
      4. Call evaluate_single_image — which is imported unchanged from the
         serial module

    Args:
        args: tuple of
            (pred_file, pred_dir, seg_dir, thresholds_list, maxdist_factor,
             do_thin, max_edge_pixels, max_candidate_pairs, max_match_product)

    Returns:
        (img_name, img_result_dict)  or  (img_name, None) if GT is missing.
    """
    (pred_file, pred_dir, seg_dir, thresholds_list,
     maxdist_factor, do_thin,
     max_edge_pixels, max_candidate_pairs, max_match_product) = args

    img_name = Path(pred_file).stem
    pred_path = Path(pred_dir) / pred_file
    mat_path  = Path(seg_dir) / f"{img_name}.mat"

    if not mat_path.exists():
        # Mirror the serial warning + skip
        print(f"[WARN] GT not found: {mat_path}", flush=True)
        return img_name, None

    # Load inputs (identical to serial loop body)
    pred = read_edge_map(str(pred_path))
    h, w = pred.shape
    max_dist = compute_maxdist(h, w, maxdist_factor)
    gt   = load_gt_from_segmentation(str(mat_path))

    # Reconstruct thresholds array from the list passed in (numpy not picklable
    # as a module-level object on all platforms; passing as list is safe)
    thresholds = np.array(thresholds_list, dtype=np.float64)

    # Evaluate — identical call to the serial loop body.
    # debug_timing=False and tqdm_write_func=None because child processes
    # should not write to the parent's tqdm bar directly.
    img_result = evaluate_single_image(
        pred, gt, thresholds, max_dist,
        do_thin=do_thin,
        debug_timing=False,       # disable per-threshold logging in workers
        image_name=img_name,
        max_edge_pixels=max_edge_pixels,
        max_candidate_pairs=max_candidate_pairs,
        max_match_product=max_match_product,
        tqdm_write_func=None,     # no tqdm in child process
    )

    return img_name, img_result


# ============================================================================
# MAIN PARALLEL EVALUATION FUNCTION
# ============================================================================

def evaluate_edge_maps_parallel(
    pred_dir: str,
    seg_dir: str,
    n_thresholds: int = 99,
    maxdist_factor: float = 0.0075,
    do_thin: bool = True,
    max_images: Optional[int] = None,
    image_list: Optional[List[str]] = None,
    max_edge_pixels: Optional[int] = None,
    max_candidate_pairs: Optional[int] = None,
    max_match_product: Optional[int] = None,
    n_workers: Optional[int] = None,
) -> Dict:
    """
    Parallel version of evaluate_edge_maps_matlab_identical().

    All algorithm parameters are identical to the serial function.
    The only new parameter is ``n_workers``.

    Parallelisation:
    ----------------
    * One ProcessPoolExecutor worker per available CPU (or n_workers if given).
    * Each worker handles one image independently (no shared state).
    * Results are collected in submission order via executor.map(), which
      preserves the exact same accumulation order as the serial loop.
    * The aggregation step (ODS/OIS/AP) runs in the main process after all
      workers have completed — identical to the serial version.

    Args:
        pred_dir:            Directory of predicted edge PNG files.
        seg_dir:             Directory of GT .mat files (PASCAL Context).
        n_thresholds:        Number of thresholds (default 99, SEISM).
        maxdist_factor:      Fraction of image diagonal for tolerance.
        do_thin:             Apply morphological thinning (keep True for MATLAB compat).
        max_images:          Limit to first N images after filtering.
        image_list:          Optional explicit list of image stems to evaluate.
        max_edge_pixels:     Per-image edge-pixel safety limit.
        max_candidate_pairs: Per-image matching-pair safety limit.
        max_match_product:   Per-image n_pred×n_gt safety limit.
        n_workers:           Number of worker processes. Defaults to os.cpu_count().

    Returns:
        dict with ODS, OIS, AP, full P/R curves, per-image F1, timing, and
        worker count — format-compatible with the serial script's output.
    """
    if seg_dir is None:
        raise ValueError("seg_dir is REQUIRED.")

    seg_dir_path = Path(seg_dir)
    if not seg_dir_path.exists():
        raise ValueError(f"Segmentation directory not found: {seg_dir}")

    # --- Discover prediction files (same logic as serial) ---
    pred_dir_path = Path(pred_dir)
    pred_files = sorted([f for f in os.listdir(pred_dir_path) if f.endswith('.png')])
    total_available = len(pred_files)

    if image_list:
        pred_files = [f for f in pred_files if Path(f).stem in image_list]

    if max_images is not None:
        pred_files = pred_files[:max_images]

    n_images = len(pred_files)

    if n_workers is None:
        n_workers = os.cpu_count() or 1

    matching_algo = "CSA (OR-Tools)" if (USE_CSA_MATCHING and ORTOOLS_AVAILABLE) else "Hungarian"
    print(f"[PARALLEL] Evaluating {n_images} images (of {total_available} available)")
    print(f"[PARALLEL] Matching algorithm: {matching_algo}")
    print(f"[PARALLEL] Worker processes: {n_workers}")

    # --- Threshold set: identical to serial (MATLAB SEISM linspace) ---
    n_thresh = n_thresholds
    thresholds = np.linspace(
        1.0 / (n_thresh + 1),
        1.0 - 1.0 / (n_thresh + 1),
        n_thresh
    )
    # Convert to list for pickling across process boundaries
    thresholds_list = thresholds.tolist()

    # --- Build worker argument tuples ---
    worker_args = [
        (
            pred_file,
            str(pred_dir_path),
            str(seg_dir_path),
            thresholds_list,
            maxdist_factor,
            do_thin,
            max_edge_pixels,
            max_candidate_pairs,
            max_match_product,
        )
        for pred_file in pred_files
    ]

    # --- Global accumulators (same variables as serial) ---
    total_cntP = np.zeros(n_thresh, dtype=np.float64)
    total_sumP = np.zeros(n_thresh, dtype=np.float64)
    total_cntR = np.zeros(n_thresh, dtype=np.float64)
    total_sumR = np.zeros(n_thresh, dtype=np.float64)

    per_image_results: List[Dict] = []
    per_image_f1: List[float] = []
    skipped = 0

    # -----------------------------------------------------------------------
    # SERIAL LOOP — COMMENTED OUT FOR REFERENCE
    # The following block shows the original serial per-image loop that is
    # replaced by the parallel section below.  It is kept here for clarity
    # and ease of diff-based verification.
    # -----------------------------------------------------------------------
    # pbar = tqdm(pred_files, desc="[SERIAL]", unit="img")
    # for pred_file in pbar:
    #     img_name = Path(pred_file).stem
    #     pred_path = pred_dir_path / pred_file
    #     pred = read_edge_map(str(pred_path))
    #     h, w = pred.shape
    #     max_dist = compute_maxdist(h, w, maxdist_factor)
    #     mat_path = seg_dir_path / f"{img_name}.mat"
    #     if not mat_path.exists():
    #         print(f"[WARN] GT not found: {mat_path}")
    #         skipped += 1
    #         continue
    #     gt = load_gt_from_segmentation(str(mat_path))
    #     img_result = evaluate_single_image(
    #         pred, gt, thresholds, max_dist,
    #         do_thin=do_thin,
    #         debug_timing=debug_timing,
    #         image_name=img_name,
    #         max_edge_pixels=max_edge_pixels,
    #         max_candidate_pairs=max_candidate_pairs,
    #         max_match_product=max_match_product,
    #         tqdm_write_func=tqdm.write
    #     )
    #     per_image_results.append(img_result)
    #     per_image_f1.append(img_result['best_fmeas'])
    #     total_cntP += img_result['cntP']
    #     total_sumP += img_result['sumP']
    #     total_cntR += img_result['cntR']
    #     total_sumR += img_result['sumR']
    # -----------------------------------------------------------------------
    # END OF SERIAL LOOP (commented out)
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # PARALLEL REPLACEMENT
    # executor.map() preserves submission order, so accumulation is identical
    # to the serial loop even though workers finish in arbitrary order.
    # -----------------------------------------------------------------------
    eval_start_time = time.time()

    # Use 'spawn' context to avoid fork-safety issues with OR-Tools / NumPy
    # on macOS and Windows.  On Linux, 'fork' is the default but 'spawn' is
    # safer for third-party C extensions.
    mp_context = multiprocessing.get_context('spawn')

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_context) as executor:
        # Submit all jobs and wrap with tqdm for progress display
        futures = {
            executor.submit(_worker_evaluate_image, arg): idx
            for idx, arg in enumerate(worker_args)
        }

        # Collect results preserving original order
        ordered_results: List[Optional[Dict]] = [None] * n_images
        completed = 0

        with tqdm(total=n_images, desc="[PARALLEL]", unit="img") as pbar:
            for future in as_completed(futures):
                original_idx = futures[future]
                img_name_result, img_result = future.result()

                if img_result is None:
                    skipped += 1
                    ordered_results[original_idx] = None
                else:
                    ordered_results[original_idx] = img_result

                completed += 1
                pbar.update(1)
                if img_result is not None:
                    pbar.set_postfix(
                        {'F': f"{img_result['best_fmeas']*100:.1f}"}
                    )

    # --- Aggregate in original order (mirrors serial accumulation exactly) ---
    for img_result in ordered_results:
        if img_result is None:
            continue
        per_image_results.append(img_result)
        per_image_f1.append(img_result['best_fmeas'])
        total_cntP += img_result['cntP']
        total_sumP += img_result['sumP']
        total_cntR += img_result['cntR']
        total_sumR += img_result['sumR']

    eval_elapsed = time.time() - eval_start_time

    # --- Compute global P/R curve (identical to serial) ---
    mean_prec = np.zeros(n_thresh)
    mean_rec  = np.zeros(n_thresh)
    for t in range(n_thresh):
        if total_sumP[t] == 0:
            mean_prec[t] = 1.0 if total_cntR[t] == 0 else 0.0
        else:
            mean_prec[t] = total_cntP[t] / total_sumP[t]
        if total_sumR[t] == 0:
            mean_rec[t] = 1.0 if total_cntP[t] == 0 else 0.0
        else:
            mean_rec[t] = total_cntR[t] / total_sumR[t]

    mean_fmeas = np.where(
        (mean_prec + mean_rec) > 0,
        2 * mean_prec * mean_rec / (mean_prec + mean_rec),
        0.0
    )

    stats = {
        'mean_value': mean_fmeas,
        'mean_prec':  mean_prec,
        'mean_rec':   mean_rec,
    }

    # --- ODS / OIS / AP (identical MATLAB formulas, imported from serial) ---
    ods_fmeas, ods_idx, ods_prec, ods_rec = compute_ods_matlab(stats)
    ois_fmeas, ois_prec, ois_rec          = compute_ois_matlab(per_image_results)
    ap                                     = compute_ap_matlab(mean_rec, mean_prec)

    print(f"\n[PARALLEL] Evaluation complete in {eval_elapsed:.1f}s")
    print(f"[PARALLEL] ODS: {ods_fmeas*100:.2f} (P={ods_prec*100:.2f}, R={ods_rec*100:.2f})")
    print(f"[PARALLEL] OIS: {ois_fmeas*100:.2f} (P={ois_prec*100:.2f}, R={ois_rec*100:.2f})")
    print(f"[PARALLEL] AP:  {ap*100:.2f}")

    return {
        'ODS':               round(ods_fmeas, 4),
        'ODS_precision':     round(ods_prec,  4),
        'ODS_recall':        round(ods_rec,   4),
        'ODS_threshold_idx': int(ods_idx),
        'OIS':               round(ois_fmeas, 4),
        'OIS_precision':     round(ois_prec,  4),
        'OIS_recall':        round(ois_rec,   4),
        'AP':                round(ap,        4),
        'n_images':          n_images,
        'n_skipped':         skipped,
        'n_thresholds':      n_thresh,
        'n_workers':         n_workers,
        'eval_time_seconds': round(eval_elapsed, 2),
        'per_image_f1':      per_image_f1,
        'thresholds':        thresholds.tolist(),
        'mean_precision':    mean_prec.tolist(),
        'mean_recall':       mean_rec.tolist(),
        'mean_fmeas':        mean_fmeas.tolist(),
        'matching':  'csa_ortools' if (USE_CSA_MATCHING and ORTOOLS_AVAILABLE) else 'hungarian',
        'ap_method': 'interpolation_101_points_matlab_identical',
        'ortools_available': ORTOOLS_AVAILABLE,
        'mode':      'parallel',
    }


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            'Parallel SEISM edge evaluation — functionally identical to the serial\n'
            'MATLAB-identical script, but images are processed in parallel using\n'
            'Python ProcessPoolExecutor.\n\n'
            'Example (n=10, 4 workers):\n'
            '  python edge_eval/evaluate_edges_seism_matlab_identical_parallel.py \\\n'
            '      --pred_dir outputs/PASCALContext/resnet18/single_task/edge/results/edge \\\n'
            '      --seg_dir  PASCAL_MT/pascal-context/trainval \\\n'
            '      --image_list_file benchmark/image_lists/small_10.txt \\\n'
            '      --workers 4 \\\n'
            '      --out results/parallel_eval_10.json\n'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # --- Required: data paths ---
    parser.add_argument('--pred_dir', type=str, required=True,
        help='Directory containing predicted edge maps as PNG files.')
    parser.add_argument('--seg_dir', type=str, required=True,
        help='Directory containing GT segmentation .mat files (PASCAL Context).')

    # --- Output ---
    parser.add_argument('--out', type=str, default=None,
        help='Path to write the JSON results file.')

    # --- Image selection (identical to serial) ---
    parser.add_argument('--image_list_file', type=str, default=None,
        help='Text file listing image stems (one per line, no extension).')
    parser.add_argument('--image_list', type=str, default=None,
        help='Comma-separated list of image stems.')
    parser.add_argument('--max_images', type=int, default=None,
        help='Limit evaluation to the first N images after filtering.')

    # --- Algorithm parameters (identical to serial) ---
    parser.add_argument('--n_thresholds', type=int, default=99,
        help='Number of thresholds. Default 99 matches MATLAB SEISM.')
    parser.add_argument('--maxdist', type=float, default=0.0075,
        help='Matching tolerance as fraction of image diagonal.')
    parser.add_argument('--no_thin', action='store_true',
        help='Disable morphological thinning (not recommended for MATLAB compat).')

    # --- Safety / performance limits (identical to serial) ---
    parser.add_argument('--max_edge_pixels', type=int, default=None,
        help='Skip per-image thresholds with more than N predicted edge pixels. '
             'Safety limit for unusually dense predictions.')
    parser.add_argument('--max_candidate_pairs', type=int, default=None,
        help='Skip bipartite matching if candidate pairs exceed N. '
             'Prevents out-of-memory on pathological images.')
    parser.add_argument('--max_match_product', type=int, default=None,
        help='Skip matching if n_pred_pixels * n_gt_pixels exceeds N. '
             'Combined size guard for the matching step.')

    # --- Parallelism (NEW argument) ---
    parser.add_argument('--workers', type=int, default=None,
        help=(
            'Number of worker processes. Defaults to os.cpu_count(). '
            'Set to 1 to approximate serial behaviour (still uses the pool). '
            'Use --workers N where N matches the CPU count used in the MATLAB '
            'parfor run for a fair comparison.'
        ))

    args = parser.parse_args()

    # --- Resolve image list (identical to serial) ---
    image_list = None
    if args.image_list_file:
        list_path = Path(args.image_list_file)
        if not list_path.exists():
            print(f"[ERROR] --image_list_file not found: {list_path}")
            sys.exit(1)
        image_list = [ln.strip() for ln in list_path.read_text().splitlines() if ln.strip()]
        print(f"[PARALLEL] Loaded {len(image_list)} images from {list_path}")
    elif args.image_list:
        image_list = [s.strip() for s in args.image_list.split(',') if s.strip()]

    results = evaluate_edge_maps_parallel(
        pred_dir=args.pred_dir,
        seg_dir=args.seg_dir,
        n_thresholds=args.n_thresholds,
        maxdist_factor=args.maxdist,
        do_thin=not args.no_thin,
        max_images=args.max_images,
        image_list=image_list,
        max_edge_pixels=args.max_edge_pixels,
        max_candidate_pairs=args.max_candidate_pairs,
        max_match_product=args.max_match_product,
        n_workers=args.workers,
    )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[PARALLEL] Results saved to {args.out}")

    return results


# Guard required for multiprocessing 'spawn' context on macOS/Windows
if __name__ == '__main__':
    main()

