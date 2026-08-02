"""
prepare_drsu_data.py
====================
Zero2Neuro Data Preparation — Dr. Su Fluorescence Time-Series

Converts Dr. Su's dataset into the pickle format expected by
zero2neuro's --data_format=pickle pipeline.

SUPPORTS TWO DATA SOURCES
--------------------------
  A) Image-only (current state):
       Reads the PNG frames in the Frames/ directory,
       extracts 7 pixel-level scalar features per time step,
       and creates a 1-sequence (T=25, F=7) dataset.

  B) Raw numerical data (recommended — ask Dr. Su):
       Reads a CSV/Excel file where each row = one time step,
       and each column = one measured variable.
       This mode produces far better features.

USAGE
-----
  Mode A (images):
    python prepare_drsu_data.py \\
        --mode image \\
        --frames_dir Frames \\
        --dataset_directory dataset \\
        --data_n_folds 3 \\
        --data_outputs growth_score

  Mode B (raw CSV from Dr. Su):
    python prepare_drsu_data.py \\
        --mode raw \\
        --data_file drsu_raw_data.csv \\
        --dataset_directory dataset \\
        --data_n_folds 3 \\
        --data_outputs <target_column_name>

OUTPUT
------
  dataset/
    fold_0.pkl   ← training fold(s)
    fold_1.pkl   ← validation fold
    fold_2.pkl   ← test fold
    metadata.json

Each pkl file is a dict:
  {
    "inputs":  np.ndarray  shape (N_examples, T_timesteps, N_features),
    "outputs": np.ndarray  shape (N_examples, N_outputs),
  }
"""

import os
import sys
import re
import json
import pickle
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


# ---------------------------------------------------------------------------
# Add z2n src to path so we can reuse the standard argument parser
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
from parser import create_parser as _z2n_create_parser


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def save_pkl(data_dict: dict, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as fp:
        pickle.dump(data_dict, fp)
    print(f"    [pkl] written: {path}  — shapes: "
          + ", ".join(f"{k}:{v.shape}" for k, v in data_dict.items()))


# ---------------------------------------------------------------------------
# Mode A — Image-derived feature extraction
# ---------------------------------------------------------------------------

def parse_timestamp(filename: str) -> float:
    m = re.match(r"([\d.]+)\s*s", os.path.basename(filename))
    return float(m.group(1)) if m else float("nan")


def load_gray(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def extract_features_from_frame(frame: np.ndarray, threshold: float = 0.02) -> np.ndarray:
    """
    7 pixel-derived scalar features from one grayscale frame.

    Returns a 1-D float32 array of length 7:
      [mean_intensity, max_intensity, nonzero_fraction,
       centroid_x_norm, centroid_y_norm, spread_x_norm, spread_y_norm]
    """
    h, w = frame.shape
    mask  = frame > threshold
    nz    = mask.mean()
    mean_ = frame.mean()
    max_  = frame.max()

    if mask.any():
        ys, xs  = np.indices(frame.shape)
        wts     = np.where(mask, frame, 0.0)
        tot     = wts.sum() + 1e-12
        cx      = (wts * xs).sum() / tot / w      # normalise to [0,1]
        cy      = (wts * ys).sum() / tot / h
        sx      = np.sqrt(((wts * (xs / w - cx) ** 2).sum()) / tot)
        sy      = np.sqrt(((wts * (ys / h - cy) ** 2).sum()) / tot)
    else:
        cx = cy = 0.5
        sx = sy = 0.0

    return np.array([mean_, max_, nz, cx, cy, sx, sy], dtype=np.float32)


def build_sequence_from_images(frames_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load all frames and return:
      X  shape (T, 7)   — feature matrix
      ts shape (T,)     — timestamps in seconds
    """
    pngs = sorted(frames_dir.glob("*.png"),
                  key=lambda p: parse_timestamp(p.name))
    pngs = [p for p in pngs if not np.isnan(parse_timestamp(p.name))]

    if not pngs:
        raise FileNotFoundError(f"No PNG frames found in {frames_dir}")

    feats, times = [], []
    for p in pngs:
        t = parse_timestamp(p.name)
        f = extract_features_from_frame(load_gray(str(p)))
        feats.append(f)
        times.append(t)

    return np.stack(feats, axis=0), np.array(times, dtype=np.float32)


# ---------------------------------------------------------------------------
# Mode B — Raw numerical data
# ---------------------------------------------------------------------------

def build_sequence_from_csv(csv_path: str,
                             feature_cols: list,
                             output_col: str,
                             time_col: str = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read a CSV/Excel file where rows = time steps.

    Returns:
      X      shape (T, F)  — input features
      y      shape (T, 1)  — target output
      ts     shape (T,)    — timestamps (or step indices if no time column)
    """
    if csv_path.endswith(".csv"):
        df = pd.read_csv(csv_path)
    else:
        df = pd.read_excel(csv_path)

    X  = df[feature_cols].values.astype(np.float32)
    y  = df[[output_col]].values.astype(np.float32)
    ts = df[time_col].values.astype(np.float32) if time_col else np.arange(len(df), dtype=np.float32)

    return X, y, ts


# ---------------------------------------------------------------------------
# Windowing — create many training examples from a single sequence
# ---------------------------------------------------------------------------

def sliding_window(X: np.ndarray,
                   y: np.ndarray,
                   window: int,
                   step: int = 1,
                   predict_ahead: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a single long sequence into overlapping windows.

    X  (T, F)  → inputs   (N, window, F)
    y  (T, 1)  → outputs  (N, 1)       [value at end of each window + predict_ahead]

    This is the standard trick to get many training examples from one sequence.
    """
    T = X.shape[0]
    xs, ys = [], []
    for start in range(0, T - window - predict_ahead + 1, step):
        xs.append(X[start: start + window])
        ys.append(y[start + window + predict_ahead - 1])
    return np.stack(xs, axis=0), np.stack(ys, axis=0)


# ---------------------------------------------------------------------------
# Fold creation and saving
# ---------------------------------------------------------------------------

def save_folds(examples_in: np.ndarray,
               examples_out: np.ndarray,
               n_folds: int,
               dataset_dir: Path,
               seed: int = 42):
    """
    Randomly split examples into n_folds and save each as a pkl.

    Each pkl has keys:
      'inputs'  : (N, T, F)
      'outputs' : (N, n_outputs)
    """
    n = examples_in.shape[0]
    rng   = np.random.default_rng(seed)
    order = rng.permutation(n)

    fold_inds = np.array_split(order, n_folds)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    metadata = {"n_total_examples": int(n),
                "n_folds": int(n_folds),
                "input_shape": list(examples_in.shape[1:]),
                "output_shape": list(examples_out.shape[1:]),
                "folds": []}

    for i, inds in enumerate(fold_inds):
        fold_data = {
            "inputs":  examples_in[inds],
            "outputs": examples_out[inds],
        }
        fold_path = str(dataset_dir / f"fold_{i}.pkl")
        save_pkl(fold_data, fold_path)
        metadata["folds"].append({"fold": i,
                                  "n_examples": int(len(inds)),
                                  "path": fold_path})

    meta_path = dataset_dir / "metadata.json"
    with open(meta_path, "w") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"    [json] {meta_path}")

    return metadata


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    script_dir = Path(__file__).parent.resolve()

    parser = _z2n_create_parser(description="Dr. Su data preparation")

    # Extra arguments specific to this script
    parser.add_argument("--mode", type=str, default="image",
                        choices=["image", "raw"],
                        help="'image' = extract features from PNG frames; "
                             "'raw' = load numerical CSV/Excel from Dr. Su")
    parser.add_argument("--frames_dir", type=str,
                        default=str(script_dir / "Frames"),
                        help="Directory with the PNG frames (mode=image only)")
    parser.add_argument("--time_col", type=str, default=None,
                        help="Name of the timestamp column in the raw CSV (mode=raw, optional)")
    parser.add_argument("--window", type=int, default=10,
                        help="Sliding-window length in time steps (default: 10)")
    parser.add_argument("--window_step", type=int, default=1,
                        help="Step size for the sliding window (default: 1)")
    parser.add_argument("--predict_ahead", type=int, default=1,
                        help="How many steps ahead to predict (default: 1)")

    args = parser.parse_args()

    # Defaults
    if args.dataset_directory is None:
        args.dataset_directory = str(script_dir / "dataset")
    if args.data_n_folds is None:
        args.data_n_folds = 3

    dataset_dir = Path(args.dataset_directory)
    print(f"\n[Dr. Su Data Preparation]")
    print(f"  Mode         : {args.mode}")
    print(f"  Window size  : {args.window} time steps")
    print(f"  Predict ahead: {args.predict_ahead} step(s)")
    print(f"  Folds        : {args.data_n_folds}")
    print(f"  Output dir   : {dataset_dir}\n")

    # -----------------------------------------------------------------------
    # Build the full sequence X (T, F) and y (T, n_outputs)
    # -----------------------------------------------------------------------
    if args.mode == "image":
        print("[1] Mode: IMAGE — extracting pixel features from PNG frames ...")
        frames_dir = Path(args.frames_dir)
        X, ts = build_sequence_from_images(frames_dir)
        T, F  = X.shape
        print(f"    Loaded {T} frames, {F} pixel features each.")

        # In image mode, we use "nonzero_fraction" as a proxy output signal
        # (fraction of bright pixels = proxy for "how much signal is visible")
        # unless the user specifies something else via --data_outputs
        NZF_IDX = 2   # index 2 in the feature vector = nonzero_fraction
        y = X[:, NZF_IDX:NZF_IDX + 1].copy()   # shape (T, 1)

        print(f"    Output (proxy): nonzero_fraction  (index {NZF_IDX} of feature vector)")
        print(f"    NOTE: This is a pixel-space proxy!  Ask Dr. Su for the raw signal.")

    elif args.mode == "raw":
        assert args.data_file is not None, "Must specify --data_file for mode=raw"
        assert args.data_inputs is not None, "Must specify --data_inputs for mode=raw"
        assert args.data_outputs is not None and len(args.data_outputs) == 1, \
            "Specify exactly one --data_outputs column for mode=raw"

        print(f"[1] Mode: RAW — loading {args.data_file} ...")
        X, y, ts = build_sequence_from_csv(args.data_file,
                                           feature_cols=args.data_inputs,
                                           output_col=args.data_outputs[0],
                                           time_col=args.time_col)
        T, F = X.shape
        print(f"    Loaded T={T} time steps, F={F} features.")

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # -----------------------------------------------------------------------
    # Sliding-window augmentation
    # -----------------------------------------------------------------------
    print(f"\n[2] Applying sliding window (size={args.window}, step={args.window_step}) ...")
    X_win, y_win = sliding_window(X, y,
                                  window=args.window,
                                  step=args.window_step,
                                  predict_ahead=args.predict_ahead)
    N = X_win.shape[0]
    print(f"    Created {N} windowed examples  —  shape: {X_win.shape}")

    if N < args.data_n_folds * 3:
        print(f"\n  WARNING: Only {N} examples for {args.data_n_folds} folds.")
        print("  This is TOO SMALL for reliable training.")
        print("  → Request more experiments from Dr. Su, OR")
        print("  → Use --window_step 1 with a larger dataset.")

    # -----------------------------------------------------------------------
    # Save folds
    # -----------------------------------------------------------------------
    print(f"\n[3] Saving {args.data_n_folds} fold pkl files to {dataset_dir} ...")
    metadata = save_folds(X_win, y_win,
                          n_folds=args.data_n_folds,
                          dataset_dir=dataset_dir,
                          seed=getattr(args, "data_seed", 42))

    print("\n[Summary]")
    print(f"  Total examples : {metadata['n_total_examples']}")
    print(f"  Input shape    : {metadata['input_shape']}  (time_steps, features)")
    print(f"  Output shape   : {metadata['output_shape']}")
    print(f"  Folds saved    : {metadata['n_folds']}")
    print("\nDone.\n")


if __name__ == "__main__":
    main()
