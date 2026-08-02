"""
analyse_frames.py
=================
Dr. Su — Fluorescence-Microscopy Time-Series Frame Analysis
Author: generated for ZeroProject / zero2neuro pipeline

PURPOSE
-------
This script is the FIRST step before any ML modelling.
It answers three concrete questions:

  1. What does the spatial signal look like at each time point?
  2. What can we extract as a 1-D numeric time series from the images?
  3. Is the dataset large enough / in the right format to train an RNN
     with zero2neuro, or should we request the raw numerical data from Dr. Su?

All outputs are saved in the same directory as this script.

USAGE
-----
    python analyse_frames.py
    python analyse_frames.py --frames_dir Frames --output_dir analysis_out
"""

import os
import sys
import re
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless / no display required
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_timestamp(filename: str) -> float:
    """Extract the float timestamp (seconds) from a frame filename like '1.20 s.png'."""
    m = re.match(r"([\d.]+)\s*s", os.path.basename(filename))
    if m:
        return float(m.group(1))
    return float("nan")


def load_frame_as_gray(path: str) -> np.ndarray:
    """Load an image and return a float32 grayscale array in [0, 1]."""
    img = Image.open(path).convert("L")          # 8-bit grayscale
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def extract_signal_features(frame: np.ndarray) -> dict:
    """
    Compute scalar descriptors from one grayscale frame.

    These are the features we CAN extract without the raw numerical signal:
      - mean_intensity   : overall brightness
      - max_intensity    : peak pixel value
      - nonzero_fraction : fraction of pixels above threshold (proxy for coverage)
      - centroid_x/y     : brightness-weighted spatial centroid
      - spread_x/y       : standard deviation of brightness-weighted positions
    """
    h, w = frame.shape
    threshold = 0.02          # pixels below this are considered background

    mask = frame > threshold
    nonzero_fraction = mask.mean()

    mean_intensity = float(frame.mean())
    max_intensity  = float(frame.max())

    # Weighted centroid and spread
    if mask.any():
        ys, xs = np.indices(frame.shape)
        weights = frame.copy()
        weights[~mask] = 0.0
        total_w = weights.sum() + 1e-12

        centroid_x = float((weights * xs).sum() / total_w)
        centroid_y = float((weights * ys).sum() / total_w)

        spread_x = float(np.sqrt(((weights * (xs - centroid_x) ** 2).sum()) / total_w))
        spread_y = float(np.sqrt(((weights * (ys - centroid_y) ** 2).sum()) / total_w))
    else:
        centroid_x = centroid_y = float(w / 2)
        spread_x   = spread_y   = 0.0

    return {
        "mean_intensity":   mean_intensity,
        "max_intensity":    max_intensity,
        "nonzero_fraction": float(nonzero_fraction),
        "centroid_x":       centroid_x,
        "centroid_y":       centroid_y,
        "spread_x":         spread_x,
        "spread_y":         spread_y,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_frame_montage(frames_data: list, output_path: str, n_cols: int = 5):
    """Thumbnail grid of all frames in time order."""
    n = len(frames_data)
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 2.2),
                              facecolor="#0d0d0d")
    axes_flat = axes.flatten()

    for i, (t, frame) in enumerate(frames_data):
        ax = axes_flat[i]
        ax.imshow(frame, cmap="inferno", vmin=0, vmax=1)
        ax.set_title(f"t = {t:.2f} s", color="white", fontsize=8)
        ax.axis("off")

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Dr. Su — Fluorescence Frames (time series)", color="white",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, facecolor="#0d0d0d")
    plt.close()
    print(f"  [saved] {output_path}")


def plot_feature_timeseries(timestamps: np.ndarray, features: dict, output_path: str):
    """Multi-panel plot of every extracted scalar feature vs. time."""
    feat_names = list(features.keys())
    n = len(feat_names)

    fig, axes = plt.subplots(n, 1, figsize=(10, 2.5 * n), facecolor="#0d0d0d", sharex=True)
    colors = ["#00d4ff", "#ff6b6b", "#b2ff59", "#ffa500", "#da70d6", "#00ff7f", "#ff8c00"]

    for i, name in enumerate(feat_names):
        ax = axes[i]
        ax.set_facecolor("#0d0d0d")
        ax.plot(timestamps, features[name], color=colors[i % len(colors)],
                linewidth=2, marker="o", markersize=5)
        ax.set_ylabel(name, color="white", fontsize=9)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")
        ax.grid(True, color="#333333", linewidth=0.5)

    axes[-1].set_xlabel("Time (s)", color="white", fontsize=10)
    axes[-1].tick_params(axis="x", colors="white")
    fig.suptitle("Extracted Scalar Features vs. Time", color="white",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, facecolor="#0d0d0d")
    plt.close()
    print(f"  [saved] {output_path}")


def plot_pixel_growth_heatmap(timestamps: np.ndarray, frames: list, output_path: str):
    """
    Average-row projection over time: each column = time, each row = image row average.
    This reveals the spatial structure of growth as a 2-D heatmap.
    """
    rows_over_time = np.stack([f.mean(axis=1) for f in frames], axis=1)  # (H, T)

    fig, ax = plt.subplots(figsize=(12, 5), facecolor="#0d0d0d")
    ax.set_facecolor("#0d0d0d")
    im = ax.imshow(rows_over_time, aspect="auto", origin="upper",
                   cmap="inferno", vmin=0, extent=[timestamps[0], timestamps[-1],
                                                   rows_over_time.shape[0], 0])
    cbar = plt.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Row-average intensity", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_xlabel("Time (s)", color="white", fontsize=11)
    ax.set_ylabel("Image Row (pixels)", color="white", fontsize=11)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

    ax.set_title("Spatial Growth Profile Over Time (row projection)", color="white",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, facecolor="#0d0d0d")
    plt.close()
    print(f"  [saved] {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    script_dir = Path(__file__).parent.resolve()

    parser = argparse.ArgumentParser(description="Dr. Su frame analysis")
    parser.add_argument("--frames_dir", type=str,
                        default=str(script_dir / "Frames"),
                        help="Directory containing *.png time-series frames")
    parser.add_argument("--output_dir", type=str,
                        default=str(script_dir / "analysis_out"),
                        help="Directory to write analysis outputs")
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Discover and sort all frames
    # -----------------------------------------------------------------------
    print("\n[1] Scanning frames directory:", frames_dir)
    png_files = sorted(frames_dir.glob("*.png"),
                       key=lambda p: parse_timestamp(p.name))
    png_files = [p for p in png_files if not np.isnan(parse_timestamp(p.name))]

    if not png_files:
        print("  ERROR: No PNG files found in", frames_dir)
        sys.exit(1)

    print(f"  Found {len(png_files)} frames")

    # -----------------------------------------------------------------------
    # 2. Load frames and extract features
    # -----------------------------------------------------------------------
    print("\n[2] Loading frames and extracting features ...")
    timestamps = []
    frames     = []
    feat_dict  = {k: [] for k in ["mean_intensity", "max_intensity", "nonzero_fraction",
                                   "centroid_x", "centroid_y", "spread_x", "spread_y"]}

    h_ref, w_ref = None, None
    for p in png_files:
        t     = parse_timestamp(p.name)
        frame = load_frame_as_gray(str(p))
        feats = extract_signal_features(frame)

        if h_ref is None:
            h_ref, w_ref = frame.shape
            print(f"  Image size: {w_ref} x {h_ref} px")

        timestamps.append(t)
        frames.append(frame)
        for k, v in feats.items():
            feat_dict[k].append(v)

    timestamps = np.array(timestamps)
    feat_dict  = {k: np.array(v) for k, v in feat_dict.items()}

    dt_intervals = np.diff(timestamps)
    print(f"  Timestamp range: {timestamps[0]:.2f} s  ->  {timestamps[-1]:.2f} s")
    print(f"  dt (uniform?): min={dt_intervals.min():.3f} s  max={dt_intervals.max():.3f} s  "
          f"mean={dt_intervals.mean():.3f} s")

    # -----------------------------------------------------------------------
    # 3. Visualise
    # -----------------------------------------------------------------------
    print("\n[3] Generating visualisations ...")
    frame_data = list(zip(timestamps, frames))
    plot_frame_montage(frame_data,
                       str(output_dir / "frame_montage.png"))
    plot_feature_timeseries(timestamps, feat_dict,
                            str(output_dir / "feature_timeseries.png"))
    plot_pixel_growth_heatmap(timestamps, frames,
                              str(output_dir / "spatial_growth_heatmap.png"))

    # -----------------------------------------------------------------------
    # 4. Feasibility assessment & recommendation
    # -----------------------------------------------------------------------
    n_timesteps = len(timestamps)
    n_features  = len(feat_dict)          # 7 extracted scalar features
    is_uniform  = (dt_intervals.max() - dt_intervals.min()) < 1e-3

    print("\n[4] Feasibility Assessment")
    print("=" * 60)
    print(f"  Frames (time steps)    : {n_timesteps}")
    print(f"  Extracted features     : {n_features}  (pixel-derived scalars)")
    print(f"  Image resolution       : {w_ref} x {h_ref} pixels")
    print(f"  Uniform sampling?      : {'YES' if is_uniform else 'NO (irregular intervals!)'}")
    print(f"  Total sequences avail. : 1  (single experiment)")
    print()
    print("  --- RNN Feasibility with current image-only data ---")
    print()
    print("  VERDICT: FEASIBLE only with severe caveats.")
    print()
    print("  REASON 1 — Only 1 sequence of 25 time steps.")
    print("    An RNN trained on a single sequence cannot generalise.")
    print("    z2n requires many examples (rows in training/val/test).")
    print("    You would need to either:")
    print("      a) Obtain more experiments from Dr. Su (recommended)")
    print("      b) Use sliding-window augmentation to create synthetic")
    print("         sub-sequences (increases N but not truly independent)")
    print()
    print("  REASON 2 — Images vs. raw signal values.")
    print("    The frames are PNG renderings of a signal; they encode the")
    print("    information visually. Pixel-derived scalars (7 features)")
    print("    ARE usable as a feature vector per time step in z2n's RNN,")
    print("    BUT they are lossy proxies.  The raw numerical time-series")
    print("    (e.g. CSV/Excel output of the microscope software) would")
    print("    provide far more accurate features.")
    print()
    print("  REASON 3 --- Data format mismatch.")
    print("    z2n's --data_format='pickle' requires (T, F) shaped arrays")
    print("    where T=timesteps, F=features.  The prepare script in this")
    print("    folder (prepare_drsu_data.py) shows exactly how to produce")
    print("    that file.  It works with EITHER pixel features OR raw values.")
    print()
    print("    RECOMMENDATION:")
    print("    >> ASK Dr. Su for the RAW numerical data (CSV / Excel).")
    print("    >> With raw data you get clean, high-resolution features.")
    print("    >> With multiple experiments you can run a proper RNN/LSTM.")
    print("    >> The config files in this folder are already prepared for")
    print("      both scenarios (image-derived and raw-numerical).")
    print()

    # -----------------------------------------------------------------------
    # 5. Save machine-readable summary
    # -----------------------------------------------------------------------
    summary = {
        "n_frames":       int(n_timesteps),
        "t_start":        float(timestamps[0]),
        "t_end":          float(timestamps[-1]),
        "dt_mean_s":      float(dt_intervals.mean()),
        "dt_uniform":     bool(is_uniform),
        "image_width_px": int(w_ref),
        "image_height_px": int(h_ref),
        "n_pixel_features": int(n_features),
        "extracted_features": list(feat_dict.keys()),
        "rnn_feasibility_with_images_only": "LIMITED — only 1 sequence; request raw data",
        "rnn_feasibility_with_raw_data":    "GOOD — if multiple experiments available",
        "z2n_data_format_needed":           "pickle",
        "z2n_network_type_recommended":     "rnn (lstm or gru)",
        "pixel_feature_timeseries": {
            k: v.tolist() for k, v in feat_dict.items()
        },
        "timestamps_s": timestamps.tolist(),
    }

    summary_path = output_dir / "analysis_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  [saved] {summary_path}")

    print("\nDone. Check the analysis_out/ directory for all outputs.\n")


if __name__ == "__main__":
    main()
