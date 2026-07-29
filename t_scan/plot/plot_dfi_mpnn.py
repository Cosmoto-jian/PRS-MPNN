#!/usr/bin/env python3
"""
Plot DFI vs MPNN Z-score per protein
=====================================
Reads dfi_results.h5 and Tmpnn_zscore.h5, matches proteins by ID,
and for each protein plots:
  - DFI_Total (z-score normalized + smoothed, window=3)
  - MPNN z-score for each temperature (already normalized)

Output: results/figs/{protein_id}_dfi_mpnn.png
"""

import os
import sys

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_H5 = os.path.join(REPO_ROOT, "results", "h5")

DFI_H5 = os.path.join(RESULTS_H5, "dfi_results.h5")
TMPNN_H5 = os.path.join(RESULTS_H5, "Tmpnn_zscore.h5")
FIGS_DIR = os.path.join(REPO_ROOT, "results", "figs")

SMOOTH_WINDOW = 3

# Temperature → line color (cool to warm)
TEMP_COLORS = {
    "T_0.05": "#4575b4",
    "T_0.10": "#74add1",
    "T_0.15": "#abd9e9",
    "T_0.20": "#e0f3f8",
    "T_0.30": "#fee090",
    "T_0.50": "#fdae61",
    "T_0.70": "#f46d43",
    "T_1.00": "#d73027",
}
DFI_COLOR = "#313695"       # dark blue for DFI curve
DFI_LINEWIDTH = 2.2
TMPNN_LINEWIDTH = 1.0
TMPNN_ALPHA = 0.85


def zscore(x: np.ndarray) -> np.ndarray:
    """Z-score normalize: (x - mean) / std."""
    x = np.asarray(x, dtype=float)
    std = x.std()
    if std == 0:
        return np.zeros_like(x)
    return (x - x.mean()) / std


def smooth(y: np.ndarray, window: int = 3) -> np.ndarray:
    """Moving average smooth with given window size."""
    if window < 2:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def load_dfi_data(h5_path: str) -> dict[str, np.ndarray]:
    """Load raw DFI_Total arrays keyed by protein ID."""
    data = {}
    with h5py.File(h5_path, "r") as f:
        for pid in f.keys():
            data[pid] = f[pid]["DFI_Total"][:]
    return data


def load_tmpnn_data(h5_path: str) -> dict[str, dict[str, np.ndarray]]:
    """Load MPNN z-score arrays keyed by [protein_id][temperature]."""
    data = {}
    with h5py.File(h5_path, "r") as f:
        for pid in f.keys():
            data[pid] = {}
            for temp in sorted(f[pid].keys()):
                z = f[pid][temp]["mpnn_zscore"]["zscore"][:]
                data[pid][temp] = z
    return data


def plot_one_protein(
    pid: str,
    dfi_raw: np.ndarray,
    tmpnn_temps: dict[str, np.ndarray],
    fig_dir: str,
):
    """Create and save a single-protein comparison plot."""
    n_res = len(dfi_raw)
    x = np.arange(1, n_res + 1)

    # DFI: z-score normalize → smooth
    dfi_z = zscore(dfi_raw)
    dfi_smooth = smooth(dfi_z, window=SMOOTH_WINDOW)

    # ---------------------------------------------------------------
    # Figure
    # ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(max(10, n_res / 25), 5))
    fig.suptitle(f"{pid}: DFI vs MPNN Z-score", fontsize=14, fontweight="bold",
                 y=0.98)

    # ---- DFI curve (thick, on top) ----
    ax.plot(x, dfi_smooth, color=DFI_COLOR, linewidth=DFI_LINEWIDTH,
            label="DFI (z-score, smooth=3)", zorder=10)

    # ---- MPNN temperature curves (smooth=3) ----
    for temp in sorted(tmpnn_temps.keys(), key=_temp_sort_key):
        z = tmpnn_temps[temp]
        if len(z) != n_res:
            print(f"  WARNING: [{pid}/{temp}] length mismatch "
                  f"(DFI={n_res}, MPNN={len(z)}), skipping.")
            continue
        z_smooth = smooth(z, window=SMOOTH_WINDOW)
        color = TEMP_COLORS.get(temp, "#888888")
        ax.plot(x, z_smooth, color=color, linewidth=TMPNN_LINEWIDTH,
                alpha=TMPNN_ALPHA, label=f"MPNN {temp}")

    # ---- Axis labels & ticks ----
    ax.set_xlabel("Residue index", fontsize=12)
    ax.set_ylabel("Z-score", fontsize=12)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)

    # X-axis ticks: show every ~50 residues
    if n_res > 100:
        step = 50
    elif n_res > 50:
        step = 20
    else:
        step = 10
    ax.xaxis.set_major_locator(MultipleLocator(step))
    ax.set_xlim(1, n_res)

    # ---- Legend (outside right) ----
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=8, framealpha=0.7, ncol=1)

    # ---- Save ----
    os.makedirs(fig_dir, exist_ok=True)
    out_path = os.path.join(fig_dir, f"{pid}_dfi_mpnn.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved {out_path}")


def _temp_sort_key(temp_name: str) -> float:
    """Extract numeric temperature from 'T_X.XX' for sorting."""
    return float(temp_name.split("_")[1])


def main():
    print("=" * 60)
    print("DFI vs MPNN Z-score plotting")
    print(f"DFI H5:   {DFI_H5}")
    print(f"Tmpnn H5: {TMPNN_H5}")
    print(f"Output:   {FIGS_DIR}")
    print("=" * 60)

    # Load data
    if not os.path.isfile(DFI_H5):
        print(f"ERROR: DFI file not found: {DFI_H5}")
        sys.exit(1)
    if not os.path.isfile(TMPNN_H5):
        print(f"ERROR: Tmpnn file not found: {TMPNN_H5}")
        sys.exit(1)

    dfi_data = load_dfi_data(DFI_H5)
    tmpnn_data = load_tmpnn_data(TMPNN_H5)

    # Find common protein IDs
    common_ids = sorted(set(dfi_data.keys()) & set(tmpnn_data.keys()))
    if not common_ids:
        print("ERROR: No common protein IDs between the two H5 files.")
        sys.exit(1)

    print(f"\nCommon proteins ({len(common_ids)}): {', '.join(common_ids)}")
    print()

    # Plot each protein
    for pid in common_ids:
        print(f"[{pid}]")
        plot_one_protein(
            pid=pid,
            dfi_raw=dfi_data[pid],
            tmpnn_temps=tmpnn_data[pid],
            fig_dir=FIGS_DIR,
        )

    print(f"\nDone. {len(common_ids)} figures saved to {FIGS_DIR}")


if __name__ == "__main__":
    main()
