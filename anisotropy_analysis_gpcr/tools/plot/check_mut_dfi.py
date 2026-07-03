#!/usr/bin/env python3
"""
Combined Mutation Landscape + DFI Profile Plot
===============================================
1. Runs dfi_calc.py and mutation_landscape.py to ensure data is current.
2. Draws the mutation landscape (bars + effect icons) as the base layer.
3. Overlays DFI and DFI_membrane curves on a secondary y-axis.
4. Truncates DFI data to match the mutation landscape x-range (1–315).
5. Saves PNG only.

Output
------
  <PROJECT>/results/DFI/figs/<protein>_dfi_landscape.png
"""

import os
import sys
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

# ══════════════════════════════════════════════════════════════════
# 0. Paths / configuration
# ══════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_DIR = os.path.dirname(TOOLS_DIR)

PYTHON = "/opt/anaconda3/envs/simulation_mech/bin/python"

# --- DFI ---
DFI_SCRIPT = os.path.join(TOOLS_DIR, "dfi_calc.py")
PDB_INPUT = os.path.join(PROJECT_DIR, "raw", "pdb", "aa2ar_Inactive.zip")
DFI_CSV = os.path.join(PROJECT_DIR, "results", "DFI", "aa2ar.csv")

# --- Mutation landscape ---
MUT_SCRIPT = os.path.join(TOOLS_DIR, "mutation_landscape.py")
MUT_CSV = os.path.join(PROJECT_DIR, "results", "mutation", "aa2ar_human", "aa2ar_mutation_summary.csv")

# --- Output ---
OUT_DIR = os.path.join(PROJECT_DIR, "results", "DFI", "figs")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PNG = os.path.join(OUT_DIR, "aa2ar_dfi_landscape.png")

PROTEIN_ID = "aa2ar"
SMOOTH_WINDOW = 2  # sliding-window half-width for DFI curve smoothing

TM_REGIONS = {
    "TM1": (8, 32), "TM2": (43, 66), "TM3": (79, 108),
    "TM4": (119, 142), "TM5": (173, 202), "TM6": (235, 258), "TM7": (267, 290),
}
TM_BG_COLOR = "#F4F6F7"

# ══════════════════════════════════════════════════════════════════
# 1. Ensure upstream data
# ══════════════════════════════════════════════════════════════════

def run_if_stale(script, output_csv, input_files):
    """Run script if output CSV is older than any input file."""
    if all(os.path.exists(f) for f in input_files) and os.path.exists(output_csv):
        out_mtime = os.path.getmtime(output_csv)
        if all(os.path.getmtime(f) <= out_mtime for f in input_files):
            print(f"  Up to date: {output_csv}")
            return
    print(f"  Running: {script}")
    result = subprocess.run([PYTHON, script], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Error:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

def ensure_data():
    print("[1/3] Ensuring mutation landscape data ...")
    run_if_stale(MUT_SCRIPT, MUT_CSV,
                 [os.path.join(PROJECT_DIR, "raw", "Mutant", "GPCRdb_aa2ar.xlsx")])

    print("[2/3] Ensuring DFI data ...")
    run_if_stale(DFI_SCRIPT, DFI_CSV, [PDB_INPUT])

# ══════════════════════════════════════════════════════════════════
# 2. Load data
# ══════════════════════════════════════════════════════════════════

def load_data():
    # --- DFI (truncate to mutation x-range: 1–315) ---
    dfi = pd.read_csv(DFI_CSV)
    mask = (dfi["ResI"] >= 1) & (dfi["ResI"] <= 315)
    dfi = dfi[mask].copy()
    dfi_positions = dfi["ResI"].values
    dfi_vals = dfi["DFI"].values / dfi["DFI"].max()  # normalise
    dfi_mem_vals = dfi["DFI_membrane"].values / dfi["DFI_membrane"].max()

    # Smooth via sliding-window average (pad edges with repeat of endpoint)
    def smooth(y, half=1):
        y = np.asarray(y, dtype=float)
        padded = np.pad(y, (half, half), mode="edge")
        window = np.ones(2 * half + 1) / (2 * half + 1)
        return np.convolve(padded, window, mode="valid")

    dfi_vals = smooth(dfi_vals, half=SMOOTH_WINDOW)
    dfi_mem_vals = smooth(dfi_mem_vals, half=SMOOTH_WINDOW)

    # --- Mutation summary ---
    mut_summary = pd.read_csv(MUT_CSV)

    # Build per-position mutation counts and icons (deduplicated across mutations)
    pos_mut_count = {}
    pos_mut_icons = {}
    for _, row in mut_summary.iterrows():
        p = row["Position"]
        pos_mut_count[p] = pos_mut_count.get(p, 0) + 1
        cats_str = row["Effect_Category"]
        if pd.isna(cats_str) or cats_str == "N/A":
            continue
        pos_mut_icons.setdefault(p, set())
        for c in cats_str.split("; "):
            c = c.strip()
            if c in CAT_STYLE:
                pos_mut_icons[p].add(c)

    return dfi_positions, dfi_vals, dfi_mem_vals, pos_mut_count, pos_mut_icons

# ══════════════════════════════════════════════════════════════════
# 3. Plot style
# ══════════════════════════════════════════════════════════════════

CAT_STYLE = {
    "Increased >10fold":  {"marker": "*", "color": "#B2182B", "size": 5,  "label": "Increased >10-fold"},
    "Increased 5-10fold": {"marker": "*", "color": "#EF8A62", "size": 4,  "label": "Increased 5-10-fold"},
    "Reduced >10fold":    {"marker": "v", "color": "#2166AC", "size": 4,  "label": "Reduced >10-fold"},
    "Reduced 5-10fold":   {"marker": "v", "color": "#67A9CF", "size": 3.5,"label": "Reduced 5-10-fold"},
    "No/low effect":      {"marker": "o", "color": "#F4A300", "size": 3,  "label": "No/low effect (<5-fold)"},
    "No effect":          {"marker": "o", "color": "#BDC3C7", "size": 3,  "label": "No effect"},
}

TM_CMAP_COLORS = ["#AEB6BF", "#2C3E50"]
LOOP_CMAP_COLORS = ["#D5D8DC", "#7F8C8D"]

RC_PARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8,
    "axes.linewidth": 0.5,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.fontsize": 6.5,
    "legend.frameon": True,
    "legend.edgecolor": "#CCCCCC",
    "legend.fancybox": False,
    "legend.borderpad": 0.3,
    "legend.handlelength": 1.0,
    "legend.handletextpad": 0.4,
}

def is_tm(pos):
    for a, b in TM_REGIONS.values():
        if a <= pos <= b:
            return True
    return False

# ══════════════════════════════════════════════════════════════════
# 4. Combined plot
# ══════════════════════════════════════════════════════════════════

def make_combined_plot(dfi_positions, dfi_vals, dfi_mem_vals,
                       pos_mut_count, pos_mut_icons):
    matplotlib.rcParams.update(RC_PARAMS)

    positions = sorted(pos_mut_count.keys())
    bar_heights = np.array([pos_mut_count[p] for p in positions])
    max_bar = max(bar_heights)
    norm_counts = bar_heights / max_bar

    tm_cmap = mcolors.LinearSegmentedColormap.from_list("tm_cmap", TM_CMAP_COLORS)
    loop_cmap = mcolors.LinearSegmentedColormap.from_list("loop_cmap", LOOP_CMAP_COLORS)

    bar_colors = []
    for p, nc in zip(positions, norm_counts):
        cmap = tm_cmap if is_tm(p) else loop_cmap
        bar_colors.append(cmap(0.15 + 0.85 * nc))

    x_min, x_max = 1, 315

    # Calculate max icons at any position for y-limit
    max_icons_at_pos = max((len(v) for v in pos_mut_icons.values()), default=0)
    y_bar_max = max_bar + max_icons_at_pos * 0.5 + 1.5

    fig, ax1 = plt.subplots(figsize=(7.2, 4.5))
    fig.subplots_adjust(left=0.10, right=0.84, top=0.78, bottom=0.16)

    # ── Left axis: mutation counts ──
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(0, y_bar_max)
    ax1.set_xlabel("Residue position", fontsize=9)
    ax1.set_ylabel("Number of mutations", fontsize=9)
    ax1.yaxis.set_major_locator(MultipleLocator(2))

    # TM background bands
    for tm_name, (a, b) in TM_REGIONS.items():
        ax1.axvspan(a, b, color=TM_BG_COLOR, alpha=0.55, zorder=0)
        ax1.text((a + b) / 2, y_bar_max * 0.985, tm_name, ha="center", va="top",
                 fontsize=5.5, color="#5D6D7E", fontweight="bold", zorder=5)

    # Bars
    ax1.bar(positions, bar_heights, width=0.6, color=bar_colors,
            edgecolor="white", linewidth=0.15, zorder=2)

    # Icons above bars (deduplicated per position)
    ICON_SPACING = 0.5
    icon_drawn_positions = 0
    for pos in positions:
        if pos not in pos_mut_icons:
            continue
        bar_top = pos_mut_count[pos]
        y_cursor = bar_top + 0.4
        for icon_cat in sorted(pos_mut_icons[pos]):
            style = CAT_STYLE[icon_cat]
            ax1.plot(pos, y_cursor, marker=style["marker"], color=style["color"],
                     markersize=style["size"], zorder=10,
                     markeredgecolor="white", markeredgewidth=0.15)
            y_cursor += ICON_SPACING
        icon_drawn_positions += 1

    # ── Right axis: DFI curves ──
    ax2 = ax1.twinx()
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel("Normalised DFI", fontsize=9)

    line1, = ax2.plot(dfi_positions, dfi_vals, color="#2166AC", linewidth=1.2,
                      zorder=3, label="DFI (isotropic)")
    line2, = ax2.plot(dfi_positions, dfi_mem_vals, color="#B2182B", linewidth=1.2,
                      zorder=4, label=r"DFI$_{membrane}$ (imANM)")

    # ── Title ──
    ax1.set_title(r"Mutation landscape & DFI — A$_{2A}$ adenosine receptor",
                  fontsize=10, fontweight="bold", pad=8)

    # ── Axis styling ──
    ax1.xaxis.set_major_locator(MultipleLocator(50))
    ax1.xaxis.set_minor_locator(MultipleLocator(10))
    ax1.spines["top"].set_visible(False)
    ax1.set_axisbelow(True)
    ax1.grid(axis="y", linewidth=0.3, color="#E0E0E0", zorder=0)
    ax2.spines["top"].set_visible(False)

    # ── Combined legend (icons + DFI lines) ──
    icon_handles = []
    for cat in ["Increased >10fold", "Increased 5-10fold", "Reduced >10fold",
                "Reduced 5-10fold", "No/low effect", "No effect"]:
        s = CAT_STYLE[cat]
        icon_handles.append(Line2D([0], [0], marker=s["marker"], color="w",
                                    markerfacecolor=s["color"], markersize=s["size"] * 0.7,
                                    markeredgecolor="white", markeredgewidth=0.2,
                                    label=s["label"]))
    leg1 = ax1.legend(handles=icon_handles, loc="lower left", fontsize=5.5,
                      title="Effect on binding/potency", title_fontsize=6,
                      framealpha=0.95, edgecolor="#CCCCCC", ncol=6,
                      bbox_to_anchor=(0.0, 1.15), borderpad=0.3,
                      handlelength=1, handletextpad=0.3, columnspacing=1)
    ax1.add_artist(leg1)

    # DFI legend below icons
    dFI_handles = [line1, line2]
    leg2 = ax1.legend(handles=dFI_handles, loc="lower left", fontsize=6,
                      framealpha=0.95, edgecolor="#CCCCCC", ncol=2,
                      bbox_to_anchor=(0.68, 1.15), borderpad=0.3,
                      handlelength=1.2, handletextpad=0.3, columnspacing=1)
    ax1.add_artist(leg2)

    # ── Save ──
    counts = [pos_mut_count[p] for p in positions]
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {OUT_PNG}")
    print(f"  Positions with mutations: {len(positions)}, total mutations: {sum(counts)}")
    print(f"  Positions with icons: {icon_drawn_positions}")
    print(f"  DFI points: {len(dfi_positions)} (truncated to 1–315)")

# ══════════════════════════════════════════════════════════════════
# 5. Main
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ensure_data()
    print("[3/3] Generating combined plot ...")
    dfi_positions, dfi_vals, dfi_mem_vals, pos_mut_count, pos_mut_icons = load_data()
    make_combined_plot(dfi_positions, dfi_vals, dfi_mem_vals, pos_mut_count, pos_mut_icons)
