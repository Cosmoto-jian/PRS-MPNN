"""
A2A Adenosine Receptor (aa2ar_human) Mutation Landscape
=========================================================
Reads GPCRdb mutant effect data, classifies each mutation's functional
effect into three INDEPENDENT directions (increased / reduced / low-low
effect), keeps only the STRONGEST icon within each direction, and draws
a publication-ready (Nature/Science double-column style) bar+icon plot.

Icon-merging rule
------------------
For each mutation, up to 3 icons can appear simultaneously:
  1) Increased effect  -> only the strongest tier shown
                           (">10-fold" beats "5-10-fold")
  2) Reduced effect     -> only the strongest tier shown
                           (">10-fold" beats "5-10-fold")
  3) No/low effect (<5-fold) -> shown once if present
These three are independent and can co-occur, but within each direction
only the single strongest instance is drawn (no duplicate same-color icons).

Output
------
  <OUT_DIR>/aa2ar_mutation_landscape.png   (300 dpi raster)
<OUT_DIR>/aa2ar_mutation_summary.csv     (per-mutation summary table)
"""

import os
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

# ══════════════════════════════════════════════════════════════════
# 0. Paths / configuration
# ══════════════════════════════════════════════════════════════════

INPUT_XLSX = "/Users/waltry/Desktop/PRS-MPNN/anisotropy_analysis_gpcr/raw/Mutant/GPCRdb_aa2ar.xlsx"
SHEET_NAME = "Sheet1"
PROTEIN_ID = "aa2ar_human"
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "results", "mutation", PROTEIN_ID
)
os.makedirs(OUT_DIR, exist_ok=True)

TM_REGIONS = {
    "TM1": (8, 32), "TM2": (43, 66), "TM3": (79, 108),
    "TM4": (119, 142), "TM5": (173, 202), "TM6": (235, 258), "TM7": (267, 290),
}

# ══════════════════════════════════════════════════════════════════
# 1. Data loading & row-level classification
# ══════════════════════════════════════════════════════════════════

def load_and_classify(xlsx_path, sheet_name, protein_id):
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    d = df[df["protein"] == protein_id].copy()
    d["mut_id"] = d["mutation_from"] + d["mutation_pos"].astype(str) + d["mutation_to"]

    def row_category(row):
        q = row["exp_mu_effect_qual"]
        fc = row["exp_fold_change"]
        if pd.notna(q) and q in ("Abolished", "Abolished effect"):
            return "N/A"
        if pd.notna(q) and q == "Gain of effect (wt had no effect)":
            return "N/A"
        if pd.notna(q) and q == "No effect":
            return "No effect"
        if fc > 10:
            return "Increased >10fold"
        if fc > 5:
            return "Increased 5-10fold"
        if fc < -10:
            return "Reduced >10fold"
        if fc < -5:
            return "Reduced 5-10fold"
        if fc != 0:
            return "No/low effect"
        return "N/A"

    d["category"] = d.apply(row_category, axis=1)
    print(f"Loaded {len(d)} rows, {d['mut_id'].nunique()} distinct mutations, "
          f"{d['mutation_pos'].nunique()} positions")
    print("\nRow-level category distribution:")
    print(d["category"].value_counts())
    return d


# ══════════════════════════════════════════════════════════════════
# 2. Per-mutation aggregation
#    Rule: 3 independent directions (increase / reduce / low-effect),
#    each direction contributes AT MOST ONE icon = its strongest member.
# ══════════════════════════════════════════════════════════════════

def aggregate_mutation(g):
    cats = set(g["category"])
    icons, rep_fcs = [], []

    # --- Increased direction: strongest wins, no duplicate ---
    if "Increased >10fold" in cats:
        icons.append("Increased >10fold")
        rep_fcs.append(g.loc[g["category"] == "Increased >10fold", "exp_fold_change"].max())
    elif "Increased 5-10fold" in cats:
        icons.append("Increased 5-10fold")
        rep_fcs.append(g.loc[g["category"] == "Increased 5-10fold", "exp_fold_change"].max())

    # --- Reduced direction: strongest (most negative) wins, no duplicate ---
    if "Reduced >10fold" in cats:
        icons.append("Reduced >10fold")
        rep_fcs.append(g.loc[g["category"] == "Reduced >10fold", "exp_fold_change"].min())
    elif "Reduced 5-10fold" in cats:
        icons.append("Reduced 5-10fold")
        rep_fcs.append(g.loc[g["category"] == "Reduced 5-10fold", "exp_fold_change"].min())

    # --- No/low effect: independent, shown once if present ---
    if "No/low effect" in cats:
        icons.append("No/low effect")
        rep_fcs.append(g.loc[g["category"] == "No/low effect", "exp_fold_change"].abs().max())

    # --- Fallback: only if nothing else qualifies ---
    if not icons and "No effect" in cats:
        icons.append("No effect")
        rep_fcs.append(0)

    return icons, rep_fcs


def get_tm_label(pos, tm_regions):
    for name, (a, b) in tm_regions.items():
        if a <= pos <= b:
            return name
    return "loop"


def is_tm(pos, tm_regions):
    return get_tm_label(pos, tm_regions) != "loop"


def build_mut_data(d, tm_regions):
    mut_data = {}
    for mid, g in d.groupby("mut_id"):
        icons, rep_fcs = aggregate_mutation(g)
        pos = g["mutation_pos"].iloc[0]
        generic = g["generic"].dropna().iloc[0] if g["generic"].notna().any() else ""
        mut_data[mid] = {
            "pos": pos,
            "generic": generic,
            "icons": icons,
            "rep_fcs": rep_fcs,
            "n_ligands": g["ligand_name"].dropna().nunique(),
            "n_exp": len(g),
            "tm_label": get_tm_label(pos, tm_regions),
            "is_tm": is_tm(pos, tm_regions),
        }
    return mut_data


def build_position_indices(mut_data):
    pos_mut_count, pos_mut_icons, pos_icon_counts = {}, {}, {}
    for mid, info in mut_data.items():
        p = info["pos"]
        pos_mut_count[p] = pos_mut_count.get(p, 0) + 1
        pos_mut_icons.setdefault(p, [])
        if info["icons"]:
            pos_mut_icons[p].append((mid, info["icons"]))
        pos_icon_counts[p] = pos_icon_counts.get(p, 0) + len(info["icons"])
    return pos_mut_count, pos_mut_icons, pos_icon_counts


# ══════════════════════════════════════════════════════════════════
# 3. Plot styling — colorblind-safe palette, Nature/Science conventions
# ══════════════════════════════════════════════════════════════════

# Red (increase) vs Blue (reduce) instead of red/green — safe for the
# most common (red-green) forms of color vision deficiency. Shape
# encoding (star / triangle / circle) is kept as a redundant channel.
CAT_STYLE = {
    "Increased >10fold":  {"marker": "*", "color": "#B2182B", "size": 5,  "label": "Increased >10-fold"},
    "Increased 5-10fold": {"marker": "*", "color": "#EF8A62", "size": 4,  "label": "Increased 5-10-fold"},
    "Reduced >10fold":    {"marker": "v", "color": "#2166AC", "size": 4,  "label": "Reduced >10-fold"},
    "Reduced 5-10fold":   {"marker": "v", "color": "#67A9CF", "size": 3.5,"label": "Reduced 5-10-fold"},
    "No/low effect":      {"marker": "o", "color": "#F4A300", "size": 3,  "label": "No/low effect (<5-fold)"},
    "No effect":          {"marker": "o", "color": "#BDC3C7", "size": 3,  "label": "No effect"},
}

TM_CMAP_COLORS = ["#AEB6BF", "#2C3E50"]      # light -> dark, TM bars
LOOP_CMAP_COLORS = ["#D5D8DC", "#7F8C8D"]    # light -> dark, non-TM bars
TM_BG_COLOR = "#F4F6F7"

NATURE_RC = {
    "font.family": "Liberation Sans",   # open-source Arial-equivalent, avoids font warnings
    "font.size": 8,
    "axes.linewidth": 0.5,
    "axes.labelsize": 9,
    "axes.titlesize": 11,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,     # embed/outline fonts as TrueType, not Type-3 bitmaps
    "legend.fontsize": 7,
    "legend.frameon": True,
    "legend.edgecolor": "#CCCCCC",
    "legend.fancybox": False,
    "legend.borderpad": 0.4,
    "legend.handlelength": 1.2,
    "legend.handletextpad": 0.5,
}


def make_plot(mut_data, pos_mut_count, pos_mut_icons, pos_icon_counts,
              tm_regions, out_prefix, panel_label="a"):
    matplotlib.rcParams.update(NATURE_RC)

    positions = sorted(pos_mut_count.keys())
    bar_heights = np.array([pos_mut_count[p] for p in positions])
    max_count = max(bar_heights)
    norm_counts = bar_heights / max_count  # 0..1 for colormap lookup

    tm_cmap = mcolors.LinearSegmentedColormap.from_list("tm_cmap", TM_CMAP_COLORS)
    loop_cmap = mcolors.LinearSegmentedColormap.from_list("loop_cmap", LOOP_CMAP_COLORS)

    bar_colors = []
    for p, nc in zip(positions, norm_counts):
        cmap = tm_cmap if is_tm(p, tm_regions) else loop_cmap
        bar_colors.append(cmap(0.15 + 0.85 * nc))  # avoid the very-lightest edge

    bar_width = 0.6  # slightly narrower than min position gap (1) for clearer separation

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.74, bottom=0.16)

    x_min, x_max = 1, 315
    ax.set_xlim(x_min, x_max)

    max_bar = max(bar_heights)
    max_icons_at_pos = max(pos_icon_counts.values())
    y_max = max_bar + max_icons_at_pos * 0.7 + 1.8
    ax.set_ylim(0, y_max)
    ax.yaxis.set_major_locator(MultipleLocator(1))

    # TM background bands
    for tm_name, (a, b) in tm_regions.items():
        ax.axvspan(a, b, color=TM_BG_COLOR, alpha=0.6, zorder=0)
        ax.text((a + b) / 2, y_max * 0.985, tm_name, ha="center", va="top",
                 fontsize=6, color="#5D6D7E", fontweight="bold", zorder=5)

    # Bars
    ax.bar(positions, bar_heights, width=bar_width, color=bar_colors,
           edgecolor="white", linewidth=0.15, zorder=2)

    # Subtle visual separator between "count" region (bars) and "effect" region (icons)
    # so the two encodings on the shared y-axis read as distinct panels.
    ax.axhline(0, color="#999999", linewidth=0.4, zorder=1)

    # Icons above bars — deduplicate same-category icons at each position
    ICON_SPACING = 0.5
    for pos in positions:
        if pos not in pos_mut_icons:
            continue
        bar_top = pos_mut_count[pos]
        y_cursor = bar_top + 0.4
        drawn_cats = set()
        for mid, icons in pos_mut_icons[pos]:
            for icon_cat in icons:
                if icon_cat in drawn_cats:
                    continue
                drawn_cats.add(icon_cat)
                style = CAT_STYLE[icon_cat]
                ax.plot(pos, y_cursor, marker=style["marker"], color=style["color"],
                        markersize=style["size"], zorder=10,
                        markeredgecolor="white", markeredgewidth=0.15)
                y_cursor += ICON_SPACING

    # Axis labels / title
    ax.set_xlabel("Residue position", fontsize=9)
    ax.set_ylabel("Number of mutations", fontsize=9)
    ax.set_title(r"Mutation landscape of A$_{2A}$ adenosine receptor",
                 fontsize=11, fontweight="bold", pad=8)

    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_minor_locator(MultipleLocator(10))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linewidth=0.3, color="#E0E0E0", zorder=0)

    # ── Panel label (Nature/Science convention: bold, top-left, outside axes) ──
    ax.text(-0.06, 1.32, panel_label, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="left")

    # ── Sample-size annotation ──
    n_positions = len(positions)
    n_mutations = len(mut_data)
    n_icons = sum(len(v["icons"]) for v in mut_data.values())
    ax.text(0.995, -0.30,
            f"n = {n_mutations} mutations across {n_positions} positions "
            f"({n_icons} effect icons); source: GPCRdb",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=6, color="#555555", style="italic")

    # ── Icon legend (shape + color, redundant encoding) ──
    icon_handles = []
    for cat in ["Increased >10fold", "Increased 5-10fold", "Reduced >10fold",
                "Reduced 5-10fold", "No/low effect", "No effect"]:
        s = CAT_STYLE[cat]
        icon_handles.append(Line2D([0], [0], marker=s["marker"], color="w",
                                    markerfacecolor=s["color"], markersize=s["size"] * 0.7,
                                    markeredgecolor="white", markeredgewidth=0.2,
                                    label=s["label"]))
    leg1 = ax.legend(handles=icon_handles, loc="lower left", fontsize=6,
                      title="Effect on binding/potency", title_fontsize=7,
                      framealpha=0.95, edgecolor="#CCCCCC", ncol=6,
                      bbox_to_anchor=(0.0, 1.20), borderpad=0.4,
                      handlelength=1, handletextpad=0.4, columnspacing=1)
    ax.add_artist(leg1)

    # Save raster format
    fig.savefig(f"{out_prefix}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Plot saved: {out_prefix}.png")
    print(f"y_max={y_max:.2f}, bar_width={bar_width}, total icons={n_icons}, "
          f"positions={n_positions}, mutations={n_mutations}")


# ══════════════════════════════════════════════════════════════════
# 4. Summary table
# ══════════════════════════════════════════════════════════════════

def build_summary_table(mut_data, out_csv):
    rows = []
    for mid, info in sorted(mut_data.items(), key=lambda x: x[1]["pos"]):
        if not info["icons"]:
            icon_str, rep_fc_str = "N/A", ""
        else:
            icon_str = "; ".join(info["icons"])
            rep_fc_str = "; ".join(f"{fc:.1f}" for fc in info["rep_fcs"])
        rows.append({
            "Mutation": mid,
            "Position": info["pos"],
            "Generic_Number": info["generic"],
            "Region": info["tm_label"],
            "In_TM": "Yes" if info["is_tm"] else "No",
            "Effect_Category": icon_str,
            "Representative_FoldChange": rep_fc_str,
            "Num_Ligands": info["n_ligands"],
            "Num_Experiments": info["n_exp"],
        })
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_csv, index=False)
    print(f"\nSummary table saved: {out_csv} ({len(summary_df)} rows)")
    print("\nEffect category distribution:")
    print(summary_df["Effect_Category"].value_counts().head(12).to_string())
    return summary_df


# ══════════════════════════════════════════════════════════════════
# 5. Main
# ══════════════════════════════════════════════════════════════════

def main():
    d = load_and_classify(INPUT_XLSX, SHEET_NAME, PROTEIN_ID)
    mut_data = build_mut_data(d, TM_REGIONS)
    pos_mut_count, pos_mut_icons, pos_icon_counts = build_position_indices(mut_data)

    icon_dist = pd.Series([len(v["icons"]) for v in mut_data.values()]).value_counts().sort_index()
    print("\nIcons-per-mutation distribution (0-3 expected):")
    print(icon_dist.to_string())

    out_prefix = os.path.join(OUT_DIR, "aa2ar_mutation_landscape")
    make_plot(mut_data, pos_mut_count, pos_mut_icons, pos_icon_counts,
              TM_REGIONS, out_prefix, panel_label="a")

    build_summary_table(mut_data, os.path.join(OUT_DIR, "aa2ar_mutation_summary.csv"))


if __name__ == "__main__":
    main()
