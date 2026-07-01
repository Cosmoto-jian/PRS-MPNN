#!/usr/bin/env python3
"""
每个蛋白质生成一张大图（2行×2列），共4个子图，对应4种条件：
  EXP   : 有膜 + DFI_XY（显示 ratio）
  CTRL1 : 无膜 + DFI_XY（显示 ratio）
  CTRL2 : 无膜 + DFI_Total（不显示 ratio）
  CTRL3 : 有膜 + DFI_Total（不显示 ratio）

突变位点直接从 mutations.csv 的 pertain 列匹配（使用蛋白质简写）
"""

import os
import re
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ========== 配置 ==========
MUTATION_DIR = '/Users/zizhenghe/Documents/DFI/mutation'   # 存放突变汇总文件的目录
MUTATION_SUMMARY_FILE = os.path.join(MUTATION_DIR, 'mutations.csv')
# 映射表不再需要，但保留以防备用
MAPPING_EXCEL = os.path.join(MUTATION_DIR, 'gpcrdb_receptor_list.xlsx')

# 跨膜区（键为蛋白质简写）
GPCR_TM_REGIONS = {
    "aa2ar": [(8, 32), (43, 66), (79, 108), (119, 142), (173, 202), (235, 258), (267, 290)],
    "adrb2": [(29, 60), (67, 96), (103, 136), (147, 171), (197, 229), (267, 298), (305, 331)],
    "cxcr4": [(39, 68), (78, 106), (112, 143), (157, 180), (203, 229), (240, 268), (280, 305)],
    "aa1r": [(12, 32), (48, 68), (78, 98), (128, 147), (174, 196), (237, 259), (268, 289)],
    "aa2br": [(9, 33), (44, 67), (79, 101), (122, 144), (179, 203), (236, 259), (268, 291)],
    "acm1": [(23, 48), (63, 84), (105, 126), (143, 164), (187, 210), (351, 372), (385, 407)],
    "ccr1": [(35, 60), (73, 95), (108, 129), (151, 175), (198, 223), (240, 264), (282, 305)],
    "ccr5": [(31, 58), (69, 89), (103, 124), (142, 166), (199, 218), (236, 261), (271, 295)],
    "cxcr1": [(39, 65), (76, 96), (111, 132), (153, 176), (198, 220), (243, 267), (277, 302)],
    "glp1r": [(145, 165), (174, 194), (228, 248), (271, 291), (317, 337), (350, 370), (383, 403)],
    "mc4r": [(44, 64), (77, 97), (115, 135), (154, 174), (196, 216), (247, 267), (280, 300)],
}

DFI_TOTAL_COLOR = '#FFB266'   # 黄色
DFI_XY_COLOR    = '#FF6B6B'   # 橙红
RATIO_COLOR     = '#9933FF'   # 紫色

CONDITIONS = [
    {
        'suffix':    '_EXP',
        'label':     'Exp: membrane + DFI_XY',
        'dfi_cols':  [('DFI_XY', DFI_XY_COLOR, 'DFI_XY')],
        'ratio_col': 'Effector_Ratio_Z',
        'show_ratio': True,
    },
    {
        'suffix':    '_CTRL1',
        'label':     'Ctrl1: no membrane + DFI_XY',
        'dfi_cols':  [('DFI_XY', DFI_XY_COLOR, 'DFI_XY')],
        'ratio_col': 'Effector_Ratio_Z',
        'show_ratio': True,
    },
    {
        'suffix':    '_CTRL2',
        'label':     'Ctrl2: no membrane + DFI_Total',
        'dfi_cols':  [('DFI_Total', DFI_TOTAL_COLOR, 'DFI_Total')],
        'ratio_col': None,
        'show_ratio': False,
    },
    {
        'suffix':    '_CTRL3',
        'label':     'Ctrl3: membrane + DFI_Total',
        'dfi_cols':  [('DFI_Total', DFI_TOTAL_COLOR, 'DFI_Total')],
        'ratio_col': None,
        'show_ratio': False,
    },
]

# ---------- 突变解析（直接匹配蛋白质简写） ----------
def parse_mutation_sites(summary_file, protein_shorthand):
    """
    从 mutations.csv 中读取指定蛋白质简写的突变位点。
    返回字典：{'red': [pos1, pos2, ...], 'green': [...]}
    """
    if not os.path.exists(summary_file):
        print(f"  Warning: Summary mutation file not found: {summary_file}")
        return {}

    df = pd.read_csv(summary_file, encoding='utf-8')
    df.columns = [c.strip() for c in df.columns]
    if 'pertain' not in df.columns:
        print(f"  Error: 'pertain' column not found in {summary_file}")
        return {}

    # 匹配蛋白质简写（忽略大小写）
    protein_lower = protein_shorthand.lower()
    row = df[df['pertain'].astype(str).str.lower() == protein_lower]
    if row.empty:
        print(f"  Warning: protein shorthand '{protein_shorthand}' not found in mutation file")
        return {}

    sites = {}
    for col in ['red', 'green']:
        if col in df.columns:
            residue_str = row.iloc[0][col]
            if pd.isna(residue_str) or residue_str == '':
                sites[col] = []
                continue
            residues = []
            # 支持用空格或逗号或-分隔
            for token in re.split(r'[,\s-]+', str(residue_str)):
                if token.strip():
                    nums = re.findall(r'\d+', token)
                    if nums:
                        residues.append(int(nums[0]))
            sites[col] = sorted(set(residues))
    print(f"  Loaded mutations: red={sites.get('red', [])}, green={sites.get('green', [])}")
    return sites

# ---------- 绘图子函数 ----------
def plot_subplot(ax, resi, dfi_series, ratio_vals,
                 tm_regions, mut_sites, condition_label,
                 first_tm_start, last_tm_end, show_ratio=True):
    ax_left  = ax
    ax_right = ax.twinx()

    for start, end in tm_regions:
        ax_left.axvspan(start, end, alpha=0.15, color='gray', zorder=0)

    for vals, color, lbl in dfi_series:
        ax_left.plot(resi, vals, color=color, linestyle='-',
                     linewidth=1.5, label=lbl)

    ax_left.set_xlabel('Residue index', fontsize=9)
    ax_left.set_ylabel('DFI', color='black', fontsize=9)
    ax_left.tick_params(axis='y', labelcolor='black')
    ax_left.set_xlim(resi.min(), resi.max())

    if show_ratio and ratio_vals is not None:
        one_minus_ratio = 1.0 - ratio_vals
        ax_right.plot(resi, one_minus_ratio, color=RATIO_COLOR,
                      linestyle='-', linewidth=1.5,
                      label='1 − Effector_Ratio_Z')
        ax_right.set_ylabel('1 − Effector_Ratio_Z', color='black', fontsize=9)
        ax_right.tick_params(axis='y', labelcolor='black')
    else:
        ax_right.set_yticks([])
        ax_right.set_ylabel('')

    # 绘制突变标记（仅在TM区内）
    if mut_sites and first_tm_start is not None and last_tm_end is not None:
        y_min, y_max = ax_left.get_ylim()
        y_offset  = 0.08 * (y_max - y_min)
        y_marker  = y_min - y_offset
        for key, color_marker, marker in [('green', '#7DB171', '^'),
                                           ('red',   '#E63323', 's')]:
            if key in mut_sites:
                for r in mut_sites[key]:
                    if first_tm_start <= r <= last_tm_end:
                        ax_left.scatter(r, y_marker,
                                        color=color_marker, marker=marker,
                                        s=30, edgecolor='k', linewidth=0.5,
                                        zorder=10, clip_on=False)
        ax_left.set_ylim(y_min - 1.8 * y_offset, y_max)

    ax_left.set_title(condition_label, fontsize=10)
    return ax_left, ax_right

# ---------- 主生成函数 ----------
def generate_dashboards(protein_list=None, data_dir='./', output_dir='./'):
    if protein_list is None:
        # 默认使用字典中的所有键（蛋白质简写）
        protein_list = list(GPCR_TM_REGIONS.keys())

    os.makedirs(output_dir, exist_ok=True)

    for protein in protein_list:
        print(f"\nProcessing {protein} ...")
        df_dict = {}
        for cond in CONDITIONS:
            fname = os.path.join(data_dir, f"{protein}{cond['suffix']}.csv")
            if os.path.exists(fname):
                df_dict[cond['suffix']] = pd.read_csv(fname)
                print(f"  Loaded: {fname}")
            else:
                print(f"  Warning: {fname} not found")

        if not df_dict:
            print(f"  No data for {protein}, skipping.")
            continue

        # ---- 直接读取突变（使用蛋白质简写） ----
        print(f"  Using protein shorthand: {protein}")
        mut_sites = parse_mutation_sites(MUTATION_SUMMARY_FILE, protein)

        # ---- 跨膜区 ----
        tm_regions = GPCR_TM_REGIONS.get(protein, [])
        if tm_regions:
            first_tm_start = tm_regions[0][0]
            last_tm_end    = tm_regions[-1][1]
        else:
            print(f"  Warning: No TM regions defined for {protein}.")
            first_tm_start = last_tm_end = None

        # ---- 截取残基范围（仅跨膜区） ----
        first_df = next(iter(df_dict.values()))
        if 'ResI' not in first_df.columns:
            print(f"  Error: 'ResI' column missing in data for {protein}. Skipping.")
            continue
        resi_all = first_df['ResI'].values.astype(int)
        if first_tm_start is not None:
            mask = (resi_all >= first_tm_start) & (resi_all <= last_tm_end)
        else:
            mask = np.ones(len(resi_all), dtype=bool)
        resi = resi_all[mask]
        if len(resi) == 0:
            print(f"  Warning: No residues in TM region for {protein}. Skipping.")
            continue

        # ---- 绘图 ----
        with plt.style.context('default'):
            fig, axes = plt.subplots(2, 2, figsize=(14, 9))
            fig.suptitle(
                f"{protein} – DFI & 1−Effector_Ratio_Z (4 conditions)",
                fontsize=13, fontweight='bold'
            )

            for idx, cond in enumerate(CONDITIONS):
                row_idx = idx // 2
                col_idx = idx % 2
                ax = axes[row_idx, col_idx]
                suffix = cond['suffix']

                if suffix not in df_dict:
                    ax.text(0.5, 0.5, 'No data',
                            ha='center', va='center',
                            transform=ax.transAxes, fontsize=12)
                    ax.set_title(cond['label'], fontsize=10)
                    continue

                df   = df_dict[suffix]
                # 只取TM区域的行
                tmdf = df.loc[mask] if len(df) == len(resi_all) else df
                # 确保 tmdf 与 resi 对齐
                if len(tmdf) != len(resi):
                    # 如果mask过滤后长度不一致，重新按ResI过滤
                    tmdf = df[df['ResI'].isin(resi)]

                dfi_series = []
                for col_name, color, lbl in cond['dfi_cols']:
                    if col_name in tmdf.columns:
                        dfi_series.append((tmdf[col_name].values, color, lbl))
                    else:
                        print(f"  [Warning] Column '{col_name}' missing in "
                              f"{protein}{suffix}.csv")

                ratio_col  = cond['ratio_col']
                ratio_vals = (tmdf[ratio_col].values
                              if ratio_col and ratio_col in tmdf.columns
                              else None)
                show_ratio = cond['show_ratio']

                plot_subplot(
                    ax, resi, dfi_series, ratio_vals,
                    tm_regions, mut_sites, cond['label'],
                    first_tm_start, last_tm_end,
                    show_ratio=show_ratio,
                )

            # ---- 图例 ----
            legend_elements = [
                Line2D([0],[0], color=DFI_TOTAL_COLOR, linewidth=1.5,
                       label='DFI (Total)'),
                Line2D([0],[0], color=DFI_XY_COLOR,  linewidth=1.5,
                       label='DFI (XY)'),
                Line2D([0],[0], color=RATIO_COLOR,  linewidth=1.5,
                       label='1 − Effector_Ratio_Z'),
                Patch(facecolor='gray', alpha=0.3,
                      label='Transmembrane region'),
                Line2D([0],[0], marker='^', color='w',
                       markerfacecolor='#7DB171', markeredgecolor='k',
                       markersize=8, label='Increased binding >10-fold'),
                Line2D([0],[0], marker='s', color='w',
                       markerfacecolor='#E63323', markeredgecolor='k',
                       markersize=8, label='Reduced binding >10-fold'),
            ]
            fig.legend(
                handles=legend_elements,
                loc='upper center', bbox_to_anchor=(0.5, 0.96),
                ncol=6, frameon=True, fancybox=False,
                edgecolor='black', facecolor='white', fontsize=8,
            )

            plt.tight_layout(rect=[0, 0, 1, 0.91])
            outfile = os.path.join(output_dir, f"{protein}_dashboard.png")
            fig.savefig(outfile, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved -> {outfile}")

    print("\nAll done.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',   default='./')
    parser.add_argument('--output_dir', default='./')
    parser.add_argument('--proteins',   nargs='+',
                        default=list(GPCR_TM_REGIONS.keys()))
    args = parser.parse_args()
    generate_dashboards(
        protein_list=args.proteins,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )
