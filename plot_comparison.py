#!/usr/bin/env python3
"""
跨条件对比热图绘制程序
===================
每个蛋白质生成一张图：
【热图】行 = 突变位点，列 = 条件×阈值，格子 = 是否命中 (红底 ✓ / 灰底 -)
百分位阈值只在跨膜区（TM region）内计算。
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ─── 路径配置（已修改为您指定的单文件路径） ──────────────────────────────────
MUTATION_FILE = '/Users/zizhenghe/Documents/DFI/mutation/mutations.csv'

# ─── 蛋白质跨膜区定义 ──────────────────────────────────────────────────────
PROTEIN_DICT = {
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
    "mc4r": [(44, 64), (77, 97), (115, 135), (154, 174), (196, 216), (247, 267), (280, 300)]
}

# ─── 4 种条件配置 ──────────────────────────────────────────────────────────
CONDITIONS = [
    {'suffix': '_EXP',   'label': 'EXP',   'short': 'EXP',   'signal_col': 'ratio'},
    {'suffix': '_CTRL1', 'label': 'CTRL1', 'short': 'CTRL1', 'signal_col': 'ratio'},
    {'suffix': '_CTRL2', 'label': 'CTRL2', 'short': 'CTRL2', 'signal_col': 'dfi'},
    {'suffix': '_CTRL3', 'label': 'CTRL3', 'short': 'CTRL3', 'signal_col': 'dfi'},
]

THRESHOLDS = [0.02, 0.1]
THRESH_LABELS = ['Top 2%', 'Top 10%']

# ─── 辅助函数 ────────────────────────────────────────────────────────────────

def minmax_norm(arr):
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    if hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)

def parse_mutation_sites(mutation_csv, protein_name):
    """从总表中解析某个蛋白质的突变位点"""
    df = pd.read_csv(mutation_csv, header=0)
    # 假设第一列是蛋白质名称 (例如 'pertain')
    protein_col = df.columns[0]
    row = df[df[protein_col].astype(str).str.lower() == protein_name.lower()]
    if row.empty:
        raise ValueError(f"Protein '{protein_name}' not found in {mutation_csv}")
    
    sites = {}
    for col in df.columns[1:]:
        col_lower = col.lower()
        if 'green' in col_lower:
            key = 'green'
        elif 'red' in col_lower:
            key = 'red'
        else:
            continue
            
        residue_str = row.iloc[0][col]
        if pd.isna(residue_str) or str(residue_str).strip() == '':
            continue
            
        # 使用正则表达式提取所有数字，完美兼容空格分隔和连字符分隔
        nums = re.findall(r'\d+', str(residue_str))
        residues = [int(n) for n in nums]
        
        if residues:
            sites.setdefault(key, []).extend(residues)
            
    for key in sites:
        sites[key] = sorted(set(sites[key]))
    return sites

def get_signal(df, mask, cond):
    if cond['signal_col'] == 'ratio':
        raw = 1.0 - df['Effector_Ratio_Z'].values.astype(float)
    else:
        raw = df['DFI_Total'].values.astype(float)
    
    # 你的新逻辑：只把跨膜区的原始数值抽出来，在它们内部进行 0-1 归一化
    raw_tm = raw[mask]
    norm_tm_only = minmax_norm(raw_tm)
    
    resi_all = df['ResI'].values.astype(int)
    resi_tm = resi_all[mask]
    
    # 为了绘图的连续性，全局线条保持原始归一化（或不画全局折线），
    # 但返回用于计算命中的 signal_tm 必须是纯局部的
    norm_full = minmax_norm(raw) 
    
    return resi_all, norm_full, resi_tm, norm_tm_only

def tm_mask(resi_all, tm_regions):
    mask = np.ones(len(resi_all), dtype=bool)
    for s, e in tm_regions:
        mask |= (resi_all >= s) & (resi_all <= e)
    return mask

def hit_residues(resi_tm, signal_tm, threshold):
    cutoff = np.nanpercentile(signal_tm, threshold * 100)
    return set(resi_tm[signal_tm <= cutoff])

# ─── 主绘图函数 ──────────────────────────────────────────────────────────────

def generate_comparison(protein_list=None, data_dir='./', output_dir='./'):
    if protein_list is None:
        protein_list = ['aa2ar', 'cxcr4', 'adrb2']
    os.makedirs(output_dir, exist_ok=True)

    for protein in protein_list:
        print(f"\nProcessing {protein.upper()} ...")

        # 1. 读取实验数据
        df_dict = {}
        for cond in CONDITIONS:
            fname = os.path.join(data_dir, f"{protein}{cond['suffix']}.csv")
            if os.path.exists(fname):
                df_dict[cond['suffix']] = pd.read_csv(fname)
                print(f"  Loaded: {fname}")
            else:
                print(f"  Warning: {fname} not found, skipping.")

        if not df_dict:
            print(f"  No data for {protein}, skipping.")
            continue

        # 2. 读取突变位点
        mut_sites = {}
        if os.path.exists(MUTATION_FILE):
            try:
                mut_sites = parse_mutation_sites(MUTATION_FILE, protein)
                print(f"  Mutations: {mut_sites}")
            except Exception as e:
                print(f"  Mutation parsing error: {e}")
        else:
            print(f"  Mutation file not found: {MUTATION_FILE}")

        all_mut = sorted(set(
            r for lst in mut_sites.values() for r in lst
        ))

        # 3. 计算跨膜区掩码
        tm_regions = PROTEIN_DICT.get(protein.upper(), [])
        first_ref_df = next(iter(df_dict.values()))
        resi_all_ref = first_ref_df['ResI'].values.astype(int)
        mask_tm = tm_mask(resi_all_ref, tm_regions)

        # 4. 提取信号并计算命中矩阵
        hit_matrix = {}
        valid_conds = []

        for cond in CONDITIONS:
            suffix = cond['suffix']
            if suffix not in df_dict:
                continue
            df = df_dict[suffix]
            resi_all = df['ResI'].values.astype(int)
            mask = tm_mask(resi_all, tm_regions)
            _, _, resi_tm, signal_tm = get_signal(df, mask, cond)
            
            for ti, thr in enumerate(THRESHOLDS):
                hit_matrix[(suffix, ti)] = hit_residues(resi_tm, signal_tm, thr)
            valid_conds.append(cond)

        if not valid_conds:
            print("  No valid conditions, skipping.")
            continue

        # ── 仅绘制热图 ───────────────────────────────────────────────────────────
        # 根据突变位点的多少动态调整图像高度，确保视觉效果更好
        fig_height = max(6, len(all_mut) * 0.3)
        fig, ax_heat = plt.subplots(figsize=(10, fig_height))
        fig.suptitle(f"{protein.upper()} – Mutation Hit Analysis", fontsize=14, fontweight='bold', y=0.98)

        if all_mut:
            n_thresh = len(THRESHOLDS)
            n_conds = len(valid_conds)

            # 列标签：条件 × 阈值
            col_labels = [f"{cond['short']}\n{tlbl}" for tlbl in THRESH_LABELS for cond in valid_conds]
            # 行标签：突变位点
            row_labels = [str(r) for r in all_mut]

            heat_data = np.zeros((len(all_mut), len(col_labels)), dtype=float)
            col_idx = 0
            for ti in range(n_thresh):
                for cond in valid_conds:
                    suffix = cond['suffix']
                    hits = hit_matrix[(suffix, ti)]
                    for ri, r in enumerate(all_mut):
                        heat_data[ri, col_idx] = 1.0 if r in hits else 0.0
                    col_idx += 1

            cmap_hit = mcolors.ListedColormap(['#F5F5F5', '#C0392B'])
            im = ax_heat.imshow(heat_data, cmap=cmap_hit, vmin=0, vmax=1, aspect='auto', interpolation='nearest')

            # 设置刻度及标签
            ax_heat.set_xticks(range(len(col_labels)))
            ax_heat.set_xticklabels(col_labels, fontsize=10, rotation=30, ha='right')
            ax_heat.set_yticks(range(len(row_labels)))
            ax_heat.set_yticklabels(row_labels, fontsize=10)
            ax_heat.set_xlabel('Condition × Threshold', fontsize=11, labelpad=10)
            ax_heat.set_ylabel('Mutation Residue Index', fontsize=11, labelpad=10)
            ax_heat.set_title('Hit map (Red = Hit, Gray = Miss)', fontsize=12, pad=15)

            # 竖分隔线区分 2% 组和 10% 组
            ax_heat.axvline(n_conds - 0.5, color='black', linewidth=1.5)

            # 填充格子内的文字（命中为✓，未命中为-）
            for ri in range(len(all_mut)):
                for ci in range(len(col_labels)):
                    val = int(heat_data[ri, ci])
                    txt = '✓' if val else '–'
                    fc  = 'white' if val else '#999999'
                    ax_heat.text(ci, ri, txt, ha='center', va='center', fontsize=10, color=fc, fontweight='bold')
        else:
            ax_heat.text(0.5, 0.5, 'No mutation data found', ha='center', va='center', transform=ax_heat.transAxes, fontsize=14)

        # ── 保存图像 ─────────────────────────────────────────────────────────────
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        # 文件名后缀改为 _heatmap.png 以体现它只包含热图
        outfile = os.path.join(output_dir, f"{protein}_heatmap.png")
        fig.savefig(outfile, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved → {outfile}")

    print("\nAll done.")

# ─── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Cross-condition Heatmap Generator')
    parser.add_argument('--data_dir',   default='./', help='Directory containing *_EXP.csv, *_CTRL*.csv files')
    parser.add_argument('--output_dir', default='./', help='Directory to save output PNG files')
    parser.add_argument('--proteins',   nargs='+', default=['aa2ar', 'cxcr4', 'adrb2'], help='List of protein names (lowercase)')
    args = parser.parse_args()
    
    generate_comparison(
        protein_list=args.proteins,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )
