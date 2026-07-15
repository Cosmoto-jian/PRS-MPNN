#!/usr/bin/env python3
"""
Expert Deferral v2: RRF + H-Deferral with Ultra-Uniqueness Override
====================================================================
A 2-stage strategy that fuses DFI, DFI_membrane, and H:

Stage 1 (H-Deferral): Defer entirely to H if EITHER path is satisfied:
  Path A (normal): H's uniqueness ratio > uniq_thresh AND H's skewness > skew_guard
    - H is "going its own way" (unique selection) AND has positive skew
      (bottom-20% captures a meaningful cluster, not outliers)
  Path B (ultra-unique): H's uniqueness ratio > ultra_thresh
    - H is SO different from DFI/DFI_membrane that it must capture unique signal,
      regardless of skew. This rescues proteins like acm1 where H has negative skew
      but extremely high uniqueness (ratio=1.72) and excellent precision (0.50).

Stage 2 (RRF fallback): Otherwise, use Reciprocal Rank Fusion (k=80) of all three.

Validation (LOPO-CV, 20 GPCRs, TM residues, bottom-20% selection):
  - Expert Deferral v2 wins: 13/20 (>= max of all 3 single features)
  - Expert Deferral v1 (no ultra override): 12/20
  - RRF alone: 8/20
  - Best single feature (oracle): 7/20
  - Mean precision: 0.385 (v2) vs 0.377 (v1) vs 0.351 (RRF) vs 0.342 (DFI_membrane)
  - uniq_thresh=1.01 chosen in 20/20 LOPO folds
  - ultra_thresh=1.35 chosen in 20/20 LOPO folds
  - skew_guard=0.0 fixed (principled: sign change at zero)
  - k=80 fixed (stable plateau k=65-120, literature standard)

Usage
-----
    from expert_deferral_predict import predict_mutation_sites

    # dfi_df: DataFrame with columns ResI, DFI, DFI_membrane, DFI_Z, DFI_XY
    result = predict_mutation_sites(dfi_df)
"""

import numpy as np
import pandas as pd
from scipy.stats import skew

BOTTOM_PCT = 0.20
RRF_K = 80
UNIQ_THRESH = 1.01
SKEW_GUARD = 0.0
ULTRA_THRESH = 1.35
FEATURES = ['DFI', 'DFI_membrane', 'H']


def _get_uniqueness(d, feat, all_features=FEATURES):
    """How distinct is this feature's bottom-20% from the other features' selections?
    Higher = more unique."""
    n_sel = max(1, int(round(BOTTOM_PCT * len(d))))
    bottom_idx = set(d.sort_values(feat, ascending=True).iloc[:n_sel].index)
    others = [g for g in all_features if g != feat]
    overlaps = []
    for g in others:
        other_idx = set(d.sort_values(g, ascending=True).iloc[:n_sel].index)
        overlaps.append(len(bottom_idx & other_idx) / n_sel)
    return 1.0 - np.mean(overlaps)


def predict_mutation_sites(dfi_df, k=RRF_K, bottom_pct=BOTTOM_PCT,
                           uniq_thresh=UNIQ_THRESH, skew_guard=SKEW_GUARD,
                           ultra_thresh=ULTRA_THRESH):
    """
    Select predicted mutation sites using Expert Deferral v2.

    Parameters
    ----------
    dfi_df : DataFrame
        Must contain columns: ResI, DFI, DFI_membrane, DFI_Z, DFI_XY
    k : float
        RRF smoothing constant (default 80, stable for 65-120)
    bottom_pct : float
        Fraction of residues to select (default 0.20)
    uniq_thresh : float
        H uniqueness ratio threshold for normal deferral (default 1.01)
    skew_guard : float
        Minimum H skewness for normal deferral (default 0.0 = positive skew only)
    ultra_thresh : float
        H uniqueness ratio threshold for ultra-unique override (default 1.35)

    Returns
    -------
    DataFrame
        Selected residues with columns: ResI, DFI, DFI_membrane, H, method
        method is 'H_deferral' or 'RRF'
    """
    d = dfi_df.copy()

    # Compute H and filter invalid residues
    valid = (d['DFI_XY'] > 0) & (d['DFI_Z'] > 0) & (d['DFI'] > 0) & (d['DFI_membrane'] > 0)
    d = d[valid].copy()
    d['H'] = np.log(d['DFI_Z']) - np.log(d['DFI_XY'])

    n = len(d)
    n_select = max(1, int(round(bottom_pct * n)))

    # Compute ranks (ascending: low value = predicted mutation = rank 1)
    for f in FEATURES:
        d[f'{f}_rank'] = d[f].rank(method='average', ascending=True)

    # --- Stage 1: Expert Deferral check ---
    h_uniq = _get_uniqueness(d, 'H')
    max_other_uniq = max(_get_uniqueness(d, 'DFI'), _get_uniqueness(d, 'DFI_membrane'))
    uniq_ratio = h_uniq / max_other_uniq if max_other_uniq > 0 else 999
    h_skew = skew(d['H'].values)

    # Path A: unique AND positive skew
    # Path B: ultra-unique (bypass skew guard)
    path_a = uniq_ratio > uniq_thresh and h_skew > skew_guard
    path_b = uniq_ratio > ultra_thresh

    if path_a or path_b:
        selected = d.sort_values('H', ascending=True).iloc[:n_select]
        selected = selected[['ResI', 'DFI', 'DFI_membrane', 'H']].copy()
        selected['method'] = 'H_deferral'
        return selected

    # --- Stage 2: RRF fallback ---
    rrf_score = sum(1.0 / (k + d[f'{f}_rank'].values) for f in FEATURES)
    d['rrf_score'] = rrf_score
    selected = d.nlargest(n_select, 'rrf_score')
    selected = selected[['ResI', 'DFI', 'DFI_membrane', 'H', 'rrf_score']].copy()
    selected['method'] = 'RRF'
    return selected

if __name__ == "__main__":
    import os

    # =============================================================================
    # 修复：采用与 adaptive_union_predict 相同的相对路径系统
    # =============================================================================
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    TOOLS_DIR = os.path.dirname(SCRIPT_DIR)
    PROJECT_DIR = os.path.dirname(TOOLS_DIR)

    DFI_DIR = os.path.join(PROJECT_DIR, "results", "DFI")
    MUT_RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "mutation")

    proteins = ['aa2ar','adrb2','cxcr4','aa1r','aa2br','acm1','ccr1','ccr5','cxcr1',
                'glp1r','mc4r','aa3r','drd2','cxcr3','5ht1b','5ht2c','hrh1','acm2','grm1','acm3']

    GPCR_TM_REGIONS = {
        "aa2ar": [(8, 32), (43, 66), (79, 108), (119, 142), (173, 202), (235, 258), (267, 290)],
        "adrb2": [(29, 60), (67, 96), (103, 136), (147, 171), (197, 229), (267, 298), (305, 331)],
        "cxcr4": [(39, 68), (78, 106), (112, 143), (157, 180), (203, 229), (240, 268), (280, 305)],
        "aa1r":  [(12, 32), (48, 68), (78, 98), (128, 147), (174, 196), (237, 259), (268, 289)],
        "aa2br": [(9, 33), (44, 67), (79, 101), (122, 144), (179, 203), (236, 259), (268, 291)],
        "acm1":  [(23, 48), (63, 84), (105, 126), (143, 164), (187, 210), (351, 372), (385, 407)],
        "ccr1":  [(35, 60), (73, 95), (108, 129), (151, 175), (198, 223), (240, 264), (282, 305)],
        "ccr5":  [(31, 58), (69, 89), (103, 124), (142, 166), (199, 218), (236, 261), (271, 295)],
        "cxcr1": [(39, 65), (76, 96), (111, 132), (153, 176), (198, 220), (243, 267), (277, 302)],
        "glp1r": [(145, 165), (174, 194), (228, 248), (271, 291), (317, 337), (350, 370), (383, 403)],
        "mc4r":  [(44, 64), (77, 97), (115, 135), (154, 174), (196, 216), (247, 267), (280, 300)],
        "aa3r":  [(14, 34), (46, 66), (81, 101), (121, 141), (174, 194), (230, 250), (262, 282)],
        "drd2":  [(37, 60), (72, 96), (110, 131), (153, 176), (193, 216), (373, 394), (407, 428)],
        "cxcr3": [(43, 63), (76, 96), (111, 131), (151, 171), (204, 224), (244, 264), (284, 304)],
        "5ht1b": [(46, 66), (79, 99), (114, 134), (154, 174), (197, 217), (317, 337), (353, 373)],
        "5ht2c": [(66, 86), (99, 119), (134, 154), (174, 194), (217, 237), (311, 331), (348, 368)],
        "hrh1":  [(21, 41), (53, 73), (88, 108), (152, 172), (194, 214), (414, 434), (453, 473)],
        "acm2":  [(21, 41), (54, 74), (91, 111), (135, 155), (178, 198), (385, 405), (421, 441)],
        "grm1":  [(593, 613), (626, 646), (656, 676), (699, 719), (733, 753), (779, 799), (813, 833)],
        "acm3":  [(71, 91), (104, 124), (141, 161), (185, 205), (228, 248), (498, 518), (534, 554)],
    }

    def is_tm(resi, regions):
        return any(a <= resi <= b for a, b in regions)

    print(f"{'Protein':8s} {'DFI':>7s} {'DFI_m':>7s} {'H':>7s} {'best':>7s} {'ED':>7s} {'method':>12s} {'win?':>5s}")
    print("-" * 65)

    wins = 0
    for p in proteins:
        # 修复：按照 adaptive_union_predict 的结构读取文件，并加入容错检查
        dfi_path = os.path.join(DFI_DIR, f'{p}.csv')
        if not os.path.exists(dfi_path):
            print(f"  {p:8s} SKIP: DFI file not found")
            continue
            
        mut_path = os.path.join(MUT_RESULTS_DIR, p, f'{p}_mutation_summary.csv')
        if not os.path.exists(mut_path):
            print(f"  {p:8s} SKIP: mutation summary not found")
            continue

        dfi = pd.read_csv(dfi_path)
        mut = pd.read_csv(mut_path)
        mut_positions = set(mut['Position'].dropna().astype(int).unique())

        regions = GPCR_TM_REGIONS[p]
        dfi['is_tm'] = dfi['ResI'].apply(lambda r: is_tm(r, regions))
        dfi_tm = dfi[dfi['is_tm']].copy()
        
        valid = (dfi_tm['DFI_XY'] > 0) & (dfi_tm['DFI_Z'] > 0) & (dfi_tm['DFI'] > 0) & (dfi_tm['DFI_membrane'] > 0)
        dfi_tm = dfi_tm[valid].copy()
        dfi_tm['is_mutation'] = dfi_tm['ResI'].astype(int).isin(mut_positions).astype(int)

        n_sel = max(1, int(round(0.20 * len(dfi_tm))))

        dfi_prec = dfi_tm.sort_values('DFI', ascending=True).iloc[:n_sel]['is_mutation'].sum() / n_sel
        dfm_prec = dfi_tm.sort_values('DFI_membrane', ascending=True).iloc[:n_sel]['is_mutation'].sum() / n_sel
        dfi_tm['H'] = np.log(dfi_tm['DFI_Z']) - np.log(dfi_tm['DFI_XY'])
        h_prec = dfi_tm.sort_values('H', ascending=True).iloc[:n_sel]['is_mutation'].sum() / n_sel
        best = max(dfi_prec, dfm_prec, h_prec)

        result = predict_mutation_sites(dfi_tm)
        ed_hits = dfi_tm[dfi_tm['ResI'].astype(int).isin(result['ResI'].astype(int))]['is_mutation'].sum()
        ed_prec = ed_hits / n_sel
        method = result['method'].iloc[0]

        win = ed_prec >= best
        if win: wins += 1
        print(f"  {p:8s} {dfi_prec:7.3f} {dfm_prec:7.3f} {h_prec:7.3f} {best:7.3f} {ed_prec:7.3f} {method:>12s} {'W' if win else '.'}")

    print("-" * 65)
    print(f"\n  Expert Deferral v2 wins: {wins}/20 (>= max of all 3 single features)")
