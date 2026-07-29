#!/usr/bin/env python3
"""
Batch MPNN X-scan Z-score Calculation Script
=============================================
Iterates over per-protein, per-temperature NPZ files in raw/0729Tmpnn/,
converts d_tensor → R_xscan matrix via core.py, then computes per-residue
z-score (column-sum normalized) via zscore.py, and saves all results into
a single HDF5 file.

Output: results/h5/Tmpnn_zscore.h5
  Hierarchy: /{protein_id}/{temperature}/mpnn_zscore/zscore
"""

import sys
import os
import glob
import traceback

import numpy as np
import h5py

# ---------------------------------------------------------------------------
# Path setup: add the MPNN_conditional_Xmax_Zscore package to sys.path
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

ZSCORE_PKG = os.path.join(REPO_ROOT, "src", "MPNN", "MPNN_conditional_Xmax_Zscore")
sys.path.insert(0, os.path.dirname(ZSCORE_PKG))  # parent so "MPNN_conditional_Xmax_Zscore" is importable

from MPNN_conditional_Xmax_Zscore.core import compute_r_xscan_from_npz

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_DIR = os.path.join(REPO_ROOT, "raw", "0729Tmpnn")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results", "h5")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "Tmpnn_zscore.h5")


def zscore_normalize(x: np.ndarray) -> np.ndarray:
    """Compute z-score: (x - mean) / std. Returns zeros if std == 0."""
    x = np.asarray(x, dtype=float)
    std = x.std()
    if std == 0:
        return np.zeros_like(x)
    return (x - x.mean()) / std


def compute_mpnn_zscore(npz_path: str) -> np.ndarray:
    """Load NPZ, compute R_xscan matrix, then return per-residue z-score.

    Workflow (matching zscore.py):
        1. d_tensor -> R_xscan_LxL  (core.py)
        2. column_sum[j] = sum_i R[i,j]
        3. normalized = column_sum / sum(column_sum)
        4. zscore(normalized)
    """
    # Step 1: convert d_tensor to R_xscan
    R = compute_r_xscan_from_npz(npz_path)

    # Step 2-3: column-sum normalize
    column_sum = R.sum(axis=0)
    total = column_sum.sum()
    if total == 0:
        normalized = np.zeros_like(column_sum, dtype=float)
    else:
        normalized = column_sum / total

    # Step 4: z-score
    return zscore_normalize(normalized)


def main():
    print("=" * 60)
    print("Batch MPNN X-scan Z-score Calculation")
    print(f"Input directory:  {INPUT_DIR}")
    print(f"Output H5 file:   {OUTPUT_FILE}")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Validate input directory
    # -----------------------------------------------------------------------
    if not os.path.isdir(INPUT_DIR):
        print(f"ERROR: Input directory not found: {INPUT_DIR}")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Discover protein folders
    # -----------------------------------------------------------------------
    protein_dirs = sorted(
        d for d in glob.glob(os.path.join(INPUT_DIR, "*"))
        if os.path.isdir(d)
    )
    if not protein_dirs:
        print(f"ERROR: No protein subdirectories found in {INPUT_DIR}")
        sys.exit(1)

    protein_ids = [os.path.basename(d) for d in protein_dirs]
    print(f"\nFound {len(protein_ids)} protein(s): {', '.join(protein_ids)}")

    # -----------------------------------------------------------------------
    # Process each protein × temperature
    # -----------------------------------------------------------------------
    results = {}   # results[protein_id][temperature] = zscore_array
    total_tasks = 0
    success_count = 0
    failed_tasks = []

    for protein_dir in protein_dirs:
        protein_id = os.path.basename(protein_dir)
        results[protein_id] = {}

        # Find all temperature subdirectories
        temp_dirs = sorted(
            d for d in glob.glob(os.path.join(protein_dir, "T_*"))
            if os.path.isdir(d)
        )
        for temp_dir in temp_dirs:
            temp_label = os.path.basename(temp_dir)  # e.g., "T_0.05"
            total_tasks += 1

            # Find the NPZ file in this temperature folder
            npz_files = glob.glob(os.path.join(temp_dir, "*_xscan_probability_tensors.npz"))
            if not npz_files:
                print(f"\n[{protein_id}/{temp_label}] SKIPPED: no NPZ file found.")
                failed_tasks.append(f"{protein_id}/{temp_label}")
                continue

            npz_path = npz_files[0]
            print(f"\n[{protein_id}/{temp_label}] Processing {os.path.basename(npz_path)}...")

            try:
                z = compute_mpnn_zscore(npz_path)
                results[protein_id][temp_label] = z
                success_count += 1
                print(f"  -> z-score computed: {len(z)} residues, "
                      f"range=[{z.min():.4f}, {z.max():.4f}], "
                      f"mean={z.mean():.6f}, std={z.std():.6f}")
            except BaseException:
                print(f"  -> ERROR:")
                traceback.print_exc()
                failed_tasks.append(f"{protein_id}/{temp_label}")

    # -----------------------------------------------------------------------
    # Write HDF5 output
    # -----------------------------------------------------------------------
    if success_count == 0:
        print("\nERROR: No tasks completed successfully. No H5 file created.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Writing results to {OUTPUT_FILE}...")

    with h5py.File(OUTPUT_FILE, 'w') as h5f:
        for protein_id, temps in results.items():
            if not temps:
                continue
            prot_grp = h5f.create_group(protein_id)
            for temp_label, z_array in temps.items():
                temp_grp = prot_grp.create_group(temp_label)
                mpnn_grp = temp_grp.create_group("mpnn_zscore")
                mpnn_grp.create_dataset("zscore", data=z_array)
                print(f"  /{protein_id}/{temp_label}/mpnn_zscore/zscore  "
                      f"shape={z_array.shape}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"  Proteins:             {len(protein_ids)}")
    print(f"  Total tasks:          {total_tasks}")
    print(f"  Successful:           {success_count}")
    print(f"  Failed:               {len(failed_tasks)}")
    if failed_tasks:
        print(f"  Failed tasks: {failed_tasks}")
    print(f"\nOutput: {OUTPUT_FILE}")
    print("Done.")


if __name__ == "__main__":
    main()
