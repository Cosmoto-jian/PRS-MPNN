# -*- coding: utf-8 -*-
"""
Pipeline: batch process selectPDB xscan probability tensor npz files.

For each protein:
    1. d_tensor -> MPNN R_xscan
    2. MPNN R_xscan -> MPNN z-score
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .core import DEFAULT_EPS, compute_r_xscan_from_npz
from .zscore import response_column_zscore, save_zscore_csv


def find_protein_npz_files(input_dir: Path) -> list[Path]:
    """Find all *_xscan_probability_tensors.npz files one level deep."""
    return sorted(input_dir.glob("*/*_xscan_probability_tensors.npz"))


def protein_id_from_npz(npz_path: Path) -> str:
    """Extract protein ID by stripping the known suffix."""
    suffix = "_xscan_probability_tensors.npz"
    if not npz_path.name.endswith(suffix):
        raise ValueError(f"Unexpected input npz name: {npz_path.name}")
    return npz_path.name[: -len(suffix)]


def save_matrix_npz(path: Path, **arrays: np.ndarray) -> None:
    """Save arrays to a compressed npz file, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def process_one_protein(
    input_npz: Path,
    output_root: Path,
    eps: float = DEFAULT_EPS,
) -> Path:
    """Run the full pipeline for one protein: tensor -> R_xscan -> z-score CSV.

    Returns the path to the final z-score CSV.
    """
    protein_id = protein_id_from_npz(input_npz)
    protein_dir = output_root / protein_id
    process_dir = protein_dir / "process"
    csv_dir = protein_dir / "csv"
    process_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{protein_id}]")

    # Step 1: d_tensor -> R_xscan
    mpnn_r_npz = process_dir / f"{protein_id}_MPNN_R_xscan_LxL.npz"
    mpnn_R = compute_r_xscan_from_npz(input_npz, eps=eps)
    save_matrix_npz(mpnn_r_npz, R_xscan_LxL=mpnn_R, eps=np.asarray(eps))

    # Step 2: R_xscan -> z-score
    mpnn_z = response_column_zscore(mpnn_r_npz, key="R_xscan_LxL")
    final_csv = csv_dir / f"{protein_id}_zscore.csv"
    save_zscore_csv(mpnn_z, final_csv)

    print(f"  -> {final_csv}")
    return final_csv


def run(
    input_dir: str | Path,
    output_dir: str | Path,
    eps: float = DEFAULT_EPS,
) -> list[Path]:
    """Batch run the pipeline over all proteins in input_dir.

    Args:
        input_dir: Directory containing per-protein subdirs with
                   *_xscan_probability_tensors.npz files.
        output_dir: Where per-protein output is written.
        eps: Numerical floor for log transforms.

    Returns:
        List of paths to the final z-score CSV files.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    input_files = find_protein_npz_files(input_dir)
    if not input_files:
        raise FileNotFoundError(f"No *_xscan_probability_tensors.npz files found in {input_dir}")

    print(f"Found {len(input_files)} proteins.")
    results = []
    for input_npz in input_files:
        csv_path = process_one_protein(input_npz, output_root=output_dir, eps=eps)
        results.append(csv_path)

    print("\nDone.")
    return results
