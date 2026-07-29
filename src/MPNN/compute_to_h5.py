# -*- coding: utf-8 -*-
"""
Batch compute R_xscan matrix and MPNN z-score from xscan probability tensor NPZ files,
then save all results into a single HDF5 file grouped by UniProt ID.

Usage:
    python src/MPNN/compute_to_h5.py
    python src/MPNN/compute_to_h5.py --input-dir raw/MPNN --output-h5 raw/MPNN/mpnn_results.h5

HDF5 structure:
    {uniprot_id}/
        R_xscan_LxL   (L×L float64) — response matrix R[i, j]
        mpnn_zscore    (L float64)   — column-normalized z-score per residue
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import h5py
import numpy as np

# Ensure the parent package is importable when run as a standalone script.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from MPNN_conditional_Xmax_Zscore.core import DEFAULT_EPS, compute_r_xscan_from_npz

# ---------------------------------------------------------------------------
# NPZ discovery
# ---------------------------------------------------------------------------

NPZ_GLOB = "*/*_xscan_probability_tensors.npz"
NPZ_SUFFIX = "_xscan_probability_tensors.npz"

# Regex to extract UniProt ID from filenames like:
#   AF-A0A0B4J1L0-F1-model_v6.pdb_xscan_probability_tensors.npz
_UNIPROT_RE = re.compile(r"^AF-(.+?)-F1-model_v6\.pdb$")


def find_protein_npz_files(input_dir: Path) -> list[Path]:
    """Find all *_xscan_probability_tensors.npz files one level deep."""
    return sorted(input_dir.glob(NPZ_GLOB))


def extract_uniprot_id(npz_path: Path) -> str:
    """Extract the UniProt accession ID from the NPZ filename.

    Example:
        AF-A0A0B4J1L0-F1-model_v6.pdb_xscan_probability_tensors.npz
        -> A0A0B4J1L0
    """
    name = npz_path.name
    if not name.endswith(NPZ_SUFFIX):
        raise ValueError(f"Unexpected NPZ filename: {name}")
    base = name[: -len(NPZ_SUFFIX)]  # e.g. AF-A0A0B4J1L0-F1-model_v6.pdb
    m = _UNIPROT_RE.match(base)
    if not m:
        raise ValueError(f"Cannot extract UniProt ID from: {base}")
    return m.group(1)


# ---------------------------------------------------------------------------
# Z-score computation (in-memory, matches zscore.py logic)
# ---------------------------------------------------------------------------

def compute_zscore_from_R(R: np.ndarray) -> np.ndarray:
    """Compute column-sum normalized z-score from an R[L, L] matrix.

    Steps (matching zscore.response_column_zscore):
        1. column_sum[j] = sum_i R[i, j]
        2. normalized = column_sum / sum(column_sum)
        3. zscore = (normalized - mean(normalized)) / std(normalized)
    """
    R = np.asarray(R, dtype=float)
    column_sum = R.sum(axis=0)
    total = column_sum.sum()
    if total == 0:
        normalized = np.zeros_like(column_sum)
    else:
        normalized = column_sum / total

    std = normalized.std()
    if std == 0:
        return np.zeros_like(normalized)
    return (normalized - normalized.mean()) / std


# ---------------------------------------------------------------------------
# HDF5 writer
# ---------------------------------------------------------------------------

def save_to_h5(
    output_h5: Path,
    uniprot_id: str,
    R: np.ndarray,
    zscore_array: np.ndarray,
) -> None:
    """Save R matrix and z-score into an HDF5 group named by UniProt ID."""
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_h5, "a") as f:
        grp = f.require_group(uniprot_id)
        if "R_xscan_LxL" in grp:
            del grp["R_xscan_LxL"]
        if "mpnn_zscore" in grp:
            del grp["mpnn_zscore"]
        grp.create_dataset("R_xscan_LxL", data=R, compression="gzip")
        grp.create_dataset("mpnn_zscore", data=zscore_array, compression="gzip")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    input_dir: str | Path = "raw/MPNN",
    output_h5: str | Path = "raw/MPNN/mpnn_results.h5",
    eps: float = DEFAULT_EPS,
) -> Path:
    """Batch process all NPZ files and write results to an HDF5 file.

    Args:
        input_dir: Directory with per-protein subdirs containing
                   *_xscan_probability_tensors.npz files.
        output_h5: Path to the output HDF5 file.
        eps: Numerical floor for log transforms.

    Returns:
        Path to the output HDF5 file.
    """
    input_dir = Path(input_dir)
    output_h5 = Path(output_h5)

    npz_files = find_protein_npz_files(input_dir)
    if not npz_files:
        raise FileNotFoundError(
            f"No *_xscan_probability_tensors.npz files found in {input_dir}"
        )

    print(f"Found {len(npz_files)} protein(s).")
    print(f"Output H5: {output_h5}\n")

    # Remove existing H5 to start fresh (we use "a" mode with del, but cleaner to recreate).
    if output_h5.exists():
        output_h5.unlink()

    for npz_path in npz_files:
        uniprot_id = extract_uniprot_id(npz_path)
        print(f"[{uniprot_id}] Loading {npz_path.name} ...")

        # Step 1: compute R_xscan_LxL via core.py
        R = compute_r_xscan_from_npz(npz_path, eps=eps)
        print(f"  R_xscan_LxL shape: {R.shape}")

        # Step 2: compute MPNN z-score from R
        z = compute_zscore_from_R(R)
        print(f"  mpnn_zscore shape: {z.shape}, "
              f"range: [{z.min():.4f}, {z.max():.4f}], "
              f"mean={z.mean():.6f}, std={z.std():.6f}")

        # Step 3: save to HDF5
        save_to_h5(output_h5, uniprot_id, R, z)
        print(f"  -> saved to H5 group '{uniprot_id}'\n")

    print(f"Done. Results written to: {output_h5}")
    return output_h5


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch compute R_xscan and MPNN z-score -> HDF5",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="raw/MPNN",
        help="Directory with per-protein subdirs containing *_xscan_probability_tensors.npz",
    )
    parser.add_argument(
        "--output-h5",
        type=str,
        default="raw/MPNN/mpnn_results.h5",
        help="Output HDF5 file path",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=DEFAULT_EPS,
        help=f"Numerical floor for log transforms (default: {DEFAULT_EPS})",
    )
    args = parser.parse_args()
    run(input_dir=args.input_dir, output_h5=args.output_h5, eps=args.eps)


if __name__ == "__main__":
    main()
