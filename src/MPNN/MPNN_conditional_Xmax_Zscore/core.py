# -*- coding: utf-8 -*-
"""
Core math: convert X-scan masked probability tensor into R_xscan response matrix.

Formula:
    delta[i] = abs(P_j_rm[i] - P_wt[i])
    delta[wt_i] = -inf
    aa_star = argmax(delta)
    R[i,j] = abs(-log(P_j_rm[i, aa_star]) - (-log(P_j_rm[i, wt_i])))
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_EPS = 1e-8


def clean_sequence(seq: str) -> str:
    return "".join(str(seq).strip().upper().split())


def wt_i(wt_seq: str, i: int, aa_order: str) -> int:
    """Return the amino-acid index of the WT residue at receiver position i."""
    seq = clean_sequence(wt_seq)
    aa = seq[i]
    if aa not in aa_order:
        raise ValueError(f"Unknown WT residue {aa!r} at position {i}; aa_order={aa_order!r}")
    return aa_order.index(aa)


def delta(masked_probs_i: np.ndarray, reference_probs_i: np.ndarray) -> np.ndarray:
    """Return abs(P_j_rm[i] - P_wt[i]) over the 20 amino acids."""
    return np.abs(np.asarray(masked_probs_i, dtype=float) - np.asarray(reference_probs_i, dtype=float))


def delta_without_wt(delta_values: np.ndarray, wt_index: int) -> np.ndarray:
    """Return a copy of delta after applying the original delta[wt_i] = -inf rule."""
    out = np.asarray(delta_values, dtype=float).copy()
    out[wt_index] = -np.inf
    return out


def aa_star(masked_delta_values: np.ndarray) -> int:
    """Return the non-WT amino-acid index with maximum probability perturbation."""
    return int(np.argmax(masked_delta_values))


def pseudo_energy(probability: float, eps: float = DEFAULT_EPS) -> float:
    """ProteinMPNN pseudo-energy E = -log(P)."""
    return float(-np.log(max(float(probability), eps)))


def validate_probability_rows(name: str, values: np.ndarray, atol: float = 1e-3) -> None:
    """Check non-negative, finite, row-normalized probability distributions."""
    arr = np.asarray(values, dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or Inf values")
    if (arr < -atol).any():
        raise ValueError(f"{name} contains negative probabilities")

    row_sums = arr.sum(axis=-1)
    bad = np.abs(row_sums - 1.0) > atol
    if bad.any():
        min_sum = float(np.min(row_sums))
        max_sum = float(np.max(row_sums))
        raise ValueError(f"{name} rows are not normalized: min_sum={min_sum:.6g}, max_sum={max_sum:.6g}")


def e_wt(masked_probs_i: np.ndarray, wt_index: int, eps: float = DEFAULT_EPS) -> float:
    """Pseudo-energy of the WT amino acid under the masked-j distribution."""
    return pseudo_energy(masked_probs_i[wt_index], eps=eps)


def e_star(masked_probs_i: np.ndarray, aa_star_index: int, eps: float = DEFAULT_EPS) -> float:
    """Pseudo-energy of aa_star under the masked-j distribution."""
    return pseudo_energy(masked_probs_i[aa_star_index], eps=eps)


def r_ij(
    masked_probs_i: np.ndarray,
    reference_probs_i: np.ndarray,
    wt_index: int,
    eps: float = DEFAULT_EPS,
) -> tuple[float, int]:
    """Compute one R[i, j] value and return (R_ij, aa_star_index)."""
    delta_values = delta(masked_probs_i, reference_probs_i)
    masked_delta_values = delta_without_wt(delta_values, wt_index)
    aa_star_index = aa_star(masked_delta_values)
    value = abs(e_star(masked_probs_i, aa_star_index, eps=eps) - e_wt(masked_probs_i, wt_index, eps=eps))
    return float(value), aa_star_index


def d_tensor_to_r_xscan(
    d_tensor: np.ndarray,
    p_ref: np.ndarray,
    wt_seq: str,
    aa_order: str,
    eps: float = DEFAULT_EPS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert d_tensor_LxLx20 to R_xscan_LxL.

    Returns:
        R_xscan: float matrix with shape L x L, convention R[receiver, source]
        aa_star_matrix: int matrix with selected amino-acid index for each pair
    """
    d_tensor = np.asarray(d_tensor, dtype=float)
    p_ref = np.asarray(p_ref, dtype=float)
    wt_seq = clean_sequence(wt_seq)

    if d_tensor.ndim != 3:
        raise ValueError(f"d_tensor must be 3D, got shape {d_tensor.shape}")
    if p_ref.ndim != 2:
        raise ValueError(f"p_ref must be 2D, got shape {p_ref.shape}")

    L, source_L, M = d_tensor.shape
    if source_L != L:
        raise ValueError(f"d_tensor must have shape L x L x 20, got {d_tensor.shape}")
    if M != len(aa_order):
        raise ValueError(f"d_tensor amino-acid axis {M} does not match aa_order length {len(aa_order)}")
    if p_ref.shape != (L, M):
        raise ValueError(f"p_ref shape {p_ref.shape} does not match d_tensor receiver axes {(L, M)}")
    if len(wt_seq) != L:
        raise ValueError(f"wt_seq length {len(wt_seq)} does not match tensor length {L}")

    validate_probability_rows("d_tensor", d_tensor)
    validate_probability_rows("p_ref", p_ref)

    R = np.zeros((L, L), dtype=np.float64)
    aa_star_matrix = np.full((L, L), -1, dtype=np.int32)

    wt_indices = [wt_i(wt_seq, i, aa_order=aa_order) for i in range(L)]

    for j in range(L):
        for i in range(L):
            if i == j:
                continue
            value, aa_star_index = r_ij(
                masked_probs_i=d_tensor[i, j, :],
                reference_probs_i=p_ref[i, :],
                wt_index=wt_indices[i],
                eps=eps,
            )
            R[i, j] = value
            aa_star_matrix[i, j] = aa_star_index

    np.fill_diagonal(R, 0.0)
    np.fill_diagonal(aa_star_matrix, -1)
    return R, aa_star_matrix


def load_npz_inputs(npz_path: str | Path) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Load P_wt, d_tensor, wt_sequence, and aa_order from the notebook npz output."""
    with np.load(npz_path) as data:
        p_wt = data["P_wt_Lx20"]
        d_tensor = data["d_tensor_LxLx20"]
        wt_sequence = str(data["wt_sequence"].item())
        aa_order = str(data["aa_order"].item())
    return p_wt, d_tensor, wt_sequence, aa_order


def compute_r_xscan_from_npz(npz_path: str | Path, eps: float = DEFAULT_EPS) -> np.ndarray:
    """Load one protein npz and compute R_xscan_LxL."""
    p_wt, d_tensor, wt_sequence, aa_order = load_npz_inputs(npz_path)
    R, _ = d_tensor_to_r_xscan(
        d_tensor=d_tensor,
        p_ref=p_wt,
        wt_seq=wt_sequence,
        aa_order=aa_order,
        eps=eps,
    )
    return R
