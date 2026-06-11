# -*- coding: utf-8 -*-
"""
Compute column-sum normalized z-score from one R matrix stored in an npz file.

Workflow:
    R[L, L]
      -> column_sum[j] = sum_i R[i, j]
      -> normalized = column_sum / sum(column_sum)
      -> zscore(normalized)

CSV output: residue_index, zscore
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def find_square_matrix(npz_path: str | Path, key: str | None = None) -> tuple[str, np.ndarray]:
    """Find one square 2D matrix in an npz file."""
    with np.load(npz_path) as data:
        if key is not None:
            if key not in data.files:
                raise KeyError(f"Matrix key {key!r} not found. Available keys: {data.files}")
            matrix = data[key]
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError(f"Array {key!r} is not a square matrix: shape={matrix.shape}")
            return key, matrix.astype(float)

        candidates = [
            name for name in data.files
            if data[name].ndim == 2 and data[name].shape[0] == data[name].shape[1]
        ]
        if not candidates:
            raise KeyError(f"No square 2D matrix found. Available keys: {data.files}")
        return candidates[0], data[candidates[0]].astype(float)


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    std = x.std()
    if std == 0:
        return np.zeros_like(x)
    return (x - x.mean()) / std


def response_column_zscore(npz_path: str | Path, key: str | None = None) -> np.ndarray:
    """Load R from npz and return z-score of normalized column sums."""
    _, R = find_square_matrix(npz_path, key=key)
    column_sum = R.sum(axis=0)
    total = column_sum.sum()
    normalized = np.zeros_like(column_sum, dtype=float) if total == 0 else column_sum / total
    return zscore(normalized)


def save_zscore_csv(z: np.ndarray, output_csv: str | Path) -> None:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output_csv,
        np.column_stack([np.arange(1, len(z) + 1), z]),
        delimiter=",",
        header="residue_index,zscore",
        comments="",
        fmt=["%d", "%.10g"],
    )
