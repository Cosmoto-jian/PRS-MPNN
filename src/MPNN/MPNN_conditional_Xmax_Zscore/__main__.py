# -*- coding: utf-8 -*-
"""CLI entry point: python -m MPNN_conditional_Xmax_Zscore"""

from __future__ import annotations

import argparse

from .pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MPNN X-scan conditional tensor -> column-max z-score pipeline.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory with per-protein subdirs containing *_xscan_probability_tensors.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="Numerical floor for log transforms (default: 1e-8)",
    )
    args = parser.parse_args()

    run(input_dir=args.input_dir, output_dir=args.output_dir, eps=args.eps)


if __name__ == "__main__":
    main()
