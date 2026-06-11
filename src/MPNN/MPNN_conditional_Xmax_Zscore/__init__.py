# -*- coding: utf-8 -*-
"""
MPNN_conditional_Xmax_Zscore
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MPNN X-scan conditional probability tensor -> column-max z-score pipeline.

Usage:
    # CLI
    python -m MPNN_conditional_Xmax_Zscore --input-dir ./selectPDB --output-dir ./results

    # Python
    from MPNN_conditional_Xmax_Zscore import run
    run(input_dir="./selectPDB", output_dir="./results")
"""

from .pipeline import run

__all__ = ["run"]
