#!/usr/bin/env python3
"""
Batch DFI Calculation Script
=============================
Iterates over all .pdb.gz protein files in raw/pdb_temp/, computes DFI_Total
for each using dfi_calc6.py, and saves results into a single HDF5 file
organized by protein ID groups.

Output: results/h5/dfi_results.h5
  - Group per protein ID (e.g., "A0A0B4J1L0")
  - Each group contains dataset "DFI_Total" (1D float array, per-residue)
"""

import sys
import os
import re
import gzip
import glob
import tempfile
import traceback

import numpy as np
import h5py

# ---------------------------------------------------------------------------
# Path setup: add repo root to sys.path so dfi_calc6 can be imported
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)  # t_scan/ -> repo root
sys.path.insert(0, REPO_ROOT)

import dfi_calc6

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_DIR = os.path.join(REPO_ROOT, "raw", "pdb_temp")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results", "h5")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "dfi_results.h5")

# Regex to extract UniProt protein ID from AlphaFold filename
# Pattern: AF-{UNIPROT_ID}-F1-model_v6.pdb.gz
PROTEIN_ID_PATTERN = re.compile(r'^AF-([A-Za-z0-9]+)-F1-model_v6\.pdb\.gz$')


def extract_protein_id(filename: str) -> str | None:
    """Extract UniProt protein ID from AlphaFold PDB filename.

    Example:
        'AF-A0A0B4J1L0-F1-model_v6.pdb.gz' -> 'A0A0B4J1L0'

    Returns None if the filename doesn't match the expected pattern.
    """
    basename = os.path.basename(filename)
    match = PROTEIN_ID_PATTERN.match(basename)
    if match:
        return match.group(1)
    return None


def process_protein(pdb_gz_path: str, protein_id: str) -> np.ndarray | None:
    """Decompress a .pdb.gz file, compute DFI_Total, and return the per-residue array.

    Args:
        pdb_gz_path: Path to the gzipped PDB file.
        protein_id: Extracted protein identifier (used as pdbid).

    Returns:
        1D numpy array of DFI_Total values (one per residue), or None on failure.
    """
    tmp_pdb = None
    try:
        # Decompress .pdb.gz to a temporary .pdb file
        tmp_pdb = tempfile.NamedTemporaryFile(
            mode='w', suffix='.pdb', delete=False, dir=tempfile.gettempdir()
        )
        tmp_path = tmp_pdb.name

        with gzip.open(pdb_gz_path, 'rt') as f_in:
            tmp_pdb.write(f_in.read())
        tmp_pdb.close()  # Close so dfi_calc6 can open it

        print(f"  [{protein_id}] Computing DFI (no membrane, total)...")

        # Call dfi_calc6's core function directly
        # no_membrane=True -> total DFI without membrane constraints
        dfx = dfi_calc6.calc_dfi_single(
            pdbfile=tmp_path,
            pdbid=protein_id,
            no_membrane=True,
            suffix='_total',
        )

        if dfx is None or dfx.empty:
            print(f"  [{protein_id}] WARNING: calc_dfi_single returned empty result.")
            return None

        dfi_total = dfx['DFI_Total'].values.astype(np.float64)
        print(f"  [{protein_id}] Done. {len(dfi_total)} residues, "
              f"DFI_Total range: [{dfi_total.min():.6f}, {dfi_total.max():.6f}]")

        # Clean up the CSV file that calc_dfi_single writes to CWD
        csv_file = f"{protein_id}_total_anisotropy_analysis.csv"
        if os.path.isfile(csv_file):
            os.remove(csv_file)

        return dfi_total

    except BaseException:
        # BaseException catches SystemExit too (calc_dfi_single calls sys.exit(1)
        # when no CA atoms are found, which raises SystemExit).
        print(f"  [{protein_id}] ERROR during DFI calculation:")
        traceback.print_exc()
        return None

    finally:
        # Clean up temporary PDB file
        if tmp_pdb is not None:
            try:
                os.unlink(tmp_pdb.name)
            except OSError:
                pass


def main():
    print("=" * 60)
    print("Batch DFI Calculation")
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
    # Discover .pdb.gz files
    # -----------------------------------------------------------------------
    pdb_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.pdb.gz")))
    if not pdb_files:
        print(f"ERROR: No .pdb.gz files found in {INPUT_DIR}")
        sys.exit(1)

    print(f"\nFound {len(pdb_files)} protein file(s):")
    for f in pdb_files:
        print(f"  - {os.path.basename(f)}")

    # -----------------------------------------------------------------------
    # Process each protein
    # -----------------------------------------------------------------------
    results = {}  # protein_id -> DFI_Total numpy array
    skipped = []
    failed = []

    for pdb_path in pdb_files:
        basename = os.path.basename(pdb_path)
        protein_id = extract_protein_id(basename)

        if protein_id is None:
            print(f"\n[{basename}] SKIPPED: filename does not match expected pattern.")
            skipped.append(basename)
            continue

        print(f"\n[{protein_id}] Processing {basename}...")
        dfi_array = process_protein(pdb_path, protein_id)

        if dfi_array is not None:
            results[protein_id] = dfi_array
        else:
            failed.append(protein_id)

    # -----------------------------------------------------------------------
    # Write HDF5 output
    # -----------------------------------------------------------------------
    if not results:
        print("\nERROR: No proteins were successfully processed. No H5 file created.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Writing results to {OUTPUT_FILE}...")

    with h5py.File(OUTPUT_FILE, 'w') as h5f:
        for protein_id, dfi_array in results.items():
            grp = h5f.create_group(protein_id)
            grp.create_dataset('DFI_Total', data=dfi_array)
            print(f"  Group '{protein_id}': DFI_Total shape={dfi_array.shape}, "
                  f"dtype={dfi_array.dtype}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"  Total files found:    {len(pdb_files)}")
    print(f"  Successfully processed: {len(results)}")
    print(f"  Skipped (bad name):   {len(skipped)}")
    print(f"  Failed (calc error):  {len(failed)}")
    if skipped:
        print(f"  Skipped files: {skipped}")
    if failed:
        print(f"  Failed proteins: {failed}")
    print(f"\nOutput: {OUTPUT_FILE}")
    print("Done.")


if __name__ == "__main__":
    main()
