#!/usr/bin/env python3
"""
DFI (Dynamic Flexibility Index) & Effector Anisotropy Analysis
===============================================================

Calculates the dynamic flexibility index (DFI) and directional 
effector metrics to study the conformational dynamics of a protein.
It decomposes the perturbation response into Total, XY-planar, 
and Z-axial components, evaluating both Sensor (Row Sum) and 
Effector (Column Sum) characteristics, alongside the Z-Axis Effector Ratio.

Usage
-----
dfi_calc2.py --pdb PDBFILE [--hess HESSFILE] [--chain CHAINID] [--cutoff DISTANCE] [--help]
"""

import sys
import os
import numpy as np
import pandas as pd
from scipy import linalg as LA
from scipy import stats

# ------------------------------------------------------------
# Atom class
# ------------------------------------------------------------
class ATOM:
    def __init__(self, record, atom_index, atom_name, alt_loc, res_name, chainID,
                 res_index, insert_code, x, y, z, occupancy,
                 temp_factor, atom_type):
        self.record = str(record).strip()
        self.atom_index = int(atom_index)
        self.atom_name = str(atom_name).strip()
        self.alt_loc = str(alt_loc).strip()
        self.res_name = str(res_name).strip()
        self.chainID = str(chainID).strip()
        self.res_index = str(res_index).strip()
        self.insert_code = str(insert_code).strip()
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.occupancy = float(occupancy) if occupancy else 1.0
        self.temp_factor = float(temp_factor) if temp_factor and temp_factor.strip() else 0.0
        self.atom_type = str(atom_type).strip()

# ------------------------------------------------------------
# PDB reader (fixed column extraction)
# ------------------------------------------------------------
def pdb_reader(filename, CAonly=False, noalc=True, chainA=False,
               chain_name='A', Verbose=False):
    ATOMS = []
    readatoms = 0
    try:
        with open(filename, 'r') as pdb:
            for line in pdb:
                if line.startswith('ENDMDL'):
                    return ATOMS

                if line.startswith('ATOM') or line.startswith('HETATM'):
                    # Extract fixed-width columns (PDB format specification)
                    atom_name = line[12:16].strip()
                    if CAonly and atom_name != 'CA':
                        continue

                    # altLoc: column 17 (1-indexed -> index 16)
                    alt_loc = line[16] if len(line) > 16 else ' '
                    if noalc and alt_loc not in (' ', 'A'):
                        continue

                    # chainID: column 22 (index 21)
                    chainID = line[21] if len(line) > 21 else ' '
                    if chainA and chainID != chain_name:
                        continue

                    record = line[0:6].strip()
                    atom_index = line[6:11].strip()
                    res_name = line[17:20].strip()
                    res_index = line[22:27].strip()
                    # insertCode: column 27 (index 26)
                    insert_code = line[26] if len(line) > 26 else ' '
                    x = line[30:38].strip()
                    y = line[38:46].strip()
                    z = line[46:54].strip()
                    occupancy = line[54:60].strip()
                    temp_factor = line[60:66].strip()
                    atom_type = line[76:78].strip() if len(line) > 78 else ''

                    if not (x and y and z):
                        continue

                    ATOMS.append(ATOM(record, atom_index, atom_name, alt_loc,
                                      res_name, chainID, res_index, insert_code,
                                      x, y, z, occupancy, temp_factor, atom_type))
                    readatoms += 1
    except Exception as e:
        print(f"Error reading PDB file {filename}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Read {readatoms} atoms from {filename}")
    return ATOMS

# ------------------------------------------------------------
# Extract coordinates
# ------------------------------------------------------------
def getcoords(ATOMS):
    x = np.array([atom.x for atom in ATOMS if atom.atom_name == 'CA'], dtype=float)
    y = np.array([atom.y for atom in ATOMS if atom.atom_name == 'CA'], dtype=float)
    z = np.array([atom.z for atom in ATOMS if atom.atom_name == 'CA'], dtype=float)
    return x, y, z

# ------------------------------------------------------------
# Distance-dependent Hessian (ANM) with optional cutoff
# ------------------------------------------------------------
def calchessian(resnum, x, y, z, gamma=100, cutoff=None):
    """
    Build the Hessian matrix for the Anisotropic Network Model (ANM).
    If cutoff is given, interactions beyond that distance are ignored.
    """
    numresthree = 3 * resnum
    hess = np.zeros((numresthree, numresthree))

    for i in range(resnum):
        for j in range(resnum):
            if i == j:
                continue
            x_ij = x[i] - x[j]
            y_ij = y[i] - y[j]
            z_ij = z[i] - z[j]
            r2 = x_ij*x_ij + y_ij*y_ij + z_ij*z_ij
            r = np.sqrt(r2)
            if r2 < 1e-12:
                continue

            # Spring constant: gamma^3 / r^3 (ANM standard)
            sprngcnst = (gamma**3) / (r2 * r)
            if cutoff is not None and r > cutoff:
                sprngcnst = 0.0

            # Diagonal blocks (i,i)
            hess[3*i,   3*i]   += sprngcnst * (x_ij*x_ij / r2)
            hess[3*i+1, 3*i+1] += sprngcnst * (y_ij*y_ij / r2)
            hess[3*i+2, 3*i+2] += sprngcnst * (z_ij*z_ij / r2)
            hess[3*i,   3*i+1] += sprngcnst * (x_ij*y_ij / r2)
            hess[3*i,   3*i+2] += sprngcnst * (x_ij*z_ij / r2)
            hess[3*i+1, 3*i]   += sprngcnst * (y_ij*x_ij / r2)
            hess[3*i+1, 3*i+2] += sprngcnst * (y_ij*z_ij / r2)
            hess[3*i+2, 3*i]   += sprngcnst * (z_ij*x_ij / r2)
            hess[3*i+2, 3*i+1] += sprngcnst * (z_ij*y_ij / r2)

            # Off-diagonal blocks (i,j)
            hess[3*i,   3*j]   -= sprngcnst * (x_ij*x_ij / r2)
            hess[3*i+1, 3*j+1] -= sprngcnst * (y_ij*y_ij / r2)
            hess[3*i+2, 3*j+2] -= sprngcnst * (z_ij*z_ij / r2)
            hess[3*i,   3*j+1] -= sprngcnst * (x_ij*y_ij / r2)
            hess[3*i,   3*j+2] -= sprngcnst * (x_ij*z_ij / r2)
            hess[3*i+1, 3*j]   -= sprngcnst * (y_ij*x_ij / r2)
            hess[3*i+1, 3*j+2] -= sprngcnst * (y_ij*z_ij / r2)
            hess[3*i+2, 3*j]   -= sprngcnst * (z_ij*x_ij / r2)
            hess[3*i+2, 3*j+1] -= sprngcnst * (z_ij*y_ij / r2)
    return hess

def calc_pseudo_inverse(numres, x, y, z, cutoff=None):
    """Compute pseudo-inverse of Hessian using SVD, removing zero modes."""
    hess = calchessian(numres, x, y, z, cutoff=cutoff)
    U, w, Vt = LA.svd(hess, full_matrices=False)
    tol = 1e-6
    # Zero out singular values below tolerance (removes rigid body modes)
    invw = np.zeros_like(w)
    mask = w > tol
    invw[mask] = 1.0 / w[mask]
    return np.dot(np.dot(U, np.diag(invw)), Vt)

# ------------------------------------------------------------
# Perturbation matrix calculation (decomposed)
# ------------------------------------------------------------
def calcperturbMat(invHrs, directions, resnum, Normalize=True):
    """
    Apply unit forces in each given direction to each residue,
    compute resulting displacement magnitudes (total, XY, Z),
    and return the averaged perturbation matrices.
    """
    n_dirs = len(directions)
    perturbMat_total = np.zeros((resnum, resnum))
    perturbMat_xy    = np.zeros((resnum, resnum))
    perturbMat_z     = np.zeros((resnum, resnum))

    for k in range(n_dirs):
        perturbDir = directions[k, :]   # unit vector
        for j in range(resnum):
            delforce = np.zeros(3 * resnum)
            delforce[3*j : 3*j+3] = perturbDir
            delXperbVex = np.dot(invHrs, delforce)
            delXperbMat = delXperbVex.reshape((resnum, 3))

            delR_total = np.sqrt(np.sum(delXperbMat * delXperbMat, axis=1))
            delR_xy    = np.sqrt(np.sum(delXperbMat[:, :2] * delXperbMat[:, :2], axis=1))
            delR_z     = np.abs(delXperbMat[:, 2])

            perturbMat_total[:, j] += delR_total
            perturbMat_xy[:, j]    += delR_xy
            perturbMat_z[:, j]     += delR_z

    perturbMat_total /= n_dirs
    perturbMat_xy    /= n_dirs
    perturbMat_z     /= n_dirs

    if Normalize:
        # Normalize each matrix so that the sum of all entries = 1
        perturbMat_total /= np.sum(perturbMat_total)
        perturbMat_xy    /= np.sum(perturbMat_xy)
        perturbMat_z     /= np.sum(perturbMat_z)

    return perturbMat_total, perturbMat_xy, perturbMat_z

# ------------------------------------------------------------
# Metrics Analysis
# ------------------------------------------------------------
def pctrank(arr, inverse=False):
    """Percentile rank (0..1) of each element in the array."""
    n = float(len(arr))
    result = []
    for val in arr:
        cnt = np.sum(arr >= val) if inverse else np.sum(arr <= val)
        result.append(cnt / n)
    return np.array(result, dtype=float)

def dfianal(values):
    """Compute DFI, relative DFI, percentile, and z-score."""
    values = np.array(values, dtype=float)
    meanval = np.mean(values)
    dfirel = values / meanval if meanval != 0 else values
    dfizscore = stats.zscore(values)
    dfiperc = pctrank(values, inverse=False)
    return values, dfirel, dfiperc, dfizscore

# ------------------------------------------------------------
# Main calculation pipeline
# ------------------------------------------------------------
def calc_dfi(pdbfile, pdbid=None, covar=None, chain_name=None, cutoff=None,
             writetofile=False, dfianalfile=None):
    """Main function: compute anisotropy metrics and return DataFrame."""
    if not pdbid:
        pdbid = os.path.splitext(os.path.basename(pdbfile))[0]
    if not dfianalfile:
        dfianalfile = f"{pdbid}_anisotropy_analysis.csv"

    # Read CA atoms, optionally filter by chain
    ATOMS = pdb_reader(pdbfile, CAonly=True, noalc=True,
                       chainA=(chain_name is not None), chain_name=chain_name or 'A')

    if len(ATOMS) == 0:
        print(f"Error: No CA atoms found in {pdbfile}. Check chain ID or file format.", file=sys.stderr)
        sys.exit(1)

    x, y, z = getcoords(ATOMS)
    numres = len(ATOMS)

    # Compute pseudo-inverse of Hessian (or load from file)
    if covar is None:
        print("Building Hessian and computing pseudo-inverse...")
        invHrs = calc_pseudo_inverse(numres, x, y, z, cutoff=cutoff)
    else:
        print(f"Loading precomputed covariance from {covar}")
        invHrs = np.loadtxt(covar)

    # Define perturbation directions: all six cardinal directions (±x, ±y, ±z)
    directions = np.array([
        [ 1,  0,  0],
        [-1,  0,  0],
        [ 0,  1,  0],
        [ 0, -1,  0],
        [ 0,  0,  1],
        [ 0,  0, -1]
    ], dtype=float)
    # Directions are already unit vectors (norm = 1). Normalize just to be safe.
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    directions = directions / norms

    # Compute perturbation response matrices
    print("Applying perturbations...")
    mat_total, mat_xy, mat_z = calcperturbMat(invHrs, directions, numres, Normalize=True)

    metrics_dict = {}

    # 1. Standard DFI (row sum) and Effector (column sum) for Total, XY, Z
    for prefix, mat in zip(['Total', 'XY', 'Z'], [mat_total, mat_xy, mat_z]):
        dfi_arr = np.sum(mat, axis=1)   # Sensor: how much a residue moves
        eff_arr = np.sum(mat, axis=0)   # Effector: how much a residue affects others

        _, _, pctdfi, _ = dfianal(dfi_arr)
        _, _, pcteff, _ = dfianal(eff_arr)

        metrics_dict[f'DFI_{prefix}'] = dfi_arr
        metrics_dict[f'PctDFI_{prefix}'] = pctdfi
        metrics_dict[f'Effector_{prefix}'] = eff_arr
        metrics_dict[f'PctEffector_{prefix}'] = pcteff

    # 2. Z-axis Effector Ratio: S_Z / (S_XY + S_Z)
    eff_xy = metrics_dict['Effector_XY']
    eff_z  = metrics_dict['Effector_Z']
    # Add small epsilon to avoid division by zero
    ratio_z = eff_z / (eff_xy + eff_z + 1e-12)
    metrics_dict['Effector_Ratio_Z'] = ratio_z

    # Build output DataFrame
    mapres = {'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H',
              'ILE':'I','LYS':'K','LEU':'L','MET':'M','PRO':'P','ARG':'R','GLN':'Q',
              'ASN':'N','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}

    dfx = pd.DataFrame()
    dfx['ResI'] = [atom.res_index for atom in ATOMS]
    dfx['ChainID'] = [atom.chainID for atom in ATOMS]
    dfx['Res'] = [atom.res_name for atom in ATOMS]
    dfx['R'] = dfx['Res'].map(mapres)

    for key, vals in metrics_dict.items():
        dfx[key] = vals

    if writetofile:
        dfx.to_csv(dfianalfile, index=False)
        print(f"Analysis successfully saved to {dfianalfile}")

    return dfx

# ------------------------------------------------------------
# Command line parsing
# ------------------------------------------------------------
def parseCommandLine(argv):
    comline_arg = {}
    i = 1
    while i < len(argv):
        opt = argv[i]
        if opt in ['--pdb', '--hess', '--chain', '--cutoff']:
            if i+1 < len(argv):
                comline_arg[opt] = argv[i+1]
                i += 2
            else:
                i += 1
        elif opt == '--help':
            print(__doc__)
            sys.exit(0)
        else:
            i += 1

    if '--pdb' not in comline_arg:
        print(__doc__)
        print("Error: Missing required --pdb argument.", file=sys.stderr)
        sys.exit(1)

    cutoff = None
    if '--cutoff' in comline_arg:
        try:
            cutoff = float(comline_arg['--cutoff'])
        except ValueError:
            print("Warning: --cutoff must be a number. Ignoring.", file=sys.stderr)

    return (comline_arg['--pdb'],
            os.path.splitext(os.path.basename(comline_arg['--pdb']))[0],
            comline_arg.get('--hess', None),
            comline_arg.get('--chain', None),
            cutoff)

if __name__ == "__main__":
    pdbfile, pdbid, covar, chain_name, cutoff = parseCommandLine(sys.argv)

    if not os.path.isfile(pdbfile):
        print(f"Error: Target PDB file does not exist: {pdbfile}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing Anisotropy Pipeline for {pdbfile}")
    if chain_name:
        print(f"Using chain {chain_name}")
    if cutoff:
        print(f"Using distance cutoff = {cutoff} Å")

    try:
        df_dfi = calc_dfi(pdbfile, pdbid, covar=covar, chain_name=chain_name,
                          cutoff=cutoff, writetofile=True)
        print("Done.")
    except Exception as e:
        print(f"Fatal error processing {pdbfile}: {e}", file=sys.stderr)
        sys.exit(1)
