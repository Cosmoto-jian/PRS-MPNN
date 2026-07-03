#!/usr/bin/env python3
"""
DFI (Dynamic Flexibility Index) & Directional Effector Anisotropy
=================================================================

Anisotropic-Network-Model (ANM / imANM) DFI with a **globally** applied,
membrane-oriented anisotropy -- there is NO membrane-interior vs. exterior
distinction anymore. Every residue pair gets the same anisotropic 3x3
super-element.

Hessian super-element (off-diagonal block, pair i-j)
----------------------------------------------------
Let d = R_i - R_j = (x, y, z), r = |d|, and w = (w_x, w_y, w_z) the
per-axis stiffness weights. Then

    H_ij = -(gamma / r^3) * (w ⊗ w) ∘ (d ⊗ d)

           = -(gamma / r^3) [[ wx^2 x^2 , wx wy x y , wx wz x z ],
                             [ wx wy x y , wy^2 y^2 , wy wz y z ],
                             [ wx wz x z , wy wz y z , wz^2 z^2 ]]

With the default w = (4, 4, 1)  (in-plane : normal stiffness = 16 : 1):

    H_ij = -(1/r^3) [[16 x^2, 16 x y,  4 x z],
                     [16 x y, 16 y^2,  4 y z],
                     [ 4 x z,  4 y z,    z^2]]

Notes
-----
* The **scalar stiffness prefactor decays as 1/r^3** (distance-cubed). The
  raw displacement outer product (d ⊗ d) is used directly (NOT the unit-vector
  outer product), so the block equals the matrix above literally.
* z is assumed to be the membrane normal. For AF2 / non-oriented PDBs the
  structure is first rotated so the TM-helix bundle axis -> +z (see
  align_membrane_normal). For OPM/PPM structures pass --no-align.
* gamma is an overall scale that cancels in the normalised DFI, so its
  numerical value does not affect the reported metrics; the pseudo-inverse
  uses a *relative* eigenvalue cutoff to stay gamma-independent.
* The diagonal super-block accumulates the same weighted outer product
  (H_ii = -sum_j H_ij), preserving translational invariance.

Output CSV columns: ResI, ChainID, Res, R, DFI, DFI_membrane, DFI_XY, DFI_Z

Usage
-----
dfi_calc7.py --pdb PDBFILE [--chain CHAINID] [--cutoff DISTANCE]
             [--ratio 16] [--gamma 1.0] [--no-align] [--help]

    --ratio   in-plane : normal stiffness ratio k_xy/k_z (default 16 -> w=(4,4,1))
    --no-align  skip membrane-normal alignment (use for OPM/PPM oriented PDBs)
"""

import sys
import os
import zipfile
import tempfile
import numpy as np
import pandas as pd
from scipy import linalg as LA

# ------------------------------------------------------------
# GPCR transmembrane helix regions (residue numbers, 1-indexed inclusive)
# Used ONLY to estimate the membrane normal for alignment (not for
# applying anisotropy -- anisotropy is now global).
# ------------------------------------------------------------
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
}

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
# PDB reader
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
                    atom_name = line[12:16].strip()
                    if CAonly and atom_name != 'CA':
                        continue
                    alt_loc = line[16] if len(line) > 16 else ' '
                    if noalc and alt_loc not in (' ', 'A'):
                        continue
                    chainID = line[21] if len(line) > 21 else ' '
                    if chainA and chainID != chain_name:
                        continue
                    record = line[0:6].strip()
                    atom_index = line[6:11].strip()
                    res_name = line[17:20].strip()
                    res_index = line[22:27].strip()
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
# Membrane-normal alignment (for AlphaFold2 / non membrane-oriented PDBs)
# ------------------------------------------------------------
def _helix_axis(coords):
    """Principal axis (unit vector) of a set of CA coords via PCA, and centroid."""
    coords = np.asarray(coords, dtype=float)
    centroid = coords.mean(axis=0)
    centered = coords - centroid
    _, _, Vt = LA.svd(centered, full_matrices=False)
    axis = Vt[0]
    return axis, centroid


def _rotation_matrix_to_z(n):
    """Rotation matrix that rotates unit vector n onto [0,0,1] (Rodrigues)."""
    n = n / np.linalg.norm(n)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(n, z)
    s = np.linalg.norm(v)
    c = np.dot(n, z)
    if s < 1e-10:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    R = np.eye(3) + vx + vx.dot(vx) * ((1 - c) / (s ** 2))
    return R


def align_membrane_normal(ATOMS, pdbid, regions=None, min_helix_residues=6, Verbose=True):
    """
    Estimate the membrane normal from known TM helix regions and rotate the
    whole structure so that this normal becomes +z; recenter so the TM-region
    centroid (membrane mid-plane) sits at z=0.

    Needed for AF2 models / most PDB depositions, which are NOT pre-oriented
    with the membrane normal along z. Without this the XY/Z decomposition and
    the global anisotropy (which hard-codes z = normal) have no physical
    meaning. For OPM/PPM structures, skip this (align_membrane=False).
    """
    if regions is None:
        regions = GPCR_TM_REGIONS.get(pdbid)
    if not regions:
        print(f"Warning: no TM regions known for '{pdbid}', cannot align "
              f"membrane normal. Pass `regions=` explicitly or use --no-align.",
              file=sys.stderr)
        return ATOMS

    ca_by_res = {}
    for atom in ATOMS:
        if atom.atom_name == 'CA':
            try:
                ca_by_res[int(atom.res_index)] = atom
            except ValueError:
                continue

    axes = []
    region_centroids = []
    for (start, end) in regions:
        coords = []
        first_atom, last_atom = None, None
        for resnum in range(start, end + 1):
            if resnum in ca_by_res:
                atom = ca_by_res[resnum]
                coords.append([atom.x, atom.y, atom.z])
                if first_atom is None:
                    first_atom = atom
                last_atom = atom
        if len(coords) < min_helix_residues:
            continue
        axis, centroid = _helix_axis(coords)
        nc_vec = np.array([last_atom.x - first_atom.x,
                           last_atom.y - first_atom.y,
                           last_atom.z - first_atom.z])
        if np.dot(axis, nc_vec) < 0:
            axis = -axis
        axes.append(axis)
        region_centroids.append(centroid)

    if len(axes) < 2:
        print(f"Warning: only {len(axes)} usable TM helices found for '{pdbid}' "
              f"(need >=2). Skipping membrane alignment.", file=sys.stderr)
        return ATOMS

    axes = np.array(axes)
    # Average *lines* (sign-invariant) via dominant eigenvector of sum of
    # axis outer products -- adjacent TM helices alternate up/down topology.
    outer_sum = np.zeros((3, 3))
    for axis in axes:
        outer_sum += np.outer(axis, axis)
    eigvals, eigvecs = np.linalg.eigh(outer_sum)
    membrane_normal = eigvecs[:, np.argmax(eigvals)]

    R = _rotation_matrix_to_z(membrane_normal)

    for atom in ATOMS:
        v = np.array([atom.x, atom.y, atom.z])
        v_rot = R.dot(v)
        atom.x, atom.y, atom.z = v_rot[0], v_rot[1], v_rot[2]

    rotated_region_centroids = []
    for (start, end) in regions:
        coords = []
        for resnum in range(start, end + 1):
            if resnum in ca_by_res:
                atom = ca_by_res[resnum]
                coords.append([atom.x, atom.y, atom.z])
        if len(coords) >= min_helix_residues:
            rotated_region_centroids.append(np.mean(coords, axis=0))
    if rotated_region_centroids:
        z_offset = np.mean([c[2] for c in rotated_region_centroids])
        for atom in ATOMS:
            atom.z -= z_offset

    if Verbose:
        print(f"Aligned '{pdbid}' to membrane normal using {len(axes)}/{len(regions)} "
              f"TM helices; recentered membrane mid-plane to z=0.")
    return ATOMS


# ------------------------------------------------------------
# Anisotropic distance-dependent Hessian (global imANM)
# ------------------------------------------------------------
def calchessian(x, y, z, gamma=1.0, cutoff=None, aniso=(1.0, 1.0, 1.0)):
    """
    Build the anisotropic ANM (imANM) Hessian, applied to EVERY residue pair.

    For each pair (i, j), with d = R_i - R_j, r = |d|, w = aniso:

        H_ij = -(gamma / r^3) * (w ⊗ w) ∘ (d ⊗ d)

    Equivalently, letting the scaled displacement d' = w * d:

        H_ij = -(gamma / r^3) * (d' ⊗ d')

    so with w = (4, 4, 1):

        H_ij = -(1/r^3) [[16 x^2, 16 x y,  4 x z],
                         [16 x y, 16 y^2,  4 y z],
                         [ 4 x z,  4 y z,    z^2]]

    The scalar stiffness prefactor is 1/r^3 (distance-cubed decay); r is the
    true (unscaled) inter-residue distance. The diagonal super-block
    accumulates +sum_j (weighted outer product) so H_ii = -sum_j H_ij, giving
    a symmetric PSD Hessian with the translational zero modes preserved.

    Parameters
    ----------
    x, y, z : 1D arrays of CA coordinates (z along the membrane normal)
    gamma   : overall scale (cancels in normalised DFI / ratio)
    cutoff  : optional distance cutoff (A); pairs beyond it are dropped
    aniso   : (w_x, w_y, w_z) per-axis stiffness weights
    """
    R = np.column_stack([x, y, z]).astype(float)      # (N,3)
    N = R.shape[0]
    w = np.asarray(aniso, dtype=float).reshape(3)     # (3,)
    H = np.zeros((3 * N, 3 * N))

    for i in range(N):
        d = R[i] - R                                  # (N,3): d_ij = R_i - R_j
        r = np.sqrt(np.einsum('ij,ij->i', d, d))      # (N,)
        r[i] = np.inf                                 # self -> prefactor 0

        pref = gamma / (r ** 3)                        # (N,): 1/r^3 stiffness
        if cutoff is not None:
            pref = np.where(r > cutoff, 0.0, pref)

        dw = d * w                                     # (N,3): scaled displacement d'
        # weighted outer product per j: block[j] = pref[j] * (d'_j ⊗ d'_j)
        block = pref[:, None, None] * (dw[:, :, None] * dw[:, None, :])   # (N,3,3)

        # off-diagonal strip for residue i:  H[i,j] = -block[j]
        H[3 * i:3 * i + 3, :] = -block.transpose(1, 0, 2).reshape(3, 3 * N)
        # diagonal super-block:  H[i,i] = +sum_j block[j]  (row-sum = 0)
        H[3 * i:3 * i + 3, 3 * i:3 * i + 3] = block.sum(axis=0)

    return H


# ------------------------------------------------------------
# Compute pseudo-inverse (symmetric PSD -> eigh, relative cutoff)
# ------------------------------------------------------------
def compute_pseudo_inverse(hess, rtol=1e-9):
    """
    Moore-Penrose pseudo-inverse of a symmetric PSD Hessian via eigendecomposition.
    Eigenvalues below rtol * max(|eigenvalue|) (the translational / broken-
    rotational zero modes) are discarded. Using a *relative* cutoff makes the
    result independent of the overall gamma scale.
    """
    hess = 0.5 * (hess + hess.T)                       # enforce exact symmetry
    evals, evecs = LA.eigh(hess)
    tol = rtol * np.max(np.abs(evals))
    inv = np.where(evals > tol, 1.0 / evals, 0.0)
    return (evecs * inv) @ evecs.T


# ------------------------------------------------------------
# Perturbation-response matrices (vectorised; per-residue XY / Z / Total)
# ------------------------------------------------------------
def calcperturbMat(invHrs, directions, resnum, Normalize=True):
    """
    Perturbation-response matrices for Total, XY and Z displacement magnitudes.

    mat_*[m, j] = (1/n_dirs) * sum_k |response of residue m to a unit force in
                  direction k applied at residue j|_(Total / XY / Z)

    Mathematically identical to the residue-by-residue loop, but computed with
    one matvec per perturbation direction. If Normalize, each matrix is
    independently divided by its own sum.
    """
    N = resnum
    n_dirs = len(directions)
    # group Hessian-inverse columns by residue: invH_r[row, j, a] = invHrs[row, 3j+a]
    invH_r = invHrs.reshape(3 * N, N, 3)

    mat_total = np.zeros((N, N))
    mat_xy = np.zeros((N, N))
    mat_z = np.zeros((N, N))

    for p in directions:
        # Disp[row, j] = displacement (row = 3m+b) from unit force p at residue j
        Disp = np.tensordot(invH_r, p, axes=([2], [0]))   # (3N, N)
        Disp = Disp.reshape(N, 3, N)                      # (m, xyz, j)
        mat_total += np.sqrt(np.einsum('mcj,mcj->mj', Disp, Disp))
        mat_xy += np.sqrt(np.einsum('mcj,mcj->mj', Disp[:, :2, :], Disp[:, :2, :]))
        mat_z += np.abs(Disp[:, 2, :])

    mat_total /= n_dirs
    mat_xy /= n_dirs
    mat_z /= n_dirs

    if Normalize:
        if mat_total.sum() > 0:
            mat_total /= mat_total.sum()
        if mat_xy.sum() > 0:
            mat_xy /= mat_xy.sum()
        if mat_z.sum() > 0:
            mat_z /= mat_z.sum()

    return mat_total, mat_xy, mat_z


# ------------------------------------------------------------
# Main calculation for a single condition (writes CSV)
# ------------------------------------------------------------
def calc_dfi_single(pdbfile, pdbid, covar=None, chain_name=None, cutoff=None,
                    gamma=1.0, aniso=(1.0, 1.0, 1.0), suffix='', align_membrane=True,
                    write_csv=True):
    """
    Compute DFI for one anisotropy setting and save CSV (if write_csv=True).
    Columns: ResI, ChainID, Res, R, DFI_membrane, DFI_XY, DFI_Z
    """
    if not pdbid:
        pdbid = os.path.splitext(os.path.basename(pdbfile))[0]

    ATOMS = pdb_reader(pdbfile, CAonly=True, noalc=True,
                       chainA=(chain_name is not None), chain_name=chain_name or 'A')

    if len(ATOMS) == 0:
        print(f"Error: No CA atoms found in {pdbfile}. Check chain ID or file format.",
              file=sys.stderr)
        sys.exit(1)

    if align_membrane:
        align_membrane_normal(ATOMS, pdbid)
    else:
        print("Skipping membrane-normal alignment (align_membrane=False); "
              "assuming coordinates are already membrane-oriented (e.g. OPM/PPM).")

    x, y, z = getcoords(ATOMS)
    numres = len(ATOMS)

    if covar is None:
        wv = np.asarray(aniso, dtype=float)
        blkscale = tuple(round(float(v), 4) for v in wv ** 2)
        print(f"Building anisotropic Hessian  (weights w = "
              f"{tuple(round(float(v), 4) for v in wv)},  "
              f"[xx,yy,zz] block scale = {blkscale})...")
        hess = calchessian(x, y, z, gamma=gamma, cutoff=cutoff, aniso=aniso)
        print("Computing pseudo-inverse...")
        invHrs = compute_pseudo_inverse(hess)
    else:
        print(f"Loading precomputed covariance from {covar}")
        invHrs = np.loadtxt(covar)

    # 3 axial + 3 face-diagonal + 1 body-diagonal (7 directions)
    directions = np.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1],
        [1, 1, 1],
    ], dtype=float)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    directions = directions / norms

    print("Applying perturbations...")
    mat_total, mat_xy, mat_z = calcperturbMat(invHrs, directions, numres, Normalize=True)

    dfi_total = np.sum(mat_total, axis=1)
    dfi_xy = np.sum(mat_xy, axis=1)
    dfi_z = np.sum(mat_z, axis=1)

    mapres = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G',
              'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'PRO': 'P',
              'ARG': 'R', 'GLN': 'Q', 'ASN': 'N', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
              'TYR': 'Y', 'VAL': 'V'}

    dfx = pd.DataFrame()
    dfx['ResI'] = [atom.res_index for atom in ATOMS]
    dfx['ChainID'] = [atom.chainID for atom in ATOMS]
    dfx['Res'] = [atom.res_name for atom in ATOMS]
    dfx['R'] = dfx['Res'].map(mapres)
    dfx['DFI_membrane'] = dfi_total
    dfx['DFI_XY'] = dfi_xy
    dfx['DFI_Z'] = dfi_z

    if write_csv:
        outfile = f"{pdbid}{suffix}_anisotropy_analysis.csv"
        dfx.to_csv(outfile, index=False)
        print(f"Saved {outfile}")
    return dfx


# ------------------------------------------------------------
# HDF5 portable database — store results for multiple proteins
# ------------------------------------------------------------
def save_to_h5(h5file, pdbid, df_merged, tm_regions, ratio, gamma, overwrite=False):
    """
    Append (or overwrite) DFI results and TM regions for one protein
    into a portable HDF5 file.

    Group layout:

        /{pdbid}               — DFI data (ResI, ChainID, Res, R, DFI, DFI_membrane, DFI_XY, DFI_Z)
        /{pdbid}/tm_helices    — start, end  (one row per TM helix; empty if unknown)

    Parameters
    ----------
    h5file  : path to .h5 file
    pdbid   : protein identifier (e.g. "aa2ar")
    df_merged : DataFrame with DFI columns
    tm_regions : list of (start, end) tuples (may be empty)
    ratio, gamma : scalar parameters
    overwrite : if True, delete existing group before writing
    """
    import warnings
    # Re-entrant: if the group already exists we remove it so we don't
    # leave stale sub-groups from a previous run.
    with pd.HDFStore(h5file, mode='a') as store:
        if f'/{pdbid}' in store:
            store.remove(f'/{pdbid}')

    # DFI data — stored directly under the protein ID
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
        df_merged.to_hdf(h5file, key=pdbid, mode='a', format='fixed')

    # TM helices — (start, end) regions
    if tm_regions:
        tm_df = pd.DataFrame(tm_regions, columns=['start', 'end'])
    else:
        tm_df = pd.DataFrame(columns=['start', 'end'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
        tm_df.to_hdf(h5file, key=f'{pdbid}/tm_helices', mode='a', format='fixed')

    n_helices = len(tm_regions)
    print(f"  -> saved to {h5file}  [{pdbid}]  ({n_helices} TM helices)")


def read_from_h5(h5file, pdbid):
    """
    Read DFI results and TM regions for a single protein from an HDF5 file.

    Returns
    -------
    dfi : DataFrame  (ResI, ChainID, Res, R, DFI, DFI_membrane, DFI_XY, DFI_Z)
    tm  : list of (start, end) tuples
    """
    with pd.HDFStore(h5file, mode='r') as store:
        dfi = store[pdbid]
        try:
            tm_df = store[f'{pdbid}/tm_helices']
            tm = [(int(r.start), int(r.end)) for r in tm_df.itertuples()]
        except KeyError:
            tm = []
    return dfi, tm


def list_proteins_h5(h5file):
    """
    List all protein IDs stored in an HDF5 file.

    Returns
    -------
    list of protein ID strings
    """
    with pd.HDFStore(h5file, mode='r') as store:
        keys = [k.lstrip('/').split('/')[0] for k in store.keys()]
    return sorted(set(keys))


# ------------------------------------------------------------
# Zip file support — extract PDB from GPCRdb-style zip archives
# ------------------------------------------------------------
def extract_pdb_from_zip(zip_path, work_dir=None):
    """
    Extract the first .pdb file from a GPCRdb-style zip archive.
    Returns the path to the extracted PDB file.
    If work_dir is given, extract there; otherwise use a temp directory.
    """
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="dfi_")
    os.makedirs(work_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        pdb_names = [n for n in zf.namelist() if n.lower().endswith('.pdb')]
        if not pdb_names:
            print(f"Error: No .pdb file found in {zip_path}", file=sys.stderr)
            sys.exit(1)
        target = pdb_names[0]
        zf.extract(target, work_dir)
        extracted = os.path.join(work_dir, target)
        print(f"Extracted PDB from zip: {extracted}")
        return extracted


def resolve_pdb_input(path):
    """If path is a .zip file, extract the PDB inside and return its path.
    Otherwise return the path unchanged."""
    if path.lower().endswith('.zip'):
        return extract_pdb_from_zip(path)
    return path


# ------------------------------------------------------------
# Command line parsing
# ------------------------------------------------------------
def parseCommandLine(argv):
    comline_arg = {}
    flags = set()
    i = 1
    while i < len(argv):
        opt = argv[i]
        if opt in ['--pdb', '--hess', '--chain', '--cutoff', '--ratio', '--gamma', '--h5', '--pdbid']:
            if i + 1 < len(argv):
                comline_arg[opt] = argv[i + 1]
                i += 2
            else:
                i += 1
        elif opt == '--no-align':
            flags.add('--no-align')
            i += 1
        elif opt == '--no-csv':
            flags.add('--no-csv')
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

    ratio = 16.0
    if '--ratio' in comline_arg:
        try:
            ratio = float(comline_arg['--ratio'])
        except ValueError:
            print("Warning: --ratio must be a number. Using default 16.", file=sys.stderr)

    gamma = 1.0
    if '--gamma' in comline_arg:
        try:
            gamma = float(comline_arg['--gamma'])
        except ValueError:
            print("Warning: --gamma must be a number. Using default 1.0.", file=sys.stderr)

    # Auto-detect short protein ID from filename (e.g. "ClassA_aa2ar_..." -> "aa2ar")
    raw_pdbid = os.path.splitext(os.path.basename(comline_arg['--pdb']))[0]
    pdbid = comline_arg.get('--pdbid', None)
    if pdbid is None:
        # Try to match known GPCR names in the filename
        for known_id in GPCR_TM_REGIONS:
            if known_id in raw_pdbid.lower():
                pdbid = known_id
                print(f"Auto-detected protein ID: {pdbid} (from filename)")
                break
        if pdbid is None:
            pdbid = raw_pdbid
            print(f"Warning: Could not auto-detect protein ID from '{raw_pdbid}'. "
                  f"Use --pdbid to specify one of: {sorted(GPCR_TM_REGIONS.keys())}")

    return (comline_arg['--pdb'],
            pdbid,
            comline_arg.get('--hess', None),
            comline_arg.get('--chain', None),
            cutoff,
            ratio,
            gamma,
            '--no-align' not in flags,
            comline_arg.get('--h5', None),
            '--no-csv' not in flags)


# ------------------------------------------------------------
# Main: isotropic baseline + global anisotropic imANM
# ------------------------------------------------------------
if __name__ == "__main__":
    (pdbfile, pdbid, covar, chain_name, cutoff,
     ratio, gamma, align_membrane, h5file, write_csv) = parseCommandLine(sys.argv)

    # Resolve zip input -> extract PDB if needed
    pdbfile = resolve_pdb_input(pdbfile)

    if not os.path.isfile(pdbfile):
        print(f"Error: Target PDB file does not exist: {pdbfile}", file=sys.stderr)
        sys.exit(1)

    # Output directory
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'DFI')
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f'{pdbid}.csv')

    # in-plane : normal stiffness ratio  ->  weight vector w = (sqrt(S), sqrt(S), 1)
    wxy = float(np.sqrt(ratio))

    print(f"Processing {pdbfile}  (ratio k_xy/k_z = {ratio}, gamma = {gamma}, "
          f"align_membrane = {align_membrane})")
    print(f"Output CSV   : {out_csv}")

    # --- isotropic run (DFI only) ---
    print("\n[1/2] Isotropic (w = (1,1,1)) ...")
    df_iso = calc_dfi_single(pdbfile, pdbid, covar=covar, chain_name=chain_name,
                              cutoff=cutoff, gamma=gamma, aniso=(1.0, 1.0, 1.0),
                              suffix='_isotropic', align_membrane=align_membrane,
                              write_csv=False)

    # --- anisotropic run (DFI_membrane, DFI_XY, DFI_Z) ---
    print(f"\n[2/2] imANM (w = ({wxy:.4f}, {wxy:.4f}, 1.0)) ...")
    df_aniso = calc_dfi_single(pdbfile, pdbid, covar=covar, chain_name=chain_name,
                                cutoff=cutoff, gamma=gamma, aniso=(wxy, wxy, 1.0),
                                suffix=f'_imANM_kxy{int(round(ratio))}',
                                align_membrane=align_membrane, write_csv=False)

    # --- merge ---
    # from isotropic: total DFI -> rename to DFI
    # from anisotropic: membrane-constrained DFI (DFI_membrane), XY / Z components
    df_merged = df_iso[['ResI', 'ChainID', 'Res', 'R']].copy()
    df_merged['DFI'] = df_iso['DFI_membrane']
    df_merged['DFI_membrane'] = df_aniso['DFI_membrane']
    df_merged['DFI_XY'] = df_aniso['DFI_XY']
    df_merged['DFI_Z'] = df_aniso['DFI_Z']

    # --- TM regions for this protein ---
    tm_regions = GPCR_TM_REGIONS.get(pdbid, [])

    # --- output ---
    df_merged.to_csv(out_csv, index=False)
    print(f"\nSaved CSV: {out_csv}")

    print(f"Columns: ResI, ChainID, Res, R, DFI, DFI_membrane, DFI_XY, DFI_Z")
    print("All conditions completed.")
