#!/usr/bin/env python3
"""
DFI (Dynamic Flexibility Index) & Effector Anisotropy Analysis
===============================================================

Calculates DFI and directional effector metrics with optional membrane constraints.
Two modes for applying membrane constraints:
  - seq : uses predefined GPCR_TM_REGIONS (by residue number)
  - z   : detects residues within |z| < depth (membrane half-thickness)

Automatically generates 6 conditions:
  - No membrane, Total DFI
  - No membrane, Z DFI
  - Membrane (scale=16), Total DFI
  - Membrane (scale=16), Z DFI
  - Membrane (scale=100), Total DFI
  - Membrane (scale=100), Z DFI

Each output CSV contains columns: ResI, ChainID, Res, R, DFI_Total, DFI_XY, DFI_Z, Effector_Ratio_Z

Usage
-----
dfi_calc5_bak.py --pdb PDBFILE [--hess HESSFILE] [--chain CHAINID] [--cutoff DISTANCE]
            [--mode {seq,z}] [--depth FLOAT] [--help]
"""

import sys
import os
import numpy as np
import pandas as pd
from scipy import linalg as LA

# ------------------------------------------------------------
# GPCR transmembrane helix regions (residue numbers, 1-indexed inclusive)
# ------------------------------------------------------------
GPCR_TM_REGIONS = {
    "aa2ar": [(8, 32), (43, 66), (79, 108), (119, 142), (173, 202), (235, 258), (267, 290)],
    #"oprm": [(69, 93), (107, 131), (147, 170), (196, 218), (236, 255), (280, 306), (315, 339)],
    "adrb2": [(29, 60), (67, 96), (103, 136), (147, 171), (197, 229), (267, 298), (305, 331)],
    #"opsd": [(35, 64), (71, 99), (107, 139), (150, 173), (200, 226), (250, 277), (286, 309)],
    "cxcr4": [(39, 68), (78, 106), (112, 143), (157, 180), (203, 229), (240, 268), (280, 305)],
    "aa1r": [(12, 32), (48, 68), (78, 98), (128, 147), (174, 196), (237, 259), (268, 289)],
    
    # Adenosine receptor A2b (ADORA2B)
    "aa2br": [(9, 33), (44, 67), (79, 101), (122, 144), (179, 203), (236, 259), (268, 291)],
    
    # Muscarinic acetylcholine receptor M1 (CHRM1)
    "acm1": [(23, 48), (63, 84), (105, 126), (143, 164), (187, 210), (351, 372), (385, 407)],
    
    # C-C chemokine receptor type 1 (CCR1)
    "ccr1": [(35, 60), (73, 95), (108, 129), (151, 175), (198, 223), (240, 264), (282, 305)],
    
    # C-C chemokine receptor type 5 (CCR5)
    "ccr5": [(31, 58), (69, 89), (103, 124), (142, 166), (199, 218), (236, 261), (271, 295)],
    
    # C-X-C chemokine receptor type 1 (CXCR1)
    "cxcr1": [(39, 65), (76, 96), (111, 132), (153, 176), (198, 220), (243, 267), (277, 302)],
    
    # Glucagon-like peptide 1 receptor (GLP1R)
    "glp1r": [(145, 165), (174, 194), (228, 248), (271, 291), (317, 337), (350, 370), (383, 403)],
    
    # Melanocortin receptor 4 (MC4R)
    "mc4r": [(44, 64), (77, 97), (115, 135), (154, 174), (196, 216), (247, 267), (280, 300)]
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
    """Return the principal axis (unit vector) of a set of CA coordinates
    via PCA, and the centroid."""
    coords = np.asarray(coords, dtype=float)
    centroid = coords.mean(axis=0)
    centered = coords - centroid
    # SVD on centered coords; first right-singular vector = principal axis
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
        # already aligned (or exactly opposite)
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    R = np.eye(3) + vx + vx.dot(vx) * ((1 - c) / (s ** 2))
    return R

def align_membrane_normal(ATOMS, pdbid, regions=None, min_helix_residues=6, Verbose=True):
    """
    Estimate the membrane normal from known TM helix regions (sequence-based,
    via GPCR_TM_REGIONS) and rotate the whole structure so that this normal
    becomes the +z axis. Also recenters the structure so the TM-residue
    centroid (approx. membrane mid-plane) sits at z=0.

    This is needed for AlphaFold2 models (and most PDB depositions), which
    are NOT pre-oriented with the membrane normal along z the way OPM/PPM
    structures are. Without this step, mode='z' membrane detection and the
    XY/Z decomposition of DFI have no physical meaning.

    Only works for receptors present in GPCR_TM_REGIONS (or an explicitly
    supplied `regions` list of (start,end) residue ranges). Returns the
    (possibly unchanged) ATOMS list; modifies atom.x/y/z in place.
    """
    if regions is None:
        regions = GPCR_TM_REGIONS.get(pdbid)
    if not regions:
        print(f"Warning: no TM regions known for '{pdbid}', cannot align "
              f"membrane normal. Pass `regions=` explicitly or skip alignment.",
              file=sys.stderr)
        return ATOMS

    # Map res_index -> CA atom, so each TM region's coordinates can be pulled
    ca_by_res = {}
    for atom in ATOMS:
        if atom.atom_name == 'CA':
            try:
                ca_by_res[int(atom.res_index)] = atom
            except ValueError:
                continue

    axes = []
    region_centroids = []
    region_endpoint_vecs = []  # N->C direction, used to keep axis orientation consistent
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
        # Orient the PCA axis along the helix's actual N->C direction so all
        # helices can be averaged consistently (PCA gives an axis, not a
        # direction; sign is arbitrary).
        if np.dot(axis, nc_vec) < 0:
            axis = -axis
        axes.append(axis)
        region_centroids.append(centroid)

    if len(axes) < 2:
        print(f"Warning: only {len(axes)} usable TM helices found for '{pdbid}' "
              f"(need >=2). Skipping membrane alignment.", file=sys.stderr)
        return ATOMS

    axes = np.array(axes)
    # Membrane normal ~ average helix-axis direction is NOT quite right by
    # itself, because adjacent TM helices alternate N->C direction (up-down
    # topology). What's common across all of them is the *line* they define,
    # not necessarily a consistently-signed vector. To average lines (not
    # vectors) we use the dominant eigenvector of the outer-product
    # (axis . axis^T) sum, which is sign-invariant.
    outer_sum = np.zeros((3, 3))
    for axis in axes:
        outer_sum += np.outer(axis, axis)
    eigvals, eigvecs = np.linalg.eigh(outer_sum)
    membrane_normal = eigvecs[:, np.argmax(eigvals)]

    R = _rotation_matrix_to_z(membrane_normal)

    # Rotate every atom
    for atom in ATOMS:
        v = np.array([atom.x, atom.y, atom.z])
        v_rot = R.dot(v)
        atom.x, atom.y, atom.z = v_rot[0], v_rot[1], v_rot[2]

    # Recenter so the TM-region centroid sits at z=0 (approx. membrane
    # mid-plane), recomputed post-rotation.
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

# ------------------------------------------------------------
# Compute pseudo-inverse
# ------------------------------------------------------------
def compute_pseudo_inverse(hess):
    U, w, Vt = LA.svd(hess, full_matrices=False)
    tol = 1e-6
    invw = np.zeros_like(w)
    mask = w > tol
    invw[mask] = 1.0 / w[mask]
    return np.dot(np.dot(U, np.diag(invw)), Vt)

# ------------------------------------------------------------
# Membrane constraint application
# ------------------------------------------------------------
def apply_membrane_constraints(hess, ATOMS, mode='seq', depth=15.0, scale=16.0, pdbid=None):
    """
    Modify Hessian to enforce membrane constraints.
    mode: 'seq'  -> use GPCR_TM_REGIONS (requires pdbid)
          'z'    -> use Z-coordinate, |z| < depth
    For residues inside membrane, set XX and YY diagonal stiffness = scale * ZZ.
    """
    numres = len(ATOMS)
    membrane_indices = []

    if mode == 'seq':
        if pdbid not in GPCR_TM_REGIONS:
            print(f"Warning: No TM regions defined for {pdbid}, skipping membrane constraints.")
            return hess
        regions = GPCR_TM_REGIONS[pdbid]
        for i, atom in enumerate(ATOMS):
            res_num = int(atom.res_index)
            if any(start <= res_num <= end for start, end in regions):
                membrane_indices.append(i)
        print(f"Using sequence-based TM regions: {len(membrane_indices)} residues in membrane.")
    elif mode == 'z':
        for i, atom in enumerate(ATOMS):
            if abs(atom.z) < depth:
                membrane_indices.append(i)
        print(f"Using Z-coordinate based detection (|z| < {depth} Å): {len(membrane_indices)} residues in membrane.")
    else:
        raise ValueError("mode must be 'seq' or 'z'")

    if not membrane_indices:
        print("No membrane residues found, skipping constraints.")
        return hess

    for i in membrane_indices:
        hz = hess[3*i+2, 3*i+2]
        # Set XY stiffness to scale * Z stiffness
        hess[3*i, 3*i] = scale * hz
        hess[3*i+1, 3*i+1] = scale * hz
        # Note: off-diagonal XY terms are not modified
    return hess

# ------------------------------------------------------------
# Perturbation matrix calculation (returns per-residue averages)
# ------------------------------------------------------------
def calcperturbMat(invHrs, directions, resnum, Normalize=True):
    """
    Computes perturbation matrices for total, XY, and Z displacements.
    Returns three matrices (resnum x resnum) and also arrays of per‑residue
    average total displacement and Z displacement (for ratio calculation).
    """
    n_dirs = len(directions)
    perturbMat_total = np.zeros((resnum, resnum))
    perturbMat_xy    = np.zeros((resnum, resnum))
    perturbMat_z     = np.zeros((resnum, resnum))

    for k in range(n_dirs):
        perturbDir = directions[k, :]
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
        # IMPORTANT: normalize total/xy/z by the SAME constant (the total
        # matrix's sum). Normalizing each matrix by its own sum independently
        # rescales xy and z onto artificially equal "budgets" (each summing
        # to 1), which destroys the real relative magnitude between lateral
        # and normal-direction motion and makes Effector_Ratio_Z meaningless
        # for comparing across residues/conditions.
        norm_const = np.sum(perturbMat_total)
        perturbMat_total /= norm_const
        perturbMat_xy    /= norm_const
        perturbMat_z     /= norm_const

    return perturbMat_total, perturbMat_xy, perturbMat_z

# ------------------------------------------------------------
# Main calculation function (outputs all DFI columns)
# ------------------------------------------------------------
def calc_dfi_single(pdbfile, pdbid, covar=None, chain_name=None, cutoff=None,
                    mode='seq', depth=15.0, no_membrane=False, membrane_scale=16.0,
                    suffix='', align_membrane=True):
    """
    Computes DFI for a single condition and saves CSV.
    Output columns: ResI, ChainID, Res, R, DFI_Total, DFI_XY, DFI_Z, Effector_Ratio_Z
    suffix: appended to output filename.
    """
    if not pdbid:
        pdbid = os.path.splitext(os.path.basename(pdbfile))[0]

    ATOMS = pdb_reader(pdbfile, CAonly=True, noalc=True,
                       chainA=(chain_name is not None), chain_name=chain_name or 'A')

    if len(ATOMS) == 0:
        print(f"Error: No CA atoms found in {pdbfile}. Check chain ID or file format.", file=sys.stderr)
        sys.exit(1)

    if align_membrane:
        align_membrane_normal(ATOMS, pdbid)
    else:
        print("Skipping membrane-normal alignment (align_membrane=False); "
              "assuming coordinates are already membrane-oriented (e.g. OPM/PPM).")

    x, y, z = getcoords(ATOMS)
    numres = len(ATOMS)

    if covar is None:
        print("Building Hessian...")
        hess = calchessian(numres, x, y, z, cutoff=cutoff)

        if not no_membrane:
            print(f"Applying membrane constraints with scale={membrane_scale}")
            hess = apply_membrane_constraints(hess, ATOMS, mode=mode, depth=depth,
                                               scale=membrane_scale, pdbid=pdbid)
        else:
            print("No membrane constraints applied.")

        print("Computing pseudo-inverse...")
        invHrs = compute_pseudo_inverse(hess)
    else:
        print(f"Loading precomputed covariance from {covar}")
        invHrs = np.loadtxt(covar)

    # Perturbation directions
    directions = np.array([
        [ 1,  0,  0],
        [-1,  0,  0],
        [ 0,  1,  0],
        [ 0, -1,  0],
        [ 0,  0,  1],
        [ 0,  0, -1]
    ], dtype=float)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    directions = directions / norms

    print("Applying perturbations...")
    mat_total, mat_xy, mat_z = calcperturbMat(invHrs, directions, numres, Normalize=True)

    # Compute per‑residue sums (DFI)
    dfi_total = np.sum(mat_total, axis=1)
    dfi_xy    = np.sum(mat_xy, axis=1)
    dfi_z     = np.sum(mat_z, axis=1)

    # Effector Ratio Z: sum_Z / (sum_XY + sum_Z)
    sum_xy = np.sum(mat_xy, axis=1)
    sum_z  = np.sum(mat_z, axis=1)
    denom = sum_xy + sum_z
    ratio_z = np.divide(sum_z, denom, out=np.zeros_like(sum_z), where=denom>1e-12)

    # Build DataFrame
    mapres = {'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H',
              'ILE':'I','LYS':'K','LEU':'L','MET':'M','PRO':'P','ARG':'R','GLN':'Q',
              'ASN':'N','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}

    dfx = pd.DataFrame()
    dfx['ResI'] = [atom.res_index for atom in ATOMS]
    dfx['ChainID'] = [atom.chainID for atom in ATOMS]
    dfx['Res'] = [atom.res_name for atom in ATOMS]
    dfx['R'] = dfx['Res'].map(mapres)
    dfx['DFI_Total'] = dfi_total
    dfx['DFI_XY']    = dfi_xy
    dfx['DFI_Z']     = dfi_z
    dfx['Effector_Ratio_Z'] = ratio_z

    outfile = f"{pdbid}{suffix}_anisotropy_analysis.csv"
    dfx.to_csv(outfile, index=False)
    print(f"Saved {outfile} (DFI_Total, DFI_XY, DFI_Z, Effector_Ratio_Z)")
    return dfx

# ------------------------------------------------------------
# Command line parsing
# ------------------------------------------------------------
def parseCommandLine(argv):
    comline_arg = {}
    i = 1
    while i < len(argv):
        opt = argv[i]
        if opt in ['--pdb', '--hess', '--chain', '--cutoff', '--depth']:
            if i+1 < len(argv):
                comline_arg[opt] = argv[i+1]
                i += 2
            else:
                i += 1
        elif opt == '--mode':
            if i+1 < len(argv):
                comline_arg['--mode'] = argv[i+1]
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

    depth = 15.0
    if '--depth' in comline_arg:
        try:
            depth = float(comline_arg['--depth'])
        except ValueError:
            print("Warning: --depth must be a number. Using default 15.0.", file=sys.stderr)

    mode = comline_arg.get('--mode', 'seq')
    if mode not in ['seq', 'z']:
        print("Warning: --mode must be 'seq' or 'z'. Using 'seq'.", file=sys.stderr)
        mode = 'seq'

    return (comline_arg['--pdb'],
            os.path.splitext(os.path.basename(comline_arg['--pdb']))[0],
            comline_arg.get('--hess', None),
            comline_arg.get('--chain', None),
            cutoff,
            mode,
            depth)

# ------------------------------------------------------------
# Main: automatically run all 6 conditions
# ------------------------------------------------------------
if __name__ == "__main__":
    pdbfile, pdbid, covar, chain_name, cutoff, mode, depth = parseCommandLine(sys.argv)

    if not os.path.isfile(pdbfile):
        print(f"Error: Target PDB file does not exist: {pdbfile}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {pdbfile} for all 6 conditions...")

    # Define all conditions
    conditions = [
        {'no_membrane': True,  'scale': None,   'suffix': '_noMemb_total'},
        {'no_membrane': True,  'scale': None,   'suffix': '_noMemb_Z'},
        {'no_membrane': False, 'scale': 2.0,   'suffix': '_memb16_total'},
        {'no_membrane': False, 'scale': 2.0,   'suffix': '_memb16_Z'},
        {'no_membrane': False, 'scale': 100.0,  'suffix': '_memb100_total'},
        {'no_membrane': False, 'scale': 100.0,  'suffix': '_memb100_Z'},
    ]

    for cond in conditions:
        try:
            calc_dfi_single(pdbfile, pdbid, covar=covar, chain_name=chain_name,
                            cutoff=cutoff, mode=mode, depth=depth,
                            no_membrane=cond['no_membrane'],
                            membrane_scale=cond['scale'],
                            suffix=cond['suffix'])
        except Exception as e:
            print(f"Error processing condition {cond['suffix']}: {e}", file=sys.stderr)
            continue

    print("All conditions completed.")
