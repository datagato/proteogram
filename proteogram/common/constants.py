import math
import numpy as np

RESIDUE_LIST = [
    ("A", "ALA"),
    ("R", "ARG"),
    ("N", "ASN"),
    ("D", "ASP"),
    ("C", "CYS"),
    ("Q", "GLN"),
    ("E", "GLU"),
    ("G", "GLY"),
    ("H", "HIS"),
    ("I", "ILE"),
    ("L", "LEU"),
    ("K", "LYS"),
    ("M", "MET"),
    ("F", "PHE"),
    ("P", "PRO"),
    ("S", "SER"),
    ("T", "THR"),
    ("W", "TRP"),
    ("Y", "TYR"),
    ("V", "VAL"),
]

MODIFIED_RESIDUES_TO_STANDARD = {
            # Methionine variants
            'MSE': 'MET',  # Selenomethionine
            'FME': 'MET',  # N-formylmethionine
            'CXM': 'MET',  # N-carboxymethionine
            # Lysine variants
            'M3L': 'LYS',  # N6,N6,N6-trimethyllysine
            'MLY': 'LYS',  # N6-methyllysine
            'MLZ': 'LYS',  # N6-methyllysine (alt code)
            'KCX': 'LYS',  # Lysine NZ-carboxylic acid
            'ALY': 'LYS',  # N6-acetyllysine
            'LLP': 'LYS',  # Lysine-pyridoxal-5-phosphate
            # Cysteine variants
            'CSO': 'CYS',  # S-hydroxycysteine
            'CME': 'CYS',  # S,S-(2-hydroxyethyl)thiocysteine
            'OCS': 'CYS',  # Cysteinesulfonic acid
            'SEC': 'CYS',  # Selenocysteine
            'SMC': 'CYS',  # S-methylcysteine
            'CSD': 'CYS',  # 3-sulfoalanine (treated as Cys)
            # Serine/Threonine variants
            'SEP': 'SER',  # Phosphoserine
            'TPO': 'THR',  # Phosphothreonine
            # Tyrosine variants
            'PTR': 'TYR',  # Phosphotyrosine
            'TYS': 'TYR',  # Sulfotyrosine
            # Proline variants
            'HYP': 'PRO',  # 4-hydroxyproline
            # Aspartate variants
            'BHD': 'ASP',  # (3S)-3-hydroxy-L-aspartic acid (beta-hydroxyaspartic acid)
            # Glutamate/Glutamine variants
            'CGU': 'GLU',  # Gamma-carboxyglutamate
            'PCA': 'GLN',  # Pyroglutamate (from Gln)
            # Histidine variants
            'NEP': 'HIS',  # N1-phosphohistidine
            'HIC': 'HIS',  # 4-methyl-histidine
        }

# Solvent residue names to exclude when parsing PDB files. This is not an exhaustive list, but covers common cases.
SOLVENT_RESIDUES = {'HOH', 'WAT', 'TIP3', 'SOL', 'NA', 'CL', 'K', 'MG', 'ZN'}

UNKNOWN_RESIDUE = ("X", "UNK")

BACKBONE_ATOMS = ["N", "CA", "C"]

BACKBONE_POSITIONS = {
    "ALA": [
        ("N", [-0.525, 1.363, 0.000]),
        ("CA", [0.000, 0.000, 0.000]),
        ("C", [1.526, -0.000, -0.000]),
    ],
    "ARG": [
        ("N", [-0.524, 1.362, -0.000]),
        ("CA", [0.000, 0.000, 0.000]),
        ("C", [1.525, -0.000, -0.000]),
    ],
    "ASN": [
        ("N", [-0.536, 1.357, 0.000]),
        ("CA", [0.000, 0.000, 0.000]),
        ("C", [1.526, -0.000, -0.000]),
    ],
    "ASP": [
        ("N", [-0.525, 1.362, -0.000]),
        ("CA", [0.000, 0.000, 0.000]),
        ("C", [1.527, 0.000, -0.000]),
    ],
    "CYS": [
        ("N", [-0.522, 1.362, -0.000]),
        ("CA", [0.000, 0.000, 0.000]),
        ("C", [1.524, 0.000, 0.000]),
    ],
    "GLN": [
        ("N", [-0.526, 1.361, -0.000]),
        ("CA", [0.000, 0.000, 0.000]),
        ("C", [1.526, 0.000, 0.000]),
    ],
    "GLU": [
        ("N", [-0.528, 1.361, 0.000]),
        ("CA", [0.000, 0.000, 0.000]),
        ("C", [1.526, -0.000, -0.000]),
    ],
    "GLY": [
        ("N", [-0.572, 1.337, 0.000]),
        ("CA", [0.000, 0.000, 0.000]),
        ("C", [1.517, -0.000, -0.000]),
    ],
    "HIS": [
        ("N", [-0.527, 1.360, 0.000]),
        ("CA", [0.000, 0.000, 0.000]),
        ("C", [1.525, 0.000, 0.000]),
    ],
    "ILE": [
        ("N", [-0.493, 1.373, -0.000]),
        ("CA", [0.000, 0.000, 0.000]),
        ("C", [1.527, -0.000, -0.000]),
    ],
    "LEU": [
        ("N", [-0.520, 1.363, 0.000]),
        ("CA", [0.000, 0.000, 0.000]),
        ("C", [1.525, -0.000, -0.000]),
    ],
    "LYS": [
        ("N", [-0.526, 1.362, -0.000]),
        ("CA", [0.000, 0.000, 0.000]),
        ("C", [1.526, 0.000, 0.000]),
    ],
    "MET": [
        ("N", [-0.521, 1.364, -0.000]),
        ("CA", [0.000, 0.000, 0.000]),
        ("C", [1.525, 0.000, 0.000]),
    ],
    "PHE": [
        ("N", [-0.518, 1.363, 0.000]),
        ("CA", [0.000, 0.000, 0.000]),
        ("C", [1.524, 0.000, -0.000]),
    ],
    "PRO": [
        ("N", [-0.566, 1.351, -0.000]),
        ("CA", [0.000, 0.000, 0.000]),
        ("C", [1.527, -0.000, 0.000]),
    ],
    "SER": [
        ("N", [-0.529, 1.360, -0.000]),
        ("CA", [0.000, 0.000, 0.000]),
        ("C", [1.525, -0.000, -0.000]),
    ],
    "THR": [
        ("N", [-0.517, 1.364, 0.000]),
        ("CA", [0.000, 0.000, 0.000]),
        ("C", [1.526, 0.000, -0.000]),
    ],
    "TRP": [
        ("N", [-0.521, 1.363, 0.000]),
        ("CA", [0.000, 0.000, 0.000]),
        ("C", [1.525, -0.000, 0.000]),
    ],
    "TYR": [
        ("N", [-0.522, 1.362, 0.000]),
        ("CA", [0.000, 0.000, 0.000]),
        ("C", [1.524, -0.000, -0.000]),
    ],
    "VAL": [
        ("N", [-0.494, 1.373, -0.000]),
        ("CA", [0.000, 0.000, 0.000]),
        ("C", [1.527, -0.000, -0.000]),
    ],
    "UNK": [
        ("N", [0.000, 0.000, 0.000]),
        ("CA", [0.000, 0.000, 0.000]),
        ("C", [0.000, 0.000, 0.000]),
    ],
}

# 0 is hydrophobic / 1 is hydrophilic
HYDROPHOBICITY_LIST_BINARY = {
    "A": 0,
    "R": 1,
    "N": 1,
    "D": 1,
    "C": 0,
    "Q": 1,
    "E": 1,
    "G": 0,
    "H": 1,
    "I": 0,
    "L": 0,
    "K": 1,
    "M": 0,
    "F": 0,
    "P": 0,
    "S": 1,
    "T": 1,
    "W": 0,
    "Y": 1,
    "V": 0,
}

# -1 is negative / 100 is neutral (sentinel value) / 1 is positive
CHARGE_LIST = {
    "A": 100,
    "R": 1,
    "N": 100,
    "D": -1,
    "C": 100,
    "Q": 100,
    "E": -1,
    "G": 100,
    "H": 1,
    "I": 100,
    "L": 100,
    "K": 1,
    "M": 100,
    "F": 100,
    "P": 100,
    "S": 100,
    "T": 100,
    "W": 100,
    "Y": 100,
    "V": 100,
}

# Hydrophobicity table of AAs (numbers are relative to Gly which is set to 0)
HYDROPHOBICITY_LIST = {
    # Very hydrophobic
    'F': 100,
    'I': 99,
    'W': 97,
    'L': 97,
    'V': 76,
    'M': 74,
    
    # Hydrophobic
    'Y': 63,
    'C': 49,
    'A': 41,
    
    # Neutral
    'T': 13,
    'H': 8,
    'G': 0,
    'S': -5,
    'Q': -20,
    
    # Hydrophilic
    'R': -14,
    'K': -23,
    'N': -28,
    'E': -31,
    'P': -46,
    'D': -55
}

# ── Martini 3 CG force field constants ───────────────────────────────────────
# Sources: https://cgmartini.nl/docs/downloads/force-field-parameters/martini3/particle-definitions.html, https://github.com/maccallumlab/martini_openmm/tree/master/tutorial/martini_v3.0.0.itp and Tironi et al. (J. Chem. Phys. 1995) for RF parameters.
# Coulomb prefactor: k_e / ε_r where ε_r = 15 (protein-interior dielectric).
# k_rf and c_rf follow Tironi et al. (J. Chem. Phys. 1995).
MARTINI_COULOMB_K         = 138.935456 / 15.0   # kJ·nm/(mol·e²), ε_r = 15
MARTINI_SOLVENT_CUTOFF_NM = 1.1                  # nm — LJ and RF cutoff
MARTINI_EPS_WATER         = 80.0                 # bulk water dielectric
MARTINI_EPS_PROT          = 15.0                 # protein-interior dielectric
MARTINI_RF_K = ((MARTINI_EPS_WATER - MARTINI_EPS_PROT)
                / (2 * MARTINI_EPS_WATER + MARTINI_EPS_PROT)
                / MARTINI_SOLVENT_CUTOFF_NM ** 3)
MARTINI_RF_C = ((3 * MARTINI_EPS_WATER)
                / (2 * MARTINI_EPS_WATER + MARTINI_EPS_PROT)
                / MARTINI_SOLVENT_CUTOFF_NM)
MARTINI_SOLVENT_BUFFER_NM = 1.2                  # nm padding around protein
MARTINI_SALT_CONC_M       = 0.15                 # mol/L — physiological NaCl

# Bead type → (sigma nm, epsilon kJ/mol); values from martini_v3.0.0.itp nonbond_params.
# Combination rule 2 (Lorentz-Berthelot): σ_ij = (σ_i+σ_j)/2, ε_ij = sqrt(ε_i*ε_j).
MARTINI_BEAD_TYPE_PARAMS = {
    'BB':     (0.47, 4.06),  # backbone: P2 random-coil self-ε from ITP
    'C':      (0.47, 3.39),  # apolar regular (C1-C5): self-ε from ITP
    'SC':     (0.41, 2.35),  # apolar small (SC1-SC4): σ & ε from ITP
    'N':      (0.47, 3.52),  # polar (N1-N2): self-ε from ITP
    'Q':      (0.47, 5.95),  # charged (Q4): self-ε from ITP
    'TC':     (0.34, 1.51),  # tiny apolar (TC1-TC5): σ & ε from ITP
    'W':      (0.47, 1.00),  # water bead (~4 H₂O)
    'ION_NA': (0.354, 1.18), # Na+: TQ5 self-ε from ITP
    'ION_CL': (0.354, 1.18), # Cl-: TQ5 self-ε from ITP
}

# Bond force constants (kJ/(mol·nm²) and nm)
MARTINI_BB_BB_K  = 3800.0
MARTINI_BB_BB_R0 = 0.35
MARTINI_BB_SC_K  = 3800.0
MARTINI_BB_SC_R0 = 0.27
MARTINI_SC_SC_K  = 2500.0
MARTINI_SC_SC_R0 = 0.27

# Backbone BB–BB–BB angle (127° generic/random-coil)
MARTINI_BB_ANGLE_K     = 40.0                  # kJ/(mol·rad²)
MARTINI_BB_ANGLE_THETA = math.radians(127.0)   # radians

# Bead masses (Da)
MARTINI_BEAD_MASS_DA = 72.0
MARTINI_WATER_MASS   = 72.0   # W bead (4 × 18 Da)
MARTINI_ION_MASS_NA  = 22.99
MARTINI_ION_MASS_CL  = 35.45

# Per-residue bead definitions: (bead_type, bead_label, charge_e, [atom_names])
# BB bead is always first. Missing atoms fall back to CA automatically.
MARTINI_RESIDUE_BEADS = {
    'GLY': [
        ('BB',  'BB',  0.0,  ['N', 'CA', 'C', 'O']),
    ],
    'ALA': [
        ('BB',  'BB',  0.0,  ['N', 'CA', 'C', 'O']),
        ('TC',  'SC1', 0.0,  ['CB']),                      # TC1 in Martini 3
    ],
    'VAL': [
        ('BB',  'BB',  0.0,  ['N', 'CA', 'C', 'O']),
        ('SC',  'SC1', 0.0,  ['CB', 'CG1', 'CG2']),        # SC2 in Martini 3
    ],
    'LEU': [
        ('BB',  'BB',  0.0,  ['N', 'CA', 'C', 'O']),
        ('SC',  'SC1', 0.0,  ['CB', 'CG', 'CD1', 'CD2']),  # SC3 in Martini 3
    ],
    'ILE': [
        ('BB',  'BB',  0.0,  ['N', 'CA', 'C', 'O']),
        ('SC',  'SC1', 0.0,  ['CB', 'CG1', 'CG2', 'CD1']), # SC4 in Martini 3
    ],
    'PRO': [
        ('BB',  'BB',  0.0,  ['N', 'CA', 'C', 'O']),
        ('SC',  'SC1', 0.0,  ['CB', 'CG', 'CD']),           # SC3 in Martini 3
    ],
    'MET': [
        ('BB',  'BB',  0.0,  ['N', 'CA', 'C', 'O']),
        ('C',   'SC1', 0.0,  ['CB', 'CG', 'SD', 'CE']),
    ],
    'CYS': [
        ('BB',  'BB',  0.0,  ['N', 'CA', 'C', 'O']),
        ('TC',  'SC1', 0.0,  ['CB', 'SG']),                 # TC4v in Martini 3
    ],
    'SER': [
        ('BB',  'BB',  0.0,  ['N', 'CA', 'C', 'O']),
        ('N',   'SC1', 0.0,  ['CB', 'OG']),
    ],
    'THR': [
        ('BB',  'BB',  0.0,  ['N', 'CA', 'C', 'O']),
        ('N',   'SC1', 0.0,  ['CB', 'OG1', 'CG2']),
    ],
    'ASN': [
        ('BB',  'BB',  0.0,  ['N', 'CA', 'C', 'O']),
        ('N',   'SC1', 0.0,  ['CB', 'CG', 'OD1', 'ND2']),
    ],
    'GLN': [
        ('BB',  'BB',  0.0,  ['N', 'CA', 'C', 'O']),
        ('N',   'SC1', 0.0,  ['CB', 'CG', 'CD', 'OE1', 'NE2']),
    ],
    'ASP': [
        ('BB',  'BB',  0.0,  ['N', 'CA', 'C', 'O']),
        ('Q',   'SC1', -1.0, ['CB', 'CG', 'OD1', 'OD2']),
    ],
    'GLU': [
        ('BB',  'BB',  0.0,  ['N', 'CA', 'C', 'O']),
        ('Q',   'SC1', -1.0, ['CB', 'CG', 'CD', 'OE1', 'OE2']),
    ],
    'LYS': [
        ('BB',  'BB',  0.0,  ['N', 'CA', 'C', 'O']),
        ('SC',  'SC1', 0.0,  ['CB', 'CG', 'CD']),           # SC3 aliphatic linker in Martini 3
        ('Q',   'SC2', +1.0, ['CE', 'NZ']),
    ],
    'ARG': [
        ('BB',  'BB',  0.0,  ['N', 'CA', 'C', 'O']),
        ('N',   'SC1', 0.0,  ['CB', 'CG', 'CD']),
        ('Q',   'SC2', +1.0, ['NE', 'CZ', 'NH1', 'NH2']),
    ],
    'HIS': [
        ('BB',  'BB',  0.0,  ['N', 'CA', 'C', 'O']),
        ('TC',  'SC1', 0.0,  ['CB', 'CG']),
        ('TC',  'SC2', +0.5, ['ND1', 'CD2', 'CE1', 'NE2']),
    ],
    'PHE': [
        ('BB',  'BB',  0.0,  ['N', 'CA', 'C', 'O']),
        ('SC',  'SC1', 0.0,  ['CB', 'CG', 'CD1', 'CD2']),   # SC4 ring bead in Martini 3
        ('SC',  'SC2', 0.0,  ['CE1', 'CE2', 'CZ']),          # SC4 ring bead in Martini 3
    ],
    'TYR': [
        ('BB',  'BB',  0.0,  ['N', 'CA', 'C', 'O']),
        ('SC',  'SC1', 0.0,  ['CB', 'CG', 'CD1', 'CD2']),   # SC4 ring bead in Martini 3
        ('SC',  'SC2', 0.0,  ['CE1', 'CE2', 'CZ', 'OH']),   # SC4 ring bead in Martini 3
    ],
    'TRP': [
        ('BB',  'BB',  0.0,  ['N', 'CA', 'C', 'O']),
        ('SC',  'SC1', 0.0,  ['CB', 'CG', 'CD1', 'NE1']),   # SC4 indole in Martini 3
        ('SC',  'SC2', 0.0,  ['CD2', 'CE2', 'CZ2', 'CH2']), # SC4 indole in Martini 3
        ('TC',  'SC3', 0.0,  ['CE3', 'CZ3']),                # TC4 small ring in Martini 3
    ],
}

# Integer index for each bead type; used to index into the pair tables below.
# ION_NA and ION_CL both map to ION (same TQ5 type in Martini 3).
MARTINI_BEAD_TYPE_INDEX: dict[str, int] = {
    'BB': 0, 'C': 1, 'SC': 2, 'N': 3, 'Q': 4, 'TC': 5, 'W': 6, 'ION': 7,
    'ION_NA': 7, 'ION_CL': 7,
}

# Explicit pairwise LJ parameters from martini_v3.0.0.itp [nonbond_params].
# Representative ITP types: BB=P2, C=C3, SC=SC3, N=N2, Q=Q4, TC=TC3, W=W, ION=TQ5.
# Row/column order matches MARTINI_BEAD_TYPE_INDEX: BB, C, SC, N, Q, TC, W, ION.
# Stored as float64 numpy arrays; symmetric (shape 8×8).
_M3_SIGMA = [
    #   BB      C       SC      N       Q       TC      W       ION
    [0.470, 0.470, 0.430, 0.470, 0.470, 0.398, 0.465, 0.395],  # BB
    [0.470, 0.470, 0.430, 0.470, 0.470, 0.395, 0.465, 0.465],  # C
    [0.430, 0.430, 0.410, 0.430, 0.430, 0.365, 0.425, 0.484],  # SC
    [0.470, 0.470, 0.430, 0.470, 0.470, 0.395, 0.465, 0.395],  # N
    [0.470, 0.470, 0.430, 0.470, 0.470, 0.401, 0.465, 0.405],  # Q
    [0.398, 0.395, 0.365, 0.395, 0.401, 0.340, 0.393, 0.505],  # TC
    [0.465, 0.465, 0.425, 0.465, 0.465, 0.393, 0.470, 0.385],  # W
    [0.395, 0.465, 0.484, 0.395, 0.405, 0.505, 0.385, 0.354],  # ION
]
_M3_EPS = [
    #    BB      C       SC      N       Q       TC      W        ION
    [ 4.060, 2.790, 2.160, 3.390, 5.148, 1.450,  4.330,  9.642],  # BB
    [ 2.790, 3.390, 2.920, 3.240, 2.787, 2.310,  2.420,  2.946],  # C
    [ 2.160, 2.920, 2.350, 2.770, 1.961, 1.910,  1.800,  2.506],  # SC
    [ 3.390, 3.240, 2.770, 3.520, 4.477, 2.110,  3.290,  8.247],  # N
    [ 5.148, 2.787, 1.961, 4.477, 5.950, 1.168,  5.960,  6.360],  # Q
    [ 1.450, 2.310, 1.910, 2.110, 1.168, 1.510,  1.120,  1.818],  # TC
    [ 4.330, 2.420, 1.800, 3.290, 5.960, 1.120,  4.650, 11.460],  # W
    [ 9.642, 2.946, 2.506, 8.247, 6.360, 1.818, 11.460,  1.180],  # ION
]
MARTINI_PAIR_SIGMA_NM  = np.array(_M3_SIGMA, dtype=np.float64)
MARTINI_PAIR_EPS_KJ    = np.array(_M3_EPS,   dtype=np.float64)

# Secondary-structure-dependent backbone BB–BB–BB angles (kJ/(mol·rad²), radians).
# Coil = default; helix (H/G/I) and sheet (E/B) from Martini 3 protein FF.
MARTINI_BB_ANGLE_HELIX = math.radians(96.0)
MARTINI_BB_ANGLE_SHEET = math.radians(134.0)