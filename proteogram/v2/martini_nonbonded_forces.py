"""Martini 3-inspired coarse-grained non-bonded force model for proteogram calculation.

Multi-bead-per-residue CG model where each amino acid is represented by 1–4
beads depending on residue type. Inspired by the Martini 3 force field
(Souza et al., Nat. Methods 2021) but simplified for the proteogram pipeline:
explicit CG water (W beads, ~4 H₂O each), Na⁺/Cl⁻ ions for charge
neutralization and 0.15 M physiological salt, no dihedral terms.

Residue representation
----------------------
Each residue contributes one backbone (BB) bead at the N/CA/C/O centroid plus
0–3 sidechain beads placed at sidechain atom centroids. Missing atoms fall back
to CA automatically (common in sidechain-stripped or coarse PDBs).

  GLY                                          1 bead  (BB only)
  ALA/VAL/LEU/ILE/PRO/MET/CYS/
  SER/THR/ASN/GLN/ASP/GLU                      2 beads (BB + SC1)
  LYS/ARG/HIS/PHE/TYR                          3 beads (BB + SC1 + SC2)
  TRP                                          4 beads (BB + SC1 + SC2 + SC3)

Bead types and LJ parameters (Martini 3 ITP values, Souza et al. 2021)
-----------------------------------------------------------------------
  BB  — backbone (P2 coil):     σ=0.47 nm, ε=4.06 kJ/mol
  C   — apolar regular (C1–C5): σ=0.47 nm, ε=3.39 kJ/mol  (MET)
  SC  — apolar small (SC1–SC4): σ=0.41 nm, ε=2.35 kJ/mol  (VAL/LEU/ILE/PRO/LYS-SC1/PHE/TYR/TRP rings)
  N   — polar (N1–N2):          σ=0.47 nm, ε=3.52 kJ/mol  (SER/THR/ASN/GLN/ARG-SC1)
  Q   — charged (Q4):           σ=0.47 nm, ε=5.95 kJ/mol  (ASP/GLU q=−1; LYS/ARG q=+1)
  TC  — tiny apolar (TC1–TC5):  σ=0.34 nm, ε=1.51 kJ/mol  (ALA/CYS sidechains; HIS/TRP small rings)
  ION — Na⁺/Cl⁻ (TQ5):         σ=0.354 nm, ε=1.18 kJ/mol

Bonded interactions
-------------------
  HarmonicBondForce   BB–BB inter-residue (k=3800, r0=0.35 nm)
                      BB–SC intra-residue (k=3800, r0=0.27 nm)
                      SC–SC intra-residue (k=2500, r0=0.27 nm)
  HarmonicAngleForce  BB–BB–BB backbone angle (k=40 kJ/(mol·rad²)),
                      θ₀ per-residue from DSSP: coil=127°, helix=96°, sheet=134°
                      (falls back to coil for all if DSSP binary unavailable)

Non-bonded interactions
-----------------------
  CustomNonbondedForce  LJ-12-6 with explicit pairwise σ/ε from martini_v3.0.0.itp
                        via Discrete2DFunction (type-index lookup); replaces Lorentz-
                        Berthelot combining rules, which deviate by up to 9 kJ/mol
                        for key pairs (BB–ION, SC–Q, TC–Q).
  CustomNonbondedForce  Coulomb (1/r) for charged beads
  Exclusions            all intra-residue pairs + 1-2/1-3 backbone BB pairs

Output format
-------------
Returns 5 NxN matrices (N = number of residues, upper triangle populated):
  [vdw_att, vdw_rep, es_att, es_rep, dist_avg]

B×B bead-level energies are aggregated to N×N by summing over all bead pairs
belonging to each residue pair. The distance matrix uses BB–BB bead distances.
Format is identical to NonBondedForceModel and CGNonBondedForceModel so
ProteogramV2 can swap models without changes to the downstream pipeline.

Typical speedup vs atomistic: 10–30× (no explicit solvent, 10× larger timestep,
fewer degrees of freedom per residue than all-atom).
"""

import gc
import io
import linecache
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from openmm.app import (
    PDBFile, Topology, Simulation, Element, StateDataReporter,
)
from openmm import (
    LangevinMiddleIntegrator, MonteCarloBarostat, Platform, System,
    HarmonicBondForce, HarmonicAngleForce, CustomNonbondedForce, Vec3,
    Discrete2DFunction,
)
from openmm.unit import (
    kelvin, picosecond, femtoseconds, nanometer,
    kilojoules_per_mole, dalton, radian, bar,
)
from Bio.PDB.PDBParser import PDBParser

from ..common.constants import (
    RESIDUE_LIST,
    MODIFIED_RESIDUES_TO_STANDARD,
    MARTINI_COULOMB_K         as _COULOMB_K,
    MARTINI_SOLVENT_CUTOFF_NM as _SOLVENT_CUTOFF_NM,
    MARTINI_RF_K              as _RF_K,
    MARTINI_RF_C              as _RF_C,
    MARTINI_SOLVENT_BUFFER_NM as _SOLVENT_BUFFER_NM,
    MARTINI_SALT_CONC_M       as _SALT_CONC_M,
    MARTINI_BEAD_TYPE_PARAMS  as _BEAD_TYPE_PARAMS,
    MARTINI_BEAD_TYPE_INDEX   as _BEAD_TYPE_INDEX,
    MARTINI_PAIR_SIGMA_NM     as _PAIR_SIGMA,
    MARTINI_PAIR_EPS_KJ       as _PAIR_EPS,
    MARTINI_BB_BB_K           as _BB_BB_K,
    MARTINI_BB_BB_R0          as _BB_BB_R0,
    MARTINI_BB_SC_K           as _BB_SC_K,
    MARTINI_BB_SC_R0          as _BB_SC_R0,
    MARTINI_SC_SC_K           as _SC_SC_K,
    MARTINI_SC_SC_R0          as _SC_SC_R0,
    MARTINI_BB_ANGLE_K        as _BB_ANGLE_K,
    MARTINI_BB_ANGLE_THETA    as _BB_ANGLE_THETA,
    MARTINI_BB_ANGLE_HELIX    as _BB_ANGLE_HELIX,
    MARTINI_BB_ANGLE_SHEET    as _BB_ANGLE_SHEET,
    MARTINI_BEAD_MASS_DA      as _BEAD_MASS_DA,
    MARTINI_WATER_MASS        as _WATER_MASS,
    MARTINI_ION_MASS_NA       as _ION_MASS_NA,
    MARTINI_ION_MASS_CL       as _ION_MASS_CL,
    MARTINI_RESIDUE_BEADS     as _RESIDUE_BEADS,
)


class MartiniNonBondedForceModel:
    """Martini 3-inspired multi-bead CG MD model for residue-residue interaction maps.

    Each residue is represented by 1–4 beads depending on sidechain complexity.
    Bead positions are computed as centroids of the corresponding heavy atoms in
    the PDB (backbone atoms for BB; sidechain atom subsets for SC beads).

    Bead-level B×B energies are aggregated to residue-level N×N matrices so
    the output format matches NonBondedForceModel and CGNonBondedForceModel.

    Attributes:
        pdb_path (str): Path to input PDB file.
        temperature (Quantity): Simulation temperature.
        timestep (Quantity): Integration timestep (default 10 fs).
        use_gpu (bool): Use CUDA platform if True.
        topology: OpenMM Topology.
        positions: Bead positions (list of Vec3).
        system (System): OpenMM System.
        simulation (Simulation): OpenMM Simulation.
    """

    DEFAULT_TEMPERATURE        = 310.15  # K (37 °C, physiological)
    DEFAULT_FRICTION           = 1.0     # 1/ps
    DEFAULT_TIMESTEP            = 10.0    # fs — equilibration (conservative while system settles)
    DEFAULT_PRODUCTION_TIMESTEP = 20.0    # fs — production (standard Martini 3 W-model timestep)
    DEFAULT_NVT_STEPS           = 25000   # 250 ps at 10 fs/step
    DEFAULT_PRODUCTION_STEPS    = 250000  # 5 ns at 20 fs/step
    DEFAULT_REPORTING_INTERVAL = 2500
    DEFAULT_ENERGY_INTERVAL    = 5000

    def __init__(
        self,
        pdb_path: str,
        output_dir: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        timestep: float = DEFAULT_TIMESTEP,
        use_gpu: bool = False,
        **kwargs,
    ):
        self.pdb_path    = pdb_path
        self.output_dir  = Path(output_dir) if output_dir else Path(pdb_path).parent
        self.temperature = temperature * kelvin
        self.timestep    = timestep * femtoseconds
        self.use_gpu     = use_gpu
        self.debug       = False

        self.topology   = None
        self.positions  = None
        self.system     = None
        self.simulation = None

        # Per-bead parameter arrays — populated by setup_system()
        self._residue_names:    list[str]           = []
        self._n_protein_beads:  int                 = 0     # B_prot; solvent beads follow
        self._bead_to_residue:  np.ndarray | None   = None  # (B_prot,) residue index per bead
        self._bb_bead_indices:  np.ndarray | None   = None  # (N,) BB bead index per residue
        self._bead_type_idx:    np.ndarray | None   = None  # (B_prot,) integer type index
        self._bead_sigmas:      np.ndarray | None   = None  # (B_prot,) LJ sigma (nm)  [kept for energy extraction]
        self._bead_epsilons:    np.ndarray | None   = None  # (B_prot,) LJ epsilon (kJ/mol)
        self._bead_charges:     np.ndarray | None   = None  # (B_prot,) charge (e)
        self._indicator:        np.ndarray | None   = None  # (N, B_prot) aggregation matrix
        self._bead_valid_mask:  np.ndarray | None   = None  # (B_prot, B_prot) non-excluded upper triangle
        self._box_lengths:      np.ndarray | None   = None  # (3,) box edge lengths in nm
        self._openmm_exclusions: list[tuple[int, int]] = []

        self.energy_log = {
            'nvt':        {'time_ps': [], 'energy_kj': [], 'stage': 'NVT Equilibration'},
            'production': {'time_ps': [], 'energy_kj': [], 'stage': 'Production'},
        }

    # ── context manager / cleanup ─────────────────────────────────────────────

    def __del__(self):
        try:
            self.cleanup_all_resources()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.cleanup_all_resources()
        return False

    def cleanup_all_resources(self, final_run: bool = False) -> None:
        try:
            linecache.clearcache()
        except Exception:
            pass
        if self.simulation is not None:
            try:
                self.simulation.reporters.clear()
            except Exception:
                pass
            sim, self.simulation = self.simulation, None
            del sim
        for _ in range(3):
            gc.collect()
        if final_run:
            self.system   = None
            self.topology = None
            self.positions = None
        if self.use_gpu:
            self._clear_cuda_cache()

    def _clear_cuda_cache(self) -> None:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # ── secondary-structure backbone angles ───────────────────────────────────

    @staticmethod
    def _get_dssp_angles(model, pdb_path: str, residue_seq_info: list) -> list[float]:
        """Return per-residue BB–BB–BB equilibrium angle (radians) from DSSP.

        Falls back to the generic coil angle for all residues if DSSP is
        unavailable (binary not installed or Bio.PDB.DSSP import fails).

        DSSP code mapping (Martini 3 protein FF):
          H/G/I  → helix  96°
          E/B    → sheet 134°
          everything else → coil 127°
        """
        default = [_BB_ANGLE_THETA] * len(residue_seq_info)
        try:
            from Bio.PDB.DSSP import DSSP
            dssp = DSSP(model, pdb_path)
            angles = []
            for chain_id, seq_num in residue_seq_info:
                key = (chain_id, (' ', seq_num, ' '))
                if key in dssp:
                    ss = dssp[key][2]
                    if ss in ('H', 'G', 'I'):
                        angles.append(_BB_ANGLE_HELIX)
                    elif ss in ('E', 'B'):
                        angles.append(_BB_ANGLE_SHEET)
                    else:
                        angles.append(_BB_ANGLE_THETA)
                else:
                    angles.append(_BB_ANGLE_THETA)
            return angles
        except Exception:
            return default

    # ── bead position helper ──────────────────────────────────────────────────

    @staticmethod
    def _compute_bead_position(residue, atom_names: list[str]) -> np.ndarray:
        """Return centroid (Å) of atom_names in residue, falling back to CA/N/C."""
        coords = [residue[name].get_vector().get_array()
                  for name in atom_names if name in residue]
        if coords:
            return np.mean(coords, axis=0)
        for fallback in ('CA', 'N', 'C'):
            if fallback in residue:
                return residue[fallback].get_vector().get_array()
        raise KeyError(
            f"Residue {residue.resname} {residue.id} has no usable atoms "
            f"(tried {atom_names} + CA/N/C fallbacks)"
        )

    # ── system setup ──────────────────────────────────────────────────────────

    def setup_system(self) -> None:
        """Build multi-bead Martini topology and force field from the PDB file.

        Steps:
          1. Extract residue heavy-atom positions via Biopython; compute bead
             centroids for each bead definition in _RESIDUE_BEADS.
          2. Build protein-only numpy parameter arrays (sigma, epsilon, charge),
             the (N, B_prot) indicator matrix, and the 1-2/1-3 exclusion mask.
          3. Solvate: place W beads on a cubic grid (_build_solvent_box), remove
             clashes with protein beads.
          4. Add ions: neutralize net protein charge and add 0.15 M NaCl
             (_add_ions), replacing randomly selected water beads.
          5. Build OpenMM Topology (protein chain P, solvent chain S, ion chain I)
             with periodic box vectors set.
          6. Build OpenMM System — protein particles, water particles, ion particles;
             HarmonicBondForce and HarmonicAngleForce for protein bonds/angles;
             LJ-12-6 and reaction-field Coulomb as CutoffPeriodic CustomNonbondedForces
             covering all particles; MonteCarloBarostat at 1 bar.
        """
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", self.pdb_path)
        model_struct = structure[0]

        # ── Step 1: collect protein beads from PDB residues ───────────────────
        beads_data: list[dict] = []
        residue_names: list[str] = []
        residue_bead_ranges: list[tuple[int, int]] = []
        bb_bead_indices: list[int] = []
        residue_seq_info: list[tuple[str, int]] = []  # (chain_id, seq_num) for gap detection

        for chain in model_struct.get_chains():
            for res in chain:
                raw = res.resname.strip()
                resname = MODIFIED_RESIDUES_TO_STANDARD.get(raw, raw)
                bead_defs = _RESIDUE_BEADS.get(resname)
                if bead_defs is None:
                    continue  # HETATM, solvent, unknown residue — skip
                if 'CA' not in res:
                    continue  # no alpha carbon — BB bead position unreliable, skip

                ri = len(residue_names)
                residue_names.append(resname)
                residue_seq_info.append((chain.id, res.id[1]))
                first_bead = len(beads_data)

                for bead_type, bead_label, charge, atom_names in bead_defs:
                    pos_ang = self._compute_bead_position(res, atom_names)
                    beads_data.append({
                        'residue_idx': ri,
                        'bead_type':   bead_type,
                        'bead_label':  bead_label,
                        'charge':      charge,
                        'pos_nm':      pos_ang / 10.0,  # Å → nm
                    })

                bb_bead_indices.append(first_bead)  # BB is always first bead
                residue_bead_ranges.append((first_bead, len(beads_data)))

        if not beads_data:
            raise ValueError(f"No recognised residues found in {self.pdb_path}")

        # ── Gap filling: insert dummy GLY beads at backbone gaps ──────────────
        # When consecutive residues within the same chain are not sequence-
        # adjacent (missing loop in crystal structure), insert dummy GLY BB
        # beads at linearly interpolated positions to restore chain connectivity.
        # Dummy beads: zero charge, BB LJ parameters → minimal energy contribution.
        filled_beads_data:        list[dict]         = []
        filled_residue_names:     list[str]           = []
        filled_residue_bead_ranges: list[tuple[int, int]] = []
        filled_bb_bead_indices:   list[int]           = []
        filled_residue_seq_info:  list[tuple[str, int]] = []
        filled_is_dummy:          list[bool]          = []
        n_dummy = 0

        for ri in range(len(residue_names)):
            if ri > 0:
                cid_prev, seq_prev = residue_seq_info[ri - 1]
                cid_curr, seq_curr = residue_seq_info[ri]
                if cid_prev == cid_curr and seq_curr > seq_prev + 1:
                    n_missing = seq_curr - seq_prev - 1
                    pos_a = beads_data[residue_bead_ranges[ri - 1][0]]['pos_nm']
                    pos_b = beads_data[residue_bead_ranges[ri][0]]['pos_nm']
                    for k in range(1, n_missing + 1):
                        t = k / (n_missing + 1)
                        new_ri = len(filled_residue_names)
                        first_bead = len(filled_beads_data)
                        filled_beads_data.append({
                            'residue_idx': new_ri,
                            'bead_type':   'BB',
                            'bead_label':  'BB',
                            'charge':      0.0,
                            'pos_nm':      pos_a + t * (pos_b - pos_a),
                        })
                        filled_residue_names.append('GLY')
                        filled_residue_seq_info.append((cid_prev, seq_prev + k))
                        filled_bb_bead_indices.append(first_bead)
                        filled_residue_bead_ranges.append((first_bead, first_bead + 1))
                        filled_is_dummy.append(True)
                        n_dummy += 1

            new_ri    = len(filled_residue_names)
            lo, hi    = residue_bead_ranges[ri]
            first_bead = len(filled_beads_data)
            for b in beads_data[lo:hi]:
                b_copy = dict(b)
                b_copy['residue_idx'] = new_ri
                filled_beads_data.append(b_copy)
            filled_residue_names.append(residue_names[ri])
            filled_residue_seq_info.append(residue_seq_info[ri])
            filled_bb_bead_indices.append(first_bead + (bb_bead_indices[ri] - lo))
            filled_residue_bead_ranges.append((first_bead, len(filled_beads_data)))
            filled_is_dummy.append(False)

        if n_dummy:
            print(f"  Inserted {n_dummy} dummy GLY bead(s) to bridge backbone gaps")

        beads_data          = filled_beads_data
        residue_names       = filled_residue_names
        residue_bead_ranges = filled_residue_bead_ranges
        bb_bead_indices     = filled_bb_bead_indices
        residue_seq_info    = filled_residue_seq_info

        N      = len(residue_names)
        B_prot = len(beads_data)

        # ── Step 2: protein-only parameter arrays and exclusion mask ──────────
        self._residue_names     = residue_names
        self._dummy_residue_mask = np.array(filled_is_dummy, dtype=bool)
        self._n_protein_beads   = B_prot
        self._bead_to_residue = np.array([b['residue_idx'] for b in beads_data], dtype=int)
        self._bb_bead_indices = np.array(bb_bead_indices, dtype=int)
        self._bead_type_idx   = np.array([_BEAD_TYPE_INDEX[b['bead_type']] for b in beads_data], dtype=np.int32)
        self._bead_sigmas     = np.array([_BEAD_TYPE_PARAMS[b['bead_type']][0] for b in beads_data])
        self._bead_epsilons   = np.array([_BEAD_TYPE_PARAMS[b['bead_type']][1] for b in beads_data])
        self._bead_charges    = np.array([b['charge'] for b in beads_data])

        # indicator[i, b] = 1 iff bead b belongs to residue i
        self._indicator = np.zeros((N, B_prot))
        for b_idx, b in enumerate(beads_data):
            self._indicator[b['residue_idx'], b_idx] = 1.0

        # Bond list (protein only)
        bond_list: list[tuple[int, int]] = []
        for ri in range(N):
            lo_r, hi_r = residue_bead_ranges[ri]
            bead_idxs = list(range(lo_r, hi_r))
            for k in range(len(bead_idxs) - 1):
                bond_list.append((bead_idxs[k], bead_idxs[k + 1]))
        # BB–BB inter-residue bonds: same chain only. Intra-chain gaps are
        # already bridged by dummy GLY above; inter-chain breaks are intentional.
        for ri in range(N - 1):
            if residue_seq_info[ri][0] == residue_seq_info[ri + 1][0]:
                bond_list.append((bb_bead_indices[ri], bb_bead_indices[ri + 1]))

        # 1-2 / 1-3 exclusion pairs
        neighbors: dict[int, set[int]] = defaultdict(set)
        for i, j in bond_list:
            neighbors[i].add(j)
            neighbors[j].add(i)

        exclude_set: set[tuple[int, int]] = set()
        for b in range(B_prot):
            for n1 in list(neighbors[b]):
                exclude_set.add((min(b, n1), max(b, n1)))
                for n2 in list(neighbors[n1]):
                    if n2 != b:
                        exclude_set.add((min(b, n2), max(b, n2)))

        excl_matrix = np.zeros((B_prot, B_prot), dtype=bool)
        for i, j in exclude_set:
            excl_matrix[i, j] = True
            excl_matrix[j, i] = True
        np.fill_diagonal(excl_matrix, True)
        self._bead_valid_mask    = np.triu(~excl_matrix, k=1)
        self._openmm_exclusions  = list(exclude_set)

        # ── Step 3: solvate ───────────────────────────────────────────────────
        protein_pos_nm = np.array([b['pos_nm'] for b in beads_data])  # (B_prot, 3)
        lo             = protein_pos_nm.min(axis=0) - _SOLVENT_BUFFER_NM
        box_lengths    = protein_pos_nm.max(axis=0) + _SOLVENT_BUFFER_NM - lo
        protein_pos_box = protein_pos_nm - lo          # translate to box frame [0, L)
        self._box_lengths = box_lengths

        print(f"  Solvation box: {box_lengths[0]:.2f} × {box_lengths[1]:.2f} × "
              f"{box_lengths[2]:.2f} nm  ({np.prod(box_lengths):.1f} nm³)")
        water_pos = self._build_solvent_box(protein_pos_box, box_lengths)

        # ── Step 4: add ions ──────────────────────────────────────────────────
        net_charge = float(self._bead_charges.sum())
        water_pos, na_pos, cl_pos = self._add_ions(water_pos, net_charge, box_lengths)

        n_water = len(water_pos)
        n_na    = len(na_pos)
        n_cl    = len(cl_pos)
        B_total = B_prot + n_water + n_na + n_cl

        # ── Step 5: OpenMM Topology ───────────────────────────────────────────
        topology = Topology()
        carbon   = Element.getBySymbol('C')
        openmm_atoms: list = []

        # Protein chain
        prot_chain = topology.addChain('P')
        for ri in range(N):
            res = topology.addResidue(residue_names[ri], prot_chain)
            lo_r, hi_r = residue_bead_ranges[ri]
            for b_idx in range(lo_r, hi_r):
                openmm_atoms.append(
                    topology.addAtom(beads_data[b_idx]['bead_label'], carbon, res))
        for i, j in bond_list:
            topology.addBond(openmm_atoms[i], openmm_atoms[j])

        # Solvent chain (one residue per W bead)
        if n_water > 0:
            sol_chain = topology.addChain('S')
            for _ in range(n_water):
                res = topology.addResidue('SOL', sol_chain)
                topology.addAtom('W', carbon, res)

        # Ion chain
        if n_na > 0 or n_cl > 0:
            ion_chain = topology.addChain('I')
            for _ in range(n_na):
                res = topology.addResidue('NA', ion_chain)
                topology.addAtom('NA', carbon, res)
            for _ in range(n_cl):
                res = topology.addResidue('CL', ion_chain)
                topology.addAtom('CL', carbon, res)

        bvx = Vec3(box_lengths[0], 0, 0) * nanometer
        bvy = Vec3(0, box_lengths[1], 0) * nanometer
        bvz = Vec3(0, 0, box_lengths[2]) * nanometer
        topology.setPeriodicBoxVectors((bvx, bvy, bvz))
        self.topology = topology

        # All positions in box frame
        pieces = [protein_pos_box, water_pos]
        if n_na > 0:
            pieces.append(na_pos)
        if n_cl > 0:
            pieces.append(cl_pos)
        all_pos_nm = np.vstack(pieces)
        self.positions = [Vec3(*p) * nanometer for p in all_pos_nm]

        # ── Step 6: OpenMM System ─────────────────────────────────────────────
        system = System()
        system.setDefaultPeriodicBoxVectors(bvx, bvy, bvz)

        for _ in range(B_prot):
            system.addParticle(_BEAD_MASS_DA * dalton)
        for _ in range(n_water):
            system.addParticle(_WATER_MASS * dalton)
        for _ in range(n_na):
            system.addParticle(_ION_MASS_NA * dalton)
        for _ in range(n_cl):
            system.addParticle(_ION_MASS_CL * dalton)

        # Harmonic bonds (protein beads only)
        bond_force = HarmonicBondForce()
        for i, j in bond_list:
            li = beads_data[i]['bead_label']
            lj = beads_data[j]['bead_label']
            is_bb_bb = (li == 'BB' and lj == 'BB')
            is_bb_sc = (li == 'BB') != (lj == 'BB')
            if is_bb_bb:
                k_b, r0 = _BB_BB_K, _BB_BB_R0
            elif is_bb_sc:
                k_b, r0 = _BB_SC_K, _BB_SC_R0
            else:
                k_b, r0 = _SC_SC_K, _SC_SC_R0
            bond_force.addBond(i, j,
                               r0 * nanometer,
                               k_b * kilojoules_per_mole / nanometer**2)
        system.addForce(bond_force)

        # Backbone angles: BB_{i-1}–BB_i–BB_{i+1}, same chain only.
        # Per-residue equilibrium angles from DSSP (helix/sheet/coil); falls
        # back to coil 127° for all if DSSP binary is unavailable.
        bb_angles = self._get_dssp_angles(model_struct, self.pdb_path, residue_seq_info)
        angle_force = HarmonicAngleForce()
        for ri in range(1, N - 1):
            if not (residue_seq_info[ri - 1][0]
                    == residue_seq_info[ri][0]
                    == residue_seq_info[ri + 1][0]):
                continue
            angle_force.addAngle(
                bb_bead_indices[ri - 1],
                bb_bead_indices[ri],
                bb_bead_indices[ri + 1],
                bb_angles[ri] * radian,
                _BB_ANGLE_K * kilojoules_per_mole / radian**2,
            )
        system.addForce(angle_force)

        # LJ-12-6, CutoffPeriodic with explicit pairwise σ/ε from Martini 3 ITP.
        # Each particle carries an integer type index; σ_ij and ε_ij are looked
        # up from Discrete2DFunction tables rather than using Lorentz-Berthelot.
        # OpenMM stores Discrete2DFunction as f(x,y) = values[x + xsize*y].
        n_types = _PAIR_SIGMA.shape[0]
        sig_flat = _PAIR_SIGMA.T.flatten().tolist()   # column-major → f(x,y)=values[x+n*y]
        eps_flat = _PAIR_EPS.T.flatten().tolist()
        lj = CustomNonbondedForce(
            "4*eps_tab(itype1,itype2)*((sig_tab(itype1,itype2)/r)^12"
            " - (sig_tab(itype1,itype2)/r)^6);"
        )
        lj.addTabulatedFunction('sig_tab', Discrete2DFunction(n_types, n_types, sig_flat))
        lj.addTabulatedFunction('eps_tab', Discrete2DFunction(n_types, n_types, eps_flat))
        lj.addPerParticleParameter('itype')
        lj.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
        lj.setCutoffDistance(_SOLVENT_CUTOFF_NM * nanometer)

        # Coulomb with reaction-field correction, CutoffPeriodic
        # E = K * q1*q2 * (1/r + k_rf*r² − c_rf)   [kJ/mol, r in nm]
        es = CustomNonbondedForce(
            f"{_COULOMB_K:.8f}*charge1*charge2*(1/r + {_RF_K:.8f}*r^2 - {_RF_C:.8f});"
        )
        es.addPerParticleParameter('charge')
        es.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
        es.setCutoffDistance(_SOLVENT_CUTOFF_NM * nanometer)

        # Add all particles to force objects
        w_idx  = float(_BEAD_TYPE_INDEX['W'])
        ion_idx = float(_BEAD_TYPE_INDEX['ION'])
        for b_idx in range(B_prot):
            lj.addParticle([float(self._bead_type_idx[b_idx])])
            es.addParticle([self._bead_charges[b_idx]])
        for _ in range(n_water):
            lj.addParticle([w_idx])
            es.addParticle([0.0])
        for _ in range(n_na):
            lj.addParticle([ion_idx])
            es.addParticle([+1.0])
        for _ in range(n_cl):
            lj.addParticle([ion_idx])
            es.addParticle([-1.0])

        # Protein-only 1-2/1-3 exclusions
        for i, j in self._openmm_exclusions:
            lj.addExclusion(i, j)
            es.addExclusion(i, j)

        system.addForce(lj)
        system.addForce(es)

        # MonteCarloBarostat at 1 atm, update every 25 steps
        system.addForce(MonteCarloBarostat(1.0 * bar, self.temperature, 25))

        self.system = system

        charged = int(np.count_nonzero(self._bead_charges))
        print(f"Martini CG system: {N} residues, {B_prot} protein beads "
              f"+ {n_water} W + {n_na} Na⁺ + {n_cl} Cl⁻ = {B_total} total; "
              f"{len(bond_list)} bonds, {charged} charged protein beads")

    # ── solvation helpers ─────────────────────────────────────────────────────

    def _build_solvent_box(
        self,
        protein_pos_box: np.ndarray,
        box_lengths: np.ndarray,
        min_clash_nm: float = 0.53,
    ) -> np.ndarray:
        """Place W beads on a cubic grid and return clash-free positions.

        Grid spacing targets the W–W LJ minimum distance (2^(1/6)·σ_W ≈
        0.527 nm) so that adjacent water beads start at zero force — placing
        them at σ (0.47 nm) puts them in the repulsive region and causes the
        minimiser to launch beads. Density is ~6.8 W/nm³, slightly below the
        Martini liquid target of ~8.4; NPT equilibration shrinks the box to
        reach the correct density. W beads within min_clash_nm of any protein
        bead are removed. Clash detection is chunked to cap peak memory.

        The grid MUST be commensurate with the periodic box: each axis is
        divided into a whole number of planes (n = round(L / spacing)) so the
        actual per-axis spacing is L/n. Using a raw np.arange(0, L, spacing)
        instead leaves a wrap-around gap of (L mod spacing) between the last
        plane and the box edge; when that remainder is near zero the top sheet
        of water beads sits ~0 nm from the bottom sheet's periodic image,
        giving LJ energies of ~10^24 kJ/mol that the minimiser cannot relieve
        (a whole sheet pinned together by periodicity) and blowing up dynamics.
        The chosen spacing stays within ~1% of 0.527 nm across realistic boxes.

        Args:
            protein_pos_box: Protein bead positions in box-frame nm coords.
            box_lengths: Box edge lengths (nm) in x, y, z.
            min_clash_nm: Minimum allowed protein–water bead distance (nm).
                Default 0.53 nm places water at or beyond the W–protein LJ
                minimum, preventing high-energy repulsive starting forces.

        Returns:
            water_pos: (W, 3) float64 array of W positions in box frame (nm).
        """
        sigma_w  = _BEAD_TYPE_PARAMS['W'][0]              # 0.47 nm
        spacing  = 2 ** (1.0 / 6.0) * sigma_w             # LJ minimum ≈ 0.527 nm
        # Commensurate grid: an integer number of planes per axis so the
        # wrap-around gap equals the in-plane spacing (never a near-zero
        # sliver). endpoint=False excludes the plane at x=L (its own periodic
        # image at x=0 is already placed).
        axes = [
            np.linspace(0.0, box_lengths[d],
                        max(1, int(round(box_lengths[d] / spacing))),
                        endpoint=False)
            for d in range(3)
        ]
        gx, gy, gz = np.meshgrid(*axes, indexing='ij')
        grid = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)  # (G, 3)

        valid = np.ones(len(grid), dtype=bool)
        chunk = 1000
        for start in range(0, len(grid), chunk):
            slc  = grid[start:start + chunk]                              # (c, 3)
            diff = slc[:, np.newaxis, :] - protein_pos_box[np.newaxis, :, :]  # (c, B, 3)
            min_d = np.sqrt((diff ** 2).sum(axis=2)).min(axis=1)          # (c,)
            valid[start:start + chunk] = min_d >= min_clash_nm

        water_pos = grid[valid]
        print(f"  Placed {len(water_pos)} W beads "
              f"({len(grid) - len(water_pos)} removed for clashes with protein)")
        return water_pos

    def _add_ions(
        self,
        water_pos: np.ndarray,
        net_charge: float,
        box_lengths: np.ndarray,
        salt_conc_M: float = _SALT_CONC_M,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Replace water beads with Na⁺/Cl⁻ for neutralization and 0.15 M NaCl.

        Neutralization ions are placed first to bring the system to zero net
        charge, then NaCl pairs are added to reach the target concentration.
        Ion positions are chosen by randomly sampling water bead slots (seed=42
        for reproducibility). His residues with charge +0.5 are rounded to the
        nearest integer when determining neutralization count.

        Args:
            water_pos: W bead positions in box frame (nm).
            net_charge: Protein net charge (e); rounded to nearest integer.
            box_lengths: Box edge lengths (nm), used to compute box volume.
            salt_conc_M: Target NaCl concentration in mol/L.

        Returns:
            (final_water, na_pos, cl_pos) — position arrays, dtype float64.
        """
        rng     = np.random.default_rng(42)
        n_water = len(water_pos)
        order   = rng.permutation(n_water)
        used    = 0

        na_indices: list[int] = []
        cl_indices: list[int] = []

        # Neutralization: positive protein → add Cl⁻; negative → add Na⁺
        net_int = round(net_charge)
        n_neutralize_na = max(0, -net_int)
        n_neutralize_cl = max(0,  net_int)
        for _ in range(n_neutralize_na):
            na_indices.append(int(order[used])); used += 1
        for _ in range(n_neutralize_cl):
            cl_indices.append(int(order[used])); used += 1

        # Additional NaCl for physiological salt concentration
        vol_l    = float(np.prod(box_lengths)) * 1e-24   # nm³ → litres
        n_pairs  = int(round(salt_conc_M * vol_l * 6.02214076e23))
        for _ in range(n_pairs):
            if used < n_water:
                na_indices.append(int(order[used])); used += 1
            if used < n_water:
                cl_indices.append(int(order[used])); used += 1

        ion_set    = set(na_indices + cl_indices)
        keep       = np.array([i for i in range(n_water) if i not in ion_set])
        final_water = water_pos[keep] if len(keep) else np.empty((0, 3))
        na_pos      = water_pos[na_indices] if na_indices else np.empty((0, 3))
        cl_pos      = water_pos[cl_indices] if cl_indices else np.empty((0, 3))

        system_charge = net_int + len(na_indices) - len(cl_indices)
        print(f"  Net protein charge = {net_int:+d}  →  "
              f"neutralization: {n_neutralize_na} Na⁺, {n_neutralize_cl} Cl⁻")
        print(f"  NaCl ({salt_conc_M} M, {n_pairs} pairs): "
              f"{len(na_indices)} Na⁺ total, {len(cl_indices)} Cl⁻ total  "
              f"[system net charge = {system_charge:+d}]")
        return final_water, na_pos, cl_pos

    # ── simulation helpers ────────────────────────────────────────────────────

    def _get_barostat(self) -> Optional[MonteCarloBarostat]:
        """Return the MonteCarloBarostat from self.system, or None."""
        if self.system is None:
            return None
        for i in range(self.system.getNumForces()):
            force = self.system.getForce(i)
            if isinstance(force, MonteCarloBarostat):
                return force
        return None

    def _get_platform(self) -> Platform:
        if self.use_gpu:
            try:
                return Platform.getPlatformByName('CUDA')
            except Exception:
                print("CUDA not available, falling back to CPU")
        return Platform.getPlatformByName('CPU')

    def _create_new_simulation(self) -> None:
        self.cleanup_all_resources(final_run=False)
        integrator = LangevinMiddleIntegrator(
            self.temperature,
            self.DEFAULT_FRICTION / picosecond,
            self.timestep,
        )
        platform = self._get_platform()
        props = {'CudaPrecision': 'mixed'} if platform.getName() == 'CUDA' else {}
        self.simulation = Simulation(
            self.topology, self.system, integrator, platform, props
        )
        self.simulation.context.setPositions(self.positions)
        bvs = self.topology.getPeriodicBoxVectors()
        if bvs is not None:
            self.simulation.context.setPeriodicBoxVectors(*bvs)

    def _resolve_bead_clashes(self, min_sep_nm: float = 0.20) -> int:
        """Push apart protein bead pairs closer than min_sep_nm.

        Prevents the L-BFGS minimizer from encountering LJ singularities caused
        by sidechain beads falling back to CA (landing on top of the BB bead),
        by dummy GLY beads interpolated into a crowded region, or by unusual
        crystallographic geometry. Only protein beads are checked — solvent is
        on a regular grid and cannot produce coincident pairs.

        Returns the number of pairs resolved.
        """
        B_prot = self._n_protein_beads
        if B_prot == 0:
            return 0

        full_state = self.simulation.context.getState(getPositions=True)
        full_pos = np.asarray(
            full_state.getPositions(asNumpy=True).value_in_unit(nanometer),
            dtype=np.float64,
        )
        del full_state

        pos = full_pos[:B_prot].copy()
        rng = np.random.default_rng(99)
        n_fixed = 0

        for i in range(B_prot):
            diffs = pos[i + 1:] - pos[i]
            dists = np.linalg.norm(diffs, axis=1)
            for ci in np.where(dists < min_sep_nm)[0]:
                j = i + 1 + ci
                dist = dists[ci]
                if dist > 1e-10:
                    direction = diffs[ci] / dist
                else:
                    direction = rng.uniform(-1, 1, 3)
                    direction /= np.linalg.norm(direction)
                pos[j] += (min_sep_nm - dist + 0.02) * direction
                n_fixed += 1

        if n_fixed:
            full_pos[:B_prot] = pos
            self.positions = [Vec3(*p) * nanometer for p in full_pos]
            self.simulation.context.setPositions(self.positions)

        return n_fixed

    def minimize_energy(self, max_iterations: int = 2000) -> None:
        """Minimize energy with up to 3 attempts, jittering positions on NaN.

        OpenMM raises an exception (not a return value) when minimization
        produces NaN coordinates, so each attempt is wrapped in try/except.
        After any failure the simulation is recreated (the context is invalid
        after an OpenMM exception) and positions are jittered with increasing
        amplitude before retrying.
        """
        print("Performing energy minimization...")

        # Pre-minimization: resolve overlapping protein beads.
        n_fixed = self._resolve_bead_clashes()
        if n_fixed:
            print(f"  Resolved {n_fixed} close bead pair(s) before minimization")

        # Snapshot positions after clash resolution — this is the restart point.
        init_state = self.simulation.context.getState(getPositions=True)
        init_pos_nm = np.asarray(
            init_state.getPositions(asNumpy=True).value_in_unit(nanometer),
            dtype=np.float64,
        )
        del init_state

        rng = np.random.default_rng(42)
        final_energy_kj = None
        for attempt in range(3):
            failed = False
            try:
                self.simulation.minimizeEnergy(maxIterations=max_iterations)
                state = self.simulation.context.getState(
                    getPositions=True, getEnergy=True, enforcePeriodicBox=True
                )
                pos_nm = np.asarray(
                    state.getPositions(asNumpy=True).value_in_unit(nanometer),
                    dtype=np.float64,
                )
                energy_kj = state.getPotentialEnergy().value_in_unit(
                    kilojoules_per_mole
                )
                del state
                if np.any(np.isnan(pos_nm)):
                    failed = True
                else:
                    self.positions = [Vec3(*p) * nanometer for p in pos_nm]
                    final_energy_kj = energy_kj
            except Exception as exc:
                if 'NaN' not in str(exc) and 'nan' not in str(exc).lower():
                    raise
                failed = True

            if not failed:
                break

            if attempt < 2:
                jitter_amp = 0.05 * (attempt + 1)   # 0.05 nm → 0.10 nm
                print(f"  Minimization attempt {attempt + 1} produced NaN; "
                      f"jittering positions by ±{jitter_amp:.2f} nm and retrying...")
                jittered = init_pos_nm + rng.uniform(
                    -jitter_amp, jitter_amp, init_pos_nm.shape
                )
                self.positions = [Vec3(*p) * nanometer for p in jittered]
                self.cleanup_all_resources(final_run=False)
                self._create_new_simulation()
        else:
            raise RuntimeError(
                f"Energy minimization failed with NaN after 3 attempts: {self.pdb_path}"
            )

        # Sanity guard: a healthy CG system minimises to a strongly *negative*
        # potential energy (order −10 kJ/mol per particle). A large positive
        # value means the minimiser could not relieve the starting geometry —
        # e.g. an entire sheet of water beads overlapping their periodic images
        # (see _build_solvent_box). L-BFGS reports success on such configs (no
        # NaN), so without this check dynamics would silently explode and the
        # resulting proteogram would be garbage. Threshold is far above any
        # legitimate energy yet far below the ~10^15 kJ/mol pathology.
        n_particles = self.system.getNumParticles() if self.system else 1
        energy_threshold_kj = 1.0e4 * max(1, n_particles)   # ~10^4 kJ/mol/particle
        if final_energy_kj is not None and final_energy_kj > energy_threshold_kj:
            print(
                f"  WARNING: post-minimization potential energy is "
                f"{final_energy_kj:.3e} kJ/mol "
                f"({final_energy_kj / n_particles:.3e} kJ/mol/particle) — "
                f"far above the expected negative range. This indicates "
                f"unrelievable overlapping particles (commonly water beads "
                f"clashing across the periodic boundary); dynamics will explode."
            )
            raise RuntimeError(
                f"Energy minimization did not converge to a physical energy for "
                f"{self.pdb_path}: {final_energy_kj:.3e} kJ/mol "
                f"(> {energy_threshold_kj:.3e} kJ/mol threshold). See warning above."
            )

        self.cleanup_all_resources(final_run=False)
        print("Energy minimization complete.")

    # ── NVT equilibration ─────────────────────────────────────────────────────

    def equilibrate_nvt(
        self,
        steps: int = DEFAULT_NVT_STEPS,
        report_interval: int = DEFAULT_REPORTING_INTERVAL,
    ) -> None:
        # Disable barostat so this is a true constant-volume phase.
        # The system is under-dense immediately after solvation; allowing
        # box rescaling before the temperature is established causes the
        # integrator to produce NaN when it encounters the resulting
        # large forces.
        barostat = self._get_barostat()
        if barostat is not None:
            barostat.setFrequency(0)
        print(f"Running NVT equilibration for {steps} steps...")
        self._create_new_simulation()
        self.simulation.context.setVelocitiesToTemperature(self.temperature)
        self.simulation.reporters.append(
            StateDataReporter(sys.stdout, report_interval,
                              step=True, potentialEnergy=True,
                              temperature=True, separator='\t')
        )
        self.simulation.step(steps)
        state = self.simulation.context.getState(
            getPositions=True, enforcePeriodicBox=True
        )
        self.positions = state.getPositions()
        del state
        self.cleanup_all_resources(final_run=False)
        # Re-enable barostat for the NPT and production phases that follow.
        if barostat is not None:
            barostat.setFrequency(25)
        print("NVT equilibration complete.")

    def equilibrate_npt(
        self,
        steps: int = DEFAULT_NVT_STEPS,
        report_interval: int = DEFAULT_REPORTING_INTERVAL,
    ) -> None:
        """NPT equilibration — allow box volume to relax under barostat control.

        The MonteCarloBarostat added during setup_system() is active throughout
        all phases; this step gives the box volume explicit time to converge
        before production data collection begins.

        Args:
            steps: Number of integration steps.
            report_interval: Steps between StateDataReporter outputs.
        """
        print(f"Running NPT equilibration for {steps} steps...")
        self._create_new_simulation()
        self.simulation.context.setVelocitiesToTemperature(self.temperature)
        self.simulation.reporters.append(
            StateDataReporter(sys.stdout, report_interval,
                              step=True, potentialEnergy=True,
                              temperature=True, volume=True, separator='\t')
        )
        self.simulation.step(steps)
        state = self.simulation.context.getState(
            getPositions=True, getParameterDerivatives=False,
            enforcePeriodicBox=True,
        )
        self.positions = state.getPositions()

        # Propagate the NPT-equilibrated box back to topology, system defaults,
        # and _box_lengths. Without this, _create_new_simulation for production
        # resets to the original (over-large) box, placing compressed positions
        # in a corner and creating extreme local density on the first MD step.
        new_bvs = self.simulation.context.getState(
            getPositions=False
        ).getPeriodicBoxVectors()
        self.topology.setPeriodicBoxVectors((new_bvs[0], new_bvs[1], new_bvs[2]))
        self.system.setDefaultPeriodicBoxVectors(*new_bvs)
        self._box_lengths = np.array([
            new_bvs[0][0].value_in_unit(nanometer),
            new_bvs[1][1].value_in_unit(nanometer),
            new_bvs[2][2].value_in_unit(nanometer),
        ])
        del state
        self.cleanup_all_resources(final_run=False)
        print("NPT equilibration complete.")

    # ── production MD ─────────────────────────────────────────────────────────

    def run_production(
        self,
        steps: int = DEFAULT_PRODUCTION_STEPS,
        energy_calc_interval: int = DEFAULT_ENERGY_INTERVAL,
        subtract_solvent: bool = False,  # no-op — no solvent in CG
    ) -> list:
        """Run production MD and accumulate frame-averaged residue-level energies.

        At each energy snapshot bead positions are extracted, pairwise bead
        energies are computed in numpy, aggregated from B×B to N×N via the
        indicator matrix, and accumulated for averaging.

        Args:
            steps: Total production steps.
            energy_calc_interval: Steps between energy-snapshot frames.
            subtract_solvent: No-op; present for API compatibility.

        Returns:
            [vdw_att, vdw_rep, es_att, es_rep, dist_avg] — each NxN ndarray,
            upper-triangle populated, matching NonBondedForceModel convention.
        """
        print(f"Running Martini production MD for {steps} steps...")
        self._create_new_simulation()
        self.simulation.context.setVelocitiesToTemperature(self.temperature)

        report_interval = max(steps // 10, self.DEFAULT_REPORTING_INTERVAL)
        self.simulation.reporters.append(
            StateDataReporter(sys.stdout, report_interval,
                              step=True, potentialEnergy=True,
                              temperature=True, separator='\t')
        )

        N = len(self._residue_names)
        vdw_att_sum = np.zeros((N, N))
        vdw_rep_sum = np.zeros((N, N))
        es_att_sum  = np.zeros((N, N))
        es_rep_sum  = np.zeros((N, N))
        dist_sum    = np.zeros((N, N))
        frame_count = 0

        timestep_ps = self.timestep.value_in_unit(picosecond)

        for step_start in range(0, steps, energy_calc_interval):
            self.simulation.step(energy_calc_interval)
            state = self.simulation.context.getState(
                getPositions=True, getEnergy=True
            )
            pos_nm = np.asarray(
                state.getPositions(asNumpy=True).value_in_unit(nanometer),
                dtype=np.float64,
            )
            current_energy = state.getPotentialEnergy().value_in_unit(
                kilojoules_per_mole
            )
            del state

            if self.debug:
                time_ps = (step_start + energy_calc_interval) * timestep_ps
                self.energy_log['production']['time_ps'].append(time_ps)
                self.energy_log['production']['energy_kj'].append(current_energy)

            vdw_a, vdw_r, es_a, es_r, dist = self._calc_pairwise_martini_energies(pos_nm)
            vdw_att_sum += vdw_a
            vdw_rep_sum += vdw_r
            es_att_sum  += es_a
            es_rep_sum  += es_r
            dist_sum    += dist
            frame_count += 1

            if frame_count % 5 == 0:
                gc.collect()

        avg = 1.0 / frame_count
        # Wrap all bead positions back into the primary periodic box before
        # storing them — without enforcePeriodicBox, coordinates accumulate
        # unbounded drift across image boundaries and cannot be written to PDB.
        state = self.simulation.context.getState(
            getPositions=True, enforcePeriodicBox=True
        )
        self.positions = state.getPositions()
        del state
        print("Martini production MD complete.")

        results = [
            vdw_att_sum * avg, vdw_rep_sum * avg,
            es_att_sum  * avg, es_rep_sum  * avg,
            dist_sum    * avg,
        ]

        # Strip dummy-residue rows/columns so the returned N×N matrices match
        # the real sequence length expected by ProteogramV2.
        if np.any(self._dummy_residue_mask):
            real = ~self._dummy_residue_mask
            results = [m[np.ix_(real, real)] for m in results]

        return results

    # ── pairwise energy calculation ───────────────────────────────────────────

    def _calc_pairwise_martini_energies(
        self, positions_nm: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute pairwise bead energies and aggregate to residue-level N×N.

        Computes all B×B bead-pair interactions using vectorised numpy, then
        collapses to N×N via the indicator matrix:
            E_residue[i, j] = sum of E_bead[b1, b2]
                               for all b1 in residue i, b2 in residue j.

        Only non-excluded upper-triangle pairs are populated (see
        _bead_valid_mask built in setup_system). The BB–BB distance matrix is
        computed separately using only backbone bead positions.

        Args:
            positions_nm: (B, 3) float64 array of all bead positions in nm.

        Returns:
            (vdw_att, vdw_rep, es_att, es_rep, dist_angstrom) — each NxN ndarray,
            upper triangle populated, lower triangle zero.
        """
        N = len(self._residue_names)

        # Slice to protein beads only — solvent beads drive dynamics but are
        # excluded from the residue-level energy maps.
        pos_prot = positions_nm[:self._n_protein_beads]  # (B_prot, 3)

        # ── B_prot×B_prot pairwise distances with minimum image convention ────
        diff = pos_prot[:, np.newaxis, :] - pos_prot[np.newaxis, :, :]
        if self._box_lengths is not None:
            diff -= np.round(diff / self._box_lengths) * self._box_lengths
        dist_nm = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
        del diff

        # Apply the same 1.1 nm cutoff as the OpenMM CutoffPeriodic forces.
        # Without this, the RF term k_rf·r² grows unboundedly beyond the
        # cutoff, producing spuriously large energies for distant residue pairs
        # and inflating map intensities relative to atomistic.
        cutoff_mask = self._bead_valid_mask & (dist_nm < _SOLVENT_CUTOFF_NM)
        r_safe = np.where(cutoff_mask, dist_nm, 1.0)

        # ── LJ-12-6 (Lorentz-Berthelot combining rules) ───────────────────────
        sig_ij  = 0.5 * (self._bead_sigmas[:, np.newaxis] + self._bead_sigmas[np.newaxis, :])
        eps_ij  = np.sqrt(self._bead_epsilons[:, np.newaxis] * self._bead_epsilons[np.newaxis, :])
        sr6     = (sig_ij / r_safe) ** 6
        lj_rep_b = np.where(cutoff_mask,  4.0 * eps_ij * sr6 * sr6, 0.0)
        lj_att_b = np.where(cutoff_mask, -4.0 * eps_ij * sr6,       0.0)
        del sr6, sig_ij, eps_ij

        # ── Coulomb with reaction-field correction ────────────────────────────
        # Matches the OpenMM CustomNonbondedForce expression exactly:
        #   E = K * q1*q2 * (1/r + k_rf*r² − c_rf)   for r < cutoff
        #   E = 0                                       for r >= cutoff
        q_ij    = self._bead_charges[:, np.newaxis] * self._bead_charges[np.newaxis, :]
        es_raw  = np.where(
            cutoff_mask,
            _COULOMB_K * q_ij * (1.0 / r_safe + _RF_K * r_safe ** 2 - _RF_C),
            0.0,
        )
        es_att_b = np.where(es_raw < 0, es_raw, 0.0)
        es_rep_b = np.where(es_raw > 0, es_raw, 0.0)
        del q_ij, es_raw, dist_nm, r_safe

        # ── Aggregate B×B → N×N via indicator matrix ─────────────────────────
        # Because beads are ordered by residue and lj_att_b is upper-triangle,
        # the aggregation naturally preserves upper-triangle structure: all
        # bead pairs b1 < b2 with residue[b1] < residue[b2] land in [i, j] i<j.
        ind = self._indicator
        vdw_att = ind @ lj_att_b @ ind.T
        vdw_rep = ind @ lj_rep_b @ ind.T
        es_att  = ind @ es_att_b @ ind.T
        es_rep  = ind @ es_rep_b @ ind.T
        del lj_att_b, lj_rep_b, es_att_b, es_rep_b

        # ── BB–BB distance matrix (N×N), upper triangle, k=3 exclusion ───────
        # Uses backbone bead positions only — matches the Cα-bead convention and
        # provides the distance map for ProteogramV2.calc_hydrophobicity_map.
        bb_pos  = pos_prot[self._bb_bead_indices]
        bb_diff = bb_pos[:, np.newaxis, :] - bb_pos[np.newaxis, :, :]
        bb_dist = np.sqrt(np.einsum('ijk,ijk->ij', bb_diff, bb_diff))
        del bb_diff

        i_idx, j_idx = np.triu_indices(N, k=3)
        dist_mask = np.zeros((N, N), dtype=bool)
        dist_mask[i_idx, j_idx] = True
        dist_ang = np.where(dist_mask, bb_dist * 10.0, 0.0)

        return vdw_att, vdw_rep, es_att, es_rep, dist_ang

    # ── full pipeline (mirrors NonBondedForceModel.run_full_pipeline) ─────────

    def run_full_pipeline(
        self,
        npt_steps: int = DEFAULT_NVT_STEPS,
        nvt_steps: int = DEFAULT_NVT_STEPS,
        production_steps: int = DEFAULT_PRODUCTION_STEPS,
        energy_calc_interval: int = DEFAULT_ENERGY_INTERVAL,
        return_simulated_pdb: bool = False,
        subtract_solvent_energies: bool = False,       # no-op for CG
        debug: bool = False,
    ) -> list:
        """Run the complete Martini CG MD pipeline with explicit solvent.

        Mirrors NonBondedForceModel.run_full_pipeline() in signature and return
        format. The pipeline includes an NPT equilibration phase to allow the
        solvation box volume to relax before production data collection.

        Pipeline:
          1. setup_system     — build multi-bead topology, solvate, add ions
          2. minimize_energy  — relax initial bead geometry
          3. equilibrate_nvt  — thermal equilibration (NVT, barostat active)
          4. equilibrate_npt  — box volume equilibration (NPT)
          5. run_production   — accumulate frame-averaged residue-level energies

        Args:
            npt_steps: NPT equilibration steps (default 25 000 = 250 ps).
            nvt_steps: NVT equilibration steps (default 25 000 = 250 ps).
            production_steps: Production steps (default 250 000 = 5 ns at 20 fs/step).
            energy_calc_interval: Steps between energy-snapshot frames.
            return_simulated_pdb: If True, append a BB-bead PDB stream as the
                last element of the returned list.
            subtract_solvent_energies: Ignored; present for API compatibility.
            debug: Log per-frame energies to self.energy_log if True.

        Returns:
            [vdw_att, vdw_rep, es_att, es_rep, dist_avg]  (5 NxN ndarrays)
            or [..., pdb_stream] if return_simulated_pdb=True.
        """
        print("=" * 60)
        print("Starting Martini CG MD pipeline (explicit solvent)")
        print("=" * 60)
        self.debug = debug

        print("\n[Step 1/5] Setting up Martini CG system...")
        self.setup_system()

        print("\n[Step 2/5] Energy minimization...")
        self._create_new_simulation()
        self.minimize_energy()

        print("\n[Step 3/5] NVT equilibration...")
        self.equilibrate_nvt(steps=nvt_steps)

        print("\n[Step 4/5] NPT equilibration...")
        n_real_residues = int((~self._dummy_residue_mask).sum())
        if n_real_residues < 50:
            print(f"  Skipping NPT for small protein ({n_real_residues} residues < 50): "
                  "pressure coupling is unstable at this scale.")
        else:
            self.equilibrate_npt(steps=npt_steps)

        print("\n[Step 5/5] Production MD...")
        # Switch to the larger production timestep now that the system is
        # thermally and mechanically equilibrated.
        self.timestep = self.DEFAULT_PRODUCTION_TIMESTEP * femtoseconds
        results = self.run_production(
            steps=production_steps,
            energy_calc_interval=energy_calc_interval,
        )

        if return_simulated_pdb:
            results.append(self._get_bb_pdb_stream())

        print("\n" + "=" * 60, flush=True)
        print("Martini CG pipeline complete!", flush=True)
        print("=" * 60, flush=True)

        self.cleanup_all_resources(final_run=True)
        return results

    # ── PDB output ────────────────────────────────────────────────────────────

    def _get_bb_pdb_stream(self) -> io.StringIO:
        """Return current BB bead positions as an in-memory CA-only PDB stream."""
        topo  = Topology()
        chain = topo.addChain()
        carbon = Element.getBySymbol('C')
        for resname in self._residue_names:
            res = topo.addResidue(resname, chain)
            topo.addAtom('CA', carbon, res)

        # Extract raw Vec3 values (no units) then re-wrap as ONE Quantity.
        # A plain Python list of individual Quantity(Vec3, nm) objects would
        # cause PDBFile.writeFile to call np.array() on the list, producing an
        # object-dtype array, and the subsequent np.isnan() call raises TypeError.
        pos_values = self.positions.value_in_unit(nanometer)
        bb_vecs = [pos_values[int(idx)] for idx in self._bb_bead_indices]

        # Wrap backbone bead coordinates into the primary box using Python
        # float64 arithmetic. OpenMM's enforcePeriodicBox uses single-precision
        # internally, which loses all decimal precision when coordinates have
        # drifted by hundreds of millions of box widths, leaving coordinates
        # unchanged or incorrectly wrapped. Python's % operator is exact in
        # float64 even for values at 10^9 nm scale.
        if self._box_lengths is not None:
            Lx, Ly, Lz = float(self._box_lengths[0]), float(self._box_lengths[1]), float(self._box_lengths[2])
            bb_vecs = [Vec3(v.x % Lx, v.y % Ly, v.z % Lz) for v in bb_vecs]

        bb_positions = bb_vecs * nanometer  # single Quantity(list[Vec3], nm)

        stream = io.StringIO()
        PDBFile.writeFile(topo, bb_positions, stream)
        stream.seek(0)
        return stream
