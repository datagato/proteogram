# Coarse-Grained Molecular Dynamics Simulation Methodology: Martini 3-Inspired Multi-Bead Model

This document describes the Martini 3-inspired coarse-grained (CG) pipeline for calculating pairwise residue-residue interaction energies from molecular dynamics simulations using OpenMM. It is a companion to the atomistic methodology document and covers the scientific basis, force-field parameterisation, explicit solvent treatment, simulation pipeline, and usage of the `MartiniNonBondedForceModel` class.

## Table of Contents

1. [Overview](#overview)
2. [Motivation: Why Martini?](#motivation-why-martini)
3. [System Representation](#system-representation)
   - [Multi-Bead Residue Mapping](#multi-bead-residue-mapping)
   - [Bead Types and LJ Parameters](#bead-types-and-lj-parameters)
   - [Bead Charges](#bead-charges)
   - [Explicit Solvent and Ions](#explicit-solvent-and-ions)
4. [Force Field](#force-field)
   - [Bonded Interactions](#bonded-interactions)
   - [Van der Waals Interactions](#van-der-waals-lennard-jones-interactions)
   - [Electrostatic Interactions: Reaction-Field Coulomb](#electrostatic-interactions-reaction-field-coulomb)
   - [Exclusions and Cutoff](#exclusions-and-cutoff)
   - [Barostat](#barostat)
5. [Solvation Protocol](#solvation-protocol)
   - [Water Bead Placement](#water-bead-placement)
   - [Ion Placement](#ion-placement)
6. [Simulation Pipeline](#simulation-pipeline)
   - [Energy Minimization](#1-energy-minimization)
   - [NVT Equilibration](#2-nvt-equilibration)
   - [NPT Equilibration](#3-npt-equilibration)
   - [Production MD](#4-production-md)
7. [Energy Calculations](#energy-calculations)
   - [Bead-Level Pairwise Energies](#bead-level-pairwise-energies)
   - [Aggregation to Residue Level](#aggregation-to-residue-level)
   - [Distance Matrix](#distance-matrix)
8. [Output Matrices](#output-matrices)
9. [Comparison with Atomistic Model](#comparison-with-atomistic-model)
10. [Usage Example](#usage-example)
11. [Appendix](#appendix)
12. [Future Work](#future-work)
13. [References](#references)

---

## Overview

The Martini pipeline computes the same **pairwise residue-residue interaction energies** as the atomistic pipeline — Van der Waals (attractive and repulsive) and electrostatic (attractive and repulsive) — using a multi-bead CG representation inspired by the Martini 3 force field [1].

Each residue is represented by **1–4 beads** depending on sidechain complexity. Bead positions are computed as centroids of the corresponding heavy atoms in the input PDB file. Compared to the atomistic model, this model:

- Encodes sidechain geometry explicitly through multiple beads per residue
- Uses distinct bead types with physically meaningful LJ parameters from Martini 3
- Simulates **explicit CG water** (W beads, each representing ~4 H₂O molecules) and **NaCl ions**
- Applies **reaction-field electrostatics** with proper solvent screening at the periodic boundary

The `MartiniNonBondedForceModel` class implements an identical `run_full_pipeline()` interface to `AtomisticNonBondedForceModel`, returning the same five output matrices. `ProteogramV2` selects between models via the `cg_method` argument (`'martini'` or `None` for atomistic).

---

## Motivation: Why Martini?

| Property | Atomistic | Martini |
|---|---|---|
| Beads/atoms per residue | ~10 heavy + H | 1–4 |
| Sidechain geometry | Full | Coarse (1–3 SC beads) |
| Solvent | Explicit TIP3P water | Explicit CG W beads |
| Electrostatics | PME (full long-range) | Reaction-field, 1.1 nm cutoff |
| Timestep | 2 fs | 10 fs (eq) / 20 fs (prod) |
| Typical speedup vs atomistic | 1× | 10–30× |

Martini provides a faster alternative to the atomistic model while retaining explicit solvent and chemically distinct sidechain representations:

- **Faster than atomistic**: the system has ~5–10× fewer particles than all-atom + TIP3P, and the 10–20 fs timestep (vs. 2 fs) further reduces wall time.
- **Explicit solvent with periodic boundaries**: solvates the protein in a periodic CG water box and equilibrates the box volume via NPT before production, providing realistic dielectric screening.
- **Chemically distinct sidechain beads**: bead types (apolar C, polar N, charged Q, aromatic TC) encode sidechain chemical identity more explicitly than a single epsilon value.

---

## System Representation

### Multi-Bead Residue Mapping

Each residue contributes one **backbone (BB)** bead placed at the centroid of `N, CA, C, O` backbone atoms, plus 0–3 **sidechain (SC)** beads at sidechain atom centroids. If an expected atom is missing from the PDB, the code falls back to the Cα position automatically.

| Residue(s) | Beads | Labels |
|---|---|---|
| GLY | 1 | BB |
| ALA, VAL, LEU, ILE, PRO, MET, CYS | 2 | BB, SC1 |
| SER, THR, ASN, GLN, ASP, GLU | 2 | BB, SC1 |
| LYS, ARG, HIS, PHE, TYR | 3 | BB, SC1, SC2 |
| TRP | 4 | BB, SC1, SC2, SC3 |

The BB bead is always the first bead within a residue. This ordering is used throughout the aggregation pipeline.

---

### Bead Types and LJ Parameters

Five protein bead types encode chemical identity through different LJ well depths and radii, following approximate Martini 3 parameters [1]:

| Bead type | Residues / role | σ (nm) | ε (kJ/mol) |
|---|---|---|---|
| BB | All backbone beads | 0.47 | 5.6 |
| C | Apolar sidechains (ALA, VAL, LEU, ILE, PRO, MET, CYS) | 0.47 | 4.5 |
| N | Polar sidechains (SER, THR, ASN, GLN; linker beads of LYS, ARG, HIS) | 0.47 | 3.6 |
| Q | Charged sidechains (ASP SC1, GLU SC1, LYS SC2, ARG SC2) | 0.47 | 5.6 |
| TC | Tiny cyclic — aromatic rings (HIS SC2, PHE SC1/SC2, TYR SC1/SC2, TRP SC1/SC2/SC3) | 0.38 | 3.1 |

Three additional bead types are used for explicit solvent and ions (not included in the residue-level energy maps):

| Bead type | Role | σ (nm) | ε (kJ/mol) | Mass (Da) |
|---|---|---|---|---|
| W | CG water (~4 H₂O per bead) | 0.47 | 1.00 | 72.0 |
| ION_NA | Na⁺ | 0.258 | 0.063 | 22.99 |
| ION_CL | Cl⁻ | 0.440 | 0.830 | 35.45 |

Pairwise parameters use Lorentz-Berthelot combining rules:

$$\sigma_{ij} = \frac{\sigma_i + \sigma_j}{2}, \qquad \epsilon_{ij} = \sqrt{\epsilon_i \cdot \epsilon_j}$$

Protein beads all have a uniform mass of **72 Da**, the approximate mass of an average amino acid fragment at this level of coarse-graining.

---

### Bead Charges

Only explicitly ionised sidechains carry charge. All backbone beads have charge 0. The charge assignments follow Martini 3 conventions:

| Residue | Charged bead | Charge (e) | Basis |
|---|---|---|---|
| ASP | SC1 (Q type) | −1.0 | Deprotonated at pH 7 |
| GLU | SC1 (Q type) | −1.0 | Deprotonated at pH 7 |
| LYS | SC2 (Q type) | +1.0 | Fully protonated at pH 7 |
| ARG | SC2 (Q type) | +1.0 | Fully protonated at pH 7 |
| HIS | SC2 (TC type) | +0.5 | ~30% protonated at pH 7 (pKa ≈ 6.5) |
| All others | All beads | 0.0 | Neutral at pH 7 |

---

### Explicit Solvent and Ions

The Martini simulation runs in an explicit solvent box:

- **W beads** representing ~4 H₂O each are placed on a cubic grid with 1.2 nm padding on each face of the protein bounding box.
- **Na⁺ and Cl⁻** ions are placed by randomly replacing W beads — first to neutralise the net protein charge, then to reach 0.15 M physiological NaCl.
- Periodic boundary conditions (PBC) are applied to all forces.

The solvent participates fully in the dynamics (driving realistic thermal fluctuations and dielectric screening) but is **excluded from the residue-level energy maps**: the numpy pairwise energy calculation slices only the first $B_{prot}$ bead positions, so water-protein, water-water, and ion-protein interactions do not appear in the output matrices.

---

## Force Field

### Bonded Interactions

#### Backbone bonds (BB–BB, inter-residue)

$$U_{\text{bond}}(r) = \frac{1}{2} k_{bb} (r - r_0)^2$$

| Parameter | Value |
|---|---|
| Force constant $k_{bb}$ | 3800 kJ/(mol·nm²) |
| Equilibrium length $r_0$ | 0.35 nm |

#### Backbone–sidechain bonds (BB–SC1, intra-residue)

$$U_{\text{bond}}(r) = \frac{1}{2} k_{bs} (r - r_0)^2$$

| Parameter | Value |
|---|---|
| Force constant $k_{bs}$ | 3800 kJ/(mol·nm²) |
| Equilibrium length $r_0$ | 0.27 nm |

#### Sidechain–sidechain bonds (SC–SC, intra-residue)

$$U_{\text{bond}}(r) = \frac{1}{2} k_{ss} (r - r_0)^2$$

| Parameter | Value |
|---|---|
| Force constant $k_{ss}$ | 2500 kJ/(mol·nm²) |
| Equilibrium length $r_0$ | 0.27 nm |

#### Backbone angles (BB–BB–BB)

$$U_{\text{angle}}(\theta) = \frac{1}{2} k_\theta (\theta - \theta_0)^2$$

| Parameter | Value |
|---|---|
| Force constant $k_\theta$ | 40 kJ/(mol·rad²) |
| Equilibrium angle $\theta_0$ | 127° (2.217 rad) |

127° is a standard Martini backbone angle for a generic / random-coil protein chain.

No torsion (dihedral) terms are included — a simplification relative to full Martini 3, acceptable for the proteogram application where the structural signal is the frame-averaged energy pattern rather than accurate free energies of specific conformations.

---

### Van der Waals (Lennard-Jones) Interactions

$$U_{LJ}(r) = 4\epsilon_{ij} \left[ \left(\frac{\sigma_{ij}}{r}\right)^{12} - \left(\frac{\sigma_{ij}}{r}\right)^{6} \right]$$

Applied as `CutoffPeriodic` with a 1.1 nm cutoff, matching the standard Martini 3 LJ cutoff [1]. The cutoff is applied to both protein–protein and protein–solvent pairs; the latter drives realistic solvation dynamics.

#### Separated energy terms

| Component | Formula | Sign | Physical meaning |
|---|---|---|---|
| **Repulsive** | $4\epsilon_{ij}(\sigma_{ij}/r)^{12}$ | + | Excluded volume / steric clash |
| **Attractive** | $-4\epsilon_{ij}(\sigma_{ij}/r)^{6}$ | − | Dispersion / hydrophobic contact |

---

### Electrostatic Interactions: Reaction-Field Coulomb

Standard Martini 3 uses **reaction-field (RF) electrostatics** [2] rather than bare Coulomb or PME. The RF formula models the dielectric response of the medium beyond the cutoff $r_c$ as a uniform continuum with permittivity $\varepsilon_s$ (bulk water):

$$U_{\text{RF}}(r) = \frac{k_e^* \cdot q_i q_j}{\varepsilon_r} \left( \frac{1}{r} + k_{\text{rf}} r^2 - c_{\text{rf}} \right), \quad r < r_c$$

Where:
- $k_e^* = 138.935456$ kJ·nm/(mol·e²) — vacuum Coulomb constant
- $\varepsilon_r = 15$ — protein-interior relative permittivity (electronic polarizability screening)
- $k_{\text{rf}} = \frac{\varepsilon_s - \varepsilon_r}{(2\varepsilon_s + \varepsilon_r) r_c^3}$ — RF correction curvature term
- $c_{\text{rf}} = \frac{3\varepsilon_s}{(2\varepsilon_s + \varepsilon_r) r_c}$ — RF correction constant (ensures continuity at $r_c$)

#### Parameter values

| Parameter | Value | Description |
|---|---|---|
| $r_c$ | 1.1 nm | LJ and RF cutoff |
| $\varepsilon_r$ | 15 | Protein-interior permittivity |
| $\varepsilon_s$ | 80 | Bulk water permittivity |
| $k_{\text{rf}}$ | 0.2791 nm⁻³ | RF curvature coefficient |
| $c_{\text{rf}}$ | 1.2468 nm⁻¹ | RF continuity constant |
| Effective $k_e$ | 138.935456 / 15 = 9.262 kJ·nm/(mol·e²) | Pre-screened Coulomb prefactor |

The full expression used in the `CustomNonbondedForce` and in the numpy energy calculation is:

$$U_{\text{RF}}(r) = 9.262 \cdot q_i q_j \left( \frac{1}{r} + 0.2791 \cdot r^2 - 1.2468 \right), \quad r < 1.1 \text{ nm}$$

The $k_{\text{rf}} r^2$ term grows without bound beyond the cutoff, so the 1.1 nm cutoff is applied strictly in both the OpenMM `CutoffPeriodic` force and in the numpy pairwise energy calculation during production snapshots.

#### Energy classification

| Condition | Energy | Type |
|---|---|---|
| $q_i q_j > 0$ (like charges) | Positive | **Repulsive** |
| $q_i q_j < 0$ (opposite charges) | Negative | **Attractive** |

---

### Exclusions and Cutoff

**Exclusions**: All intra-residue bead pairs are excluded from both LJ and Coulomb forces (backbone and sidechain beads within the same residue interact only through the bonded terms). Additionally, 1-2 and 1-3 backbone BB pairs across sequential residues are excluded.

**Cutoff**: `CutoffPeriodic` at 1.1 nm for both LJ and Coulomb. This is the standard Martini 3 non-bonded cutoff.

---

### Barostat

A `MonteCarloBarostat` is added to the system during `setup_system()` at 1 bar / 25-step update frequency. Its frequency is managed across pipeline stages:

| Stage | Barostat frequency |
|---|---|
| NVT equilibration | **0** (disabled — constant volume) |
| NPT equilibration | 25 (active — box volume adjusts) |
| Production | 25 (active — maintains pressure during sampling) |

Disabling the barostat during NVT is essential: the solvation box starts slightly under-dense (grid spacing at the LJ minimum rather than liquid density). Allowing volume rescaling before the system is thermally equilibrated causes extreme local forces that drive coordinates to NaN.

---

## Solvation Protocol

### Water Bead Placement

W beads are placed on a **cubic grid** with spacing set to the W–W LJ minimum distance:

$$d_{\text{grid}} = 2^{1/6} \cdot \sigma_W = 2^{1/6} \times 0.47 \approx 0.527 \text{ nm}$$

This spacing ensures adjacent W beads start at zero force — placing them at $\sigma_W$ (0.47 nm, the LJ zero-crossing) would put them in the repulsive region, launching beads during minimisation.

The box extends 1.2 nm beyond the protein bounding box on each face:

$$L_x = x_{\max} - x_{\min} + 2 \times 1.2 \text{ nm}, \quad \text{etc.}$$

W beads within **0.53 nm** of any protein bead are removed to prevent high-energy initial contacts. The resulting grid density is ~6.8 W beads/nm³, slightly below the Martini liquid-water target of ~8.4 W/nm³ — NPT equilibration shrinks the box to the correct density.

### Ion Placement

Ions are placed by randomly sampling W bead slots (fixed seed 42 for reproducibility):

1. **Neutralisation**: if the protein has net charge $q_{\text{net}}$ (rounded to nearest integer), add $|q_{\text{net}}|$ counterions (Na⁺ for negative protein, Cl⁻ for positive).
2. **Physiological salt**: add NaCl pairs to reach 0.15 M. The number of pairs is:

$$n_{\text{pairs}} = \text{round}\left( 0.15 \text{ mol/L} \times V_{\text{box}} \text{ (L)} \times N_A \right)$$

The selected W beads are replaced by ion particles; the remaining W beads stay in place.

---

## Simulation Pipeline

### Default Parameters

| Parameter | Value | Description |
|---|---|---|
| Temperature | 310.15 K (37 °C) | Physiological temperature |
| Equilibration timestep | 10 fs | Conservative while system settles (NVT + NPT) |
| Production timestep | 20 fs | Standard Martini 3 W-model timestep |
| Integrator | Langevin Middle | 1 ps⁻¹ friction coefficient |
| Protein bead mass | 72 Da | Uniform for all protein beads |
| W bead mass | 72 Da | ~4 × 18 Da |
| Box padding | 1.2 nm | Solvent buffer on each face |

> **State continuity**: bead positions are propagated between all pipeline stages. After each stage, `getState(enforcePeriodicBox=True)` is called to wrap positions back into the primary box before creating the next simulation object. After NPT equilibration, the equilibrated box vectors are explicitly propagated back to the topology, system defaults, and `_box_lengths` so that the production simulation starts with the correct box geometry.

---

### 1. Energy Minimization

**Purpose**: Relax any strained initial bead geometry and high-energy water–protein contacts.

| Parameter | Value |
|---|---|
| Algorithm | L-BFGS (OpenMM default) |
| Max iterations | 2,000 |

---

### 2. NVT Equilibration

**Purpose**: Thermalise the system to 310 K at constant volume before allowing box relaxation.

| Parameter | Value |
|---|---|
| Ensemble | NVT (barostat disabled) |
| Steps | 25,000 (250 ps at 10 fs/step) |
| Temperature | 310.15 K |
| Reporting interval | 2,500 steps (25 ps) |

Velocities are initialised from a Maxwell-Boltzmann distribution at 310 K. The barostat is disabled (frequency = 0) throughout this stage. Allowing box rescaling before thermalisation is complete causes the minimiser to produce NaN coordinates — the slightly under-dense initial grid generates large inter-bead forces that NPT would amplify before Langevin friction has damped them.

---

### 3. NPT Equilibration

**Purpose**: Allow the simulation box volume to relax to the correct liquid-water density under barostat control.

| Parameter | Value |
|---|---|
| Ensemble | NPT (barostat frequency = 25) |
| Steps | 25,000 (250 ps at 10 fs/step) |
| Pressure | 1.0 bar |
| Temperature | 310.15 K |
| Reporting interval | 2,500 steps (25 ps) with volume output |

After NPT equilibration, the converged box vectors are **propagated back** to the OpenMM topology, system defaults, and the internal `_box_lengths` array. Without this propagation, the subsequent production simulation is created with the original (over-large) box while bead positions correspond to the compressed NPT box — this creates extreme local density at one corner and immediately produces NaN.

---

### 4. Production MD

**Purpose**: Generate a thermally and mechanically equilibrated trajectory for residue-level energy sampling.

| Parameter | Value |
|---|---|
| Ensemble | NPT (barostat frequency = 25) |
| Steps | 250,000 (5 ns at 20 fs/step) |
| Energy snapshot interval | 5,000 steps (100 ps) |
| Frames collected | 50 |

At each snapshot, all bead positions are extracted and protein-only pairwise energies are computed in numpy (see [Energy Calculations](#energy-calculations)). Solvent and ion beads drive the dynamics but do not contribute to the output energy maps.

---

## Energy Calculations

### Bead-Level Pairwise Energies

At each production snapshot, bead-level $B_{prot} \times B_{prot}$ interaction matrices are computed in vectorised numpy, **consistent with the OpenMM force expressions**:

#### LJ energy (protein beads only, upper triangle)

$$U_{LJ,b_1 b_2} = \begin{cases}
4\epsilon_{b_1 b_2}\left[\left(\frac{\sigma_{b_1 b_2}}{r_{b_1 b_2}}\right)^{12} - \left(\frac{\sigma_{b_1 b_2}}{r_{b_1 b_2}}\right)^{6}\right] & r < 1.1 \text{ nm, non-excluded} \\
0 & \text{otherwise}
\end{cases}$$

Split into repulsive (positive $r^{-12}$ term) and attractive (negative $r^{-6}$ term) components.

#### Reaction-field Coulomb (protein beads only, upper triangle)

$$U_{\text{RF},b_1 b_2} = \begin{cases}
9.262 \cdot q_{b_1} q_{b_2} \left(\frac{1}{r} + 0.2791 r^2 - 1.2468\right) & r < 1.1 \text{ nm, non-excluded} \\
0 & \text{otherwise}
\end{cases}$$

Split into attractive ($U < 0$) and repulsive ($U > 0$) components.

#### Minimum image convention

Bead-pair distances use the minimum image convention (MIC) for periodic boundary conditions:

$$\Delta\vec{r}_{ij} \leftarrow \Delta\vec{r}_{ij} - \text{round}\!\left(\frac{\Delta\vec{r}_{ij}}{L}\right) \cdot L$$

where $L$ is the box edge length vector. This matches OpenMM's `CutoffPeriodic` treatment.

---

### Aggregation to Residue Level

Bead-level $B_{prot} \times B_{prot}$ matrices are collapsed to residue-level $N \times N$ matrices via the **indicator matrix** $\mathbf{I} \in \{0,1\}^{N \times B_{prot}}$, where $I_{ib} = 1$ iff bead $b$ belongs to residue $i$:

$$E^{\text{residue}}[i,j] = \sum_{b_1 \in i,\; b_2 \in j} E^{\text{bead}}[b_1, b_2] = \left(\mathbf{I} \cdot E^{\text{bead}} \cdot \mathbf{I}^T\right)_{ij}$$

This matrix multiply aggregates all bead-pair contributions (BB–BB, BB–SC, SC–SC) between residue $i$ and residue $j$ into a single residue-pair energy.

> **Contrast with atomistic**: the atomistic model iterates over all atom pairs within each residue pair and normalises by the number of atom pairs. The Martini model sums bead-pair energies without normalisation — each bead already represents multiple heavy atoms, so the energies are inherently coarser.

---

### Distance Matrix

The distance map uses **BB bead positions only**, providing a residue-level backbone distance matrix analogous to the Cα distance matrix in `ProteogramV2`:

$$d_{ij}^{\text{BB}} = \|\vec{r}_{BB,i} - \vec{r}_{BB,j}\| \times 10 \quad \text{(Å)}$$

Upper triangle only ($j \geq i + 3$); lower triangle is zero.

### Accumulation and averaging

All five matrices are accumulated over all production frames and averaged:

$$\bar{E}_{ij} = \frac{1}{N_{\text{frames}}} \sum_{f=1}^{N_{\text{frames}}} E_{ij}^{(f)}$$

---

## Output Matrices

The pipeline produces **5 N×N matrices** (where N = number of protein residues), identical in format to `AtomisticNonBondedForceModel`:

| Matrix | Formula | Units |
|---|---|---|
| `vdw_energy_attractive` | $-4\epsilon_{ij}(\sigma_{ij}/r)^6$, summed over bead pairs | kJ/mol |
| `vdw_energy_repulsive` | $4\epsilon_{ij}(\sigma_{ij}/r)^{12}$, summed over bead pairs | kJ/mol |
| `es_energy_attractive` | $U_{\text{RF}}$ when $q_{b_1}q_{b_2} < 0$, summed over bead pairs | kJ/mol |
| `es_energy_repulsive` | $U_{\text{RF}}$ when $q_{b_1}q_{b_2} > 0$, summed over bead pairs | kJ/mol |
| `dist_avg` | BB–BB bead distance | Å |

### Matrix properties

- **Dimensions**: N × N (protein residues only; solvent excluded)
- **Storage**: Upper triangle only
- **Averaging**: Frame-averaged over all production snapshots
- **Normalisation**: None — bead-pair energies are summed, not averaged per pair

### Normalisation convention in ProteogramV2

`ProteogramV2.normalize_map()` rescales each matrix independently to [0–255]. Attractive energy channels (`vdw_att`, `es_att`) have values ≤ 0; their absolute value is taken first so that zero (no interaction) maps to 0 (dark) and large-magnitude interactions map to 255 (bright). Repulsive and distance channels are already ≥ 0 and normalise correctly without transformation.

---

## Comparison with Atomistic Model

| Property | Atomistic | Martini |
|---|---|---|
| Class | `AtomisticNonBondedForceModel` | `MartiniNonBondedForceModel` |
| Particles | All heavy + H + TIP3P water | 1–4 CG beads/residue + W + ions |
| Solvent | Explicit TIP3P | Explicit W beads (CG) |
| Electrostatics | PME, long-range | Reaction-field, 1.1 nm cutoff |
| Dielectric | RESP partial charges | ε_r=15 protein, ε_s=80 RF boundary |
| Charges | RESP (all atoms) | Ionised sidechains only (on SC bead) |
| Timestep | 2 fs | 10 fs (eq) / 20 fs (prod) |
| Equilibration | NVT (100 ps) + NPT (100 ps) | NVT (250 ps) + NPT (250 ps) |
| Production | 1 ns (500,000 steps) | 5 ns (250,000 steps) |
| Energy interval | 20 ps (50 frames) | 100 ps (50 frames) |
| Typical speedup | 1× baseline | 10–30× |
| API / output | `run_full_pipeline()` → 5 matrices | identical |

---

## Usage Example

### Minimal API usage

```python
from proteogram.v2 import MartiniNonBondedForceModel

model = MartiniNonBondedForceModel(
    pdb_path='protein.pdb',
    output_dir='output',
    temperature=310.15,  # Kelvin
    use_gpu=False,
)

vdw_att, vdw_rep, es_att, es_rep, dist_avg = model.run_full_pipeline(
    nvt_steps=25000,           # 250 ps NVT equilibration
    npt_steps=25000,           # 250 ps NPT equilibration
    production_steps=250000,   # 5 ns production (at 20 fs/step)
    energy_calc_interval=5000, # snapshot every 100 ps (50 frames)
    debug=False,
)

model.cleanup_all_resources(final_run=True)
```

### Via ProteogramV2 (recommended)

```python
from proteogram.v2 import ProteogramV2

# Set cg_method at construction time — all calls use Martini CG
pg = ProteogramV2(
    pdb_path='protein.pdb',
    output_dir='output',
    chain_id='A',
    cg_method='martini',
)
proteogram_array, errors = pg.calculate_proteogram()

# Or override per-call to compare models on the same protein
pg = ProteogramV2('protein.pdb', 'output', 'A')
array_atomistic, _ = pg.calculate_proteogram(cg_method=None)
array_martini, _   = pg.calculate_proteogram(cg_method='martini')
```

---

## Appendix

### Appendix A: Energy Monitoring

#### Expected energy ranges (Martini)

| Stage | Expectation |
|---|---|
| After minimization | Large negative; W bead repulsions resolved |
| NVT equilibration | Energy decreases and stabilises; temperature converges to 310 K |
| NPT equilibration | Volume decreases from initial grid density (~6.8 W/nm³) to Martini water density (~8.4 W/nm³) |
| Production | Fluctuates around a stable mean; no systematic drift |

#### Per-bead energy reference

| System size | Approximate system particles | Typical potential energy |
|---|---|---|
| Small protein (50 residues) | ~150 protein + ~2,000 W + ions | −10,000 to −25,000 kJ/mol |
| Medium protein (100 residues) | ~300 protein + ~4,000 W + ions | −20,000 to −50,000 kJ/mol |
| Large protein (200 residues) | ~600 protein + ~8,000 W + ions | −40,000 to −100,000 kJ/mol |

Martini energies are larger in absolute magnitude than atomistic energies on a per-residue basis due to the coarser force field, but smaller in total because the system has fewer particles. Track stability and trends rather than specific values.

#### Setup summary printed by `setup_system()`

```
  Solvation box: 7.40 × 7.10 × 7.20 nm  (378.4 nm³)
  Placed 2431 W beads (89 removed for clashes with protein)
  Net protein charge = -2  →  neutralization: 2 Na⁺, 0 Cl⁻
  NaCl (0.15 M, 34 pairs): 36 Na⁺ total, 34 Cl⁻ total  [system net charge = 0]
Martini CG system: 153 residues, 287 protein beads + 2431 W + 36 Na⁺ + 34 Cl⁻ = 2788 total; 286 bonds, 18 charged protein beads
```

Sanity checks:
- System net charge should be 0 (or ±1 for rounding of His fractional charges)
- Box volume should grow with protein size
- W bead count should be several times the protein bead count

---

### Appendix B: Residue Bead Definitions

Full bead assignment table. Format: bead type / label / charge (e) / centroid atom names.

| Residue | BB | SC1 | SC2 | SC3 |
|---|---|---|---|---|
| GLY | BB/0.0/N,CA,C,O | — | — | — |
| ALA | BB/0.0/N,CA,C,O | C/SC1/0.0/CB | — | — |
| VAL | BB/0.0/N,CA,C,O | C/SC1/0.0/CB,CG1,CG2 | — | — |
| LEU | BB/0.0/N,CA,C,O | C/SC1/0.0/CB,CG,CD1,CD2 | — | — |
| ILE | BB/0.0/N,CA,C,O | C/SC1/0.0/CB,CG1,CG2,CD1 | — | — |
| PRO | BB/0.0/N,CA,C,O | C/SC1/0.0/CB,CG,CD | — | — |
| MET | BB/0.0/N,CA,C,O | C/SC1/0.0/CB,CG,SD,CE | — | — |
| CYS | BB/0.0/N,CA,C,O | C/SC1/0.0/CB,SG | — | — |
| SER | BB/0.0/N,CA,C,O | N/SC1/0.0/CB,OG | — | — |
| THR | BB/0.0/N,CA,C,O | N/SC1/0.0/CB,OG1,CG2 | — | — |
| ASN | BB/0.0/N,CA,C,O | N/SC1/0.0/CB,CG,OD1,ND2 | — | — |
| GLN | BB/0.0/N,CA,C,O | N/SC1/0.0/CB,CG,CD,OE1,NE2 | — | — |
| ASP | BB/0.0/N,CA,C,O | Q/SC1/−1.0/CB,CG,OD1,OD2 | — | — |
| GLU | BB/0.0/N,CA,C,O | Q/SC1/−1.0/CB,CG,CD,OE1,OE2 | — | — |
| LYS | BB/0.0/N,CA,C,O | N/SC1/0.0/CB,CG,CD | Q/SC2/+1.0/CE,NZ | — |
| ARG | BB/0.0/N,CA,C,O | N/SC1/0.0/CB,CG,CD | Q/SC2/+1.0/NE,CZ,NH1,NH2 | — |
| HIS | BB/0.0/N,CA,C,O | TC/SC1/0.0/CB,CG | TC/SC2/+0.5/ND1,CD2,CE1,NE2 | — |
| PHE | BB/0.0/N,CA,C,O | TC/SC1/0.0/CB,CG,CD1,CD2 | TC/SC2/0.0/CE1,CE2,CZ | — |
| TYR | BB/0.0/N,CA,C,O | TC/SC1/0.0/CB,CG,CD1,CD2 | TC/SC2/0.0/CE1,CE2,CZ,OH | — |
| TRP | BB/0.0/N,CA,C,O | TC/SC1/0.0/CB,CG,CD1,NE1 | TC/SC2/0.0/CD2,CE2,CZ2,CH2 | TC/SC3/0.0/CE3,CZ3 |

---

### Appendix C: Reaction-Field Parameter Derivation

The RF parameters follow Tironi et al. [2]:

$$k_{\text{rf}} = \frac{\varepsilon_s - \varepsilon_r}{2\varepsilon_s + \varepsilon_r} \cdot \frac{1}{r_c^3} = \frac{80 - 15}{2 \times 80 + 15} \cdot \frac{1}{1.1^3} = \frac{65}{175} \cdot \frac{1}{1.331} \approx 0.2791 \text{ nm}^{-3}$$

$$c_{\text{rf}} = \frac{3\varepsilon_s}{2\varepsilon_s + \varepsilon_r} \cdot \frac{1}{r_c} = \frac{3 \times 80}{2 \times 80 + 15} \cdot \frac{1}{1.1} = \frac{240}{175} \cdot \frac{1}{1.1} \approx 1.2468 \text{ nm}^{-1}$$

These terms ensure that:
1. The electrostatic potential is continuous at $r = r_c$
2. The electrostatic force is continuous at $r = r_c$ (no abrupt truncation artefact)
3. Long-range interactions ($r > r_c$) are approximated by the response of a dielectric continuum with $\varepsilon_s = 80$

The $c_{\text{rf}}$ constant shifts the potential so that $U_{\text{RF}}(r_c) = 0$, preventing a discontinuous energy jump at the cutoff that would introduce systematic errors in the simulation.

---

### Appendix D: Scientific Basis and Limitations

#### What the Martini CG model captures

- **Sidechain chemical identity**: distinct bead types (C, N, Q, TC) encode apolar, polar, charged, and aromatic character, providing chemically meaningful residue differentiation.
- **Salt bridges and electrostatic interactions**: charges placed on the correct sidechain bead give geometrically meaningful electrostatic interactions between charged pairs.
- **Solvent screening via explicit water**: W beads provide realistic dielectric boundary conditions and drive protein conformational sampling through collisions, analogously to TIP3P in the atomistic model.
- **Aromatic interactions**: TC bead types have smaller σ (0.38 nm vs. 0.47 nm) and lower ε (3.1 kJ/mol), partly encoding the reduced effective size of aromatic ring contacts.
- **Chain geometry**: backbone bonds (0.35 nm), BB-SC bonds (0.27 nm), and the 127° backbone angle maintain realistic protein topology.

#### What the Martini CG model does not capture

- **Dihedral terms**: the implementation omits backbone and sidechain torsion potentials present in full Martini 3. Secondary structure propensity is therefore weaker than the full force field.
- **Specific hydrogen bonding**: no explicit H-bond terms. Polar interactions are represented only through N-bead LJ parameters.
- **Desolvation penalties beyond LJ**: the cost of burying charged residues is partially captured through explicit water competition but not through an explicit transfer free energy term.
- **Side-chain conformational detail**: SC bead centroids cannot represent rotamer diversity; packing interactions are averaged into a single centroid position per bead.

#### Why these limitations are acceptable for proteograms

`ProteogramV2.normalize_map()` rescales each energy channel independently to [0–255] (attractive channels via `abs()` first). The **relative spatial pattern** of residue-residue interactions — which pairs are strongly interacting relative to others — is what the downstream model uses for structure comparison. The Martini model captures this pattern while running significantly faster than the atomistic model.

---

## Future Work

### Equilibration protocol refinements

The current equilibration protocol prioritises throughput and relies on a disabled barostat during NVT plus Langevin friction for stability. Two refinements, both common in canonical Martini and all-atom workflows, are worth exploring:

- **Less aggressive equilibration timestep.** Equilibration currently runs at 10 fs — already production-scale for the Martini W-model. More conservative protocols equilibrate at a smaller step (e.g. 2–5 fs) while the system is still settling, then ramp up to the 20 fs production timestep. A smaller equilibration step would reduce the risk of large-force instabilities on the initially under-dense solvation grid and could allow the barostat to be enabled earlier (or the NVT stage to be shortened), at the cost of some wall-clock time. Worth benchmarking whether it improves robustness on difficult (large, highly charged, or poorly packed) inputs without materially slowing the pipeline.

- **Position restraints on the solute during equilibration.** The current pipeline has no position-restraint stage; the protein beads are free to move throughout NVT and NPT. The more standard approach restrains solute (BB, and optionally SC) bead positions with a harmonic potential during equilibration, letting solvent and box relax around a fixed protein before releasing the restraints for production. This decouples solvent/density equilibration from protein relaxation, further reduces the chance of NaN blow-ups, and better preserves the input structure through the density-equilibration phase. This could be implemented as an optional `CustomExternalForce` restraint added during `setup_system()` and removed (or its force constant ramped to zero) before production.

Both changes trade a modest amount of speed for a more conservative, widely-validated equilibration path, and would make the pipeline more forgiving on structures where the current aggressive protocol occasionally requires retries.

---

## References

1. Souza, P. C. T., et al. (2021). "Martini 3: a general purpose force field for coarse-grained molecular dynamics". *Nature Methods*. 18(4): 382–388.
2. Tironi, I. G., et al. (1995). "A generalized reaction field method for molecular dynamics simulations". *The Journal of Chemical Physics*. 102(13): 5451–5459.
3. Eastman, P., et al. (2023). "OpenMM 8: Molecular Dynamics Simulation with Machine Learning Potentials". *Journal of Physical Chemistry B*. 128(1): 109-116.
