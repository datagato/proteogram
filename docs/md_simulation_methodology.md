# Molecular Dynamics Simulation Methodology for Residue-Residue Interaction Energies and Forces in Proteograms

This document describes the complete pipeline for calculating pairwise residue-residue interaction energies and forces from molecular dynamics (MD) simulations using OpenMM.

## Table of Contents

1. [Overview](#overview)
2. [PDB Preprocessing](#pdb-preprocessing)
3. [System Setup](#system-setup)
4. [Simulation Pipeline](#simulation-pipeline)
   - [Energy Minimization](#1-energy-minimization)
   - [NPT Equilibration](#2-npt-equilibration)
   - [NVT Equilibration](#3-nvt-equilibration)
   - [Production MD](#4-production-md)
5. [Force and Energy Calculations](#force-and-energy-calculations)
   - [Van der Waals Interactions](#van-der-waals-lennard-jones-interactions)
   - [Electrostatic Interactions](#electrostatic-coulomb-interactions)
6. [Output Matrices](#output-matrices)
7. [Usage Example](#usage-example)
8. [Appendix](#appendix)
9. [References](#references)

---

## Overview

The goal of this pipeline is to compute **pairwise residue-residue interaction energies and forces** for a protein structure. These interactions are separated into:

- **Van der Waals (VdW)**: Attractive and repulsive components
- **Electrostatic (ES)**: Attractive and repulsive components

The pipeline uses the **AMBER ff19SB** force field [4] with **TIP3P-FB** water model, implemented in OpenMM [3].

---

## PDB Preprocessing

Before simulation, the PDB structure is "fixed" using PDBFixer:

| Step | Description |
|------|-------------|
| 1 | Find missing residues |
| 2 | Find and replace non-standard residues |
| 3 | Remove heterogens (ligands, crystal waters) |
| 4 | Find and add missing atoms |
| 5 | Add hydrogens at pH 7.0 |

---

## System Setup

### Force Field
- **Protein**: AMBER ff19SB (`amber19-all.xml`) [2, 4]
- **Water**: TIP3P-FB (`amber19/tip3pfb.xml`)

### Solvation
| Parameter | Value |
|-----------|-------|
| Water model | TIP3P |
| Box padding | 1.0 nm |
| Neutralization | Yes ($Na^+$/$Cl^-$ ions) |

### Partial Charge Assignment

When `forcefield.createSystem()` is called, OpenMM automatically assigns **partial charges** to all atoms using pre-computed values from the AMBER ff19SB force field.


#### RESP Charges (Restrained Electrostatic Potential)

AMBER force fields use **RESP charges** [8], which are derived from:

1. **Quantum mechanical (QM) calculations** at the HF/6-31G* level of theory
2. **Electrostatic potential (ESP) fitting** - charges are optimized to reproduce the QM electrostatic potential around the molecule
3. **Restraints** - equivalent atoms (e.g., methyl hydrogens) are constrained to have equal charges
4. **Multi-conformational fitting** - multiple conformations are averaged for robust charges


RESP charges are specifically parameterized to work synergistically with the other AMBER force field terms (VdW, bonds, angles, dihedrals), ensuring accurate reproduction of experimental properties like solvation free energies and protein folding thermodynamics.

---

## Simulation Pipelines

### Default Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Temperature | 300 K | Simulation temperature |
| Pressure | 1 atm | Simulation pressure |
| Timestep | 2 fs | Integration timestep |
| Integrator | Langevin Middle | With 1 $ps^{-1}$ friction |

> **Note: State Continuity During Equilibration**
>
> The equilibration stages preserve **atomic positions** between stages, but systems are recreated as needed:
>
> 1. **Energy Minimization** → Optimized positions are saved
> 2. **NPT Equilibration** → Starts from minimized positions; barostat is **added** to allow volume changes; uses alpha-carbon restraints
> 3. **NVT Equilibration** → A **new system is created** from NPT positions; no barostat, with alpha-carbon restraints and temperature ramping
> 4. **Production MD** → A **new system is created** (without constraints) for accurate force calculations
>
> This approach ensures each stage has the appropriate force configuration while preserving the structural progress from previous stages.

---

### 1. Energy Minimization

**Purpose**: Remove steric clashes and bad contacts in the initial structure.

| Parameter | Value |
|-----------|-------|
| Algorithm | L-BFGS (OpenMM default) |
| Max iterations | 1000 |
| Constraints | Alpha-carbon (CA) position restraints (1000 kJ/mol/nm²) |

The system's potential energy is minimized by adjusting atomic positions until forces are below a tolerance threshold. Alpha-carbon position restraints prevent large-scale structural changes during minimization while allowing local relaxation of side chains and removal of steric clashes.

---

### 2. NPT Equilibration

**Purpose**: Equilibrate the system at constant **N**umber of particles, **P**ressure, and **T**emperature to achieve proper density.

| Parameter | Value |
|-----------|-------|
| Ensemble | NPT |
| Steps | 50,000 (100 ps) |
| Barostat | Monte Carlo Barostat |
| Pressure | 1 atm |
| Temperature | 300 K |
| Reporting interval | 5,000 steps (10 ps) |
| Constraints | Alpha-carbon (CA) position restraints (1000 kJ/mol/nm²) |

The Monte Carlo Barostat adjusts the simulation box volume to maintain constant pressure. Alpha-carbon atoms are restrained to their initial positions using harmonic restraints with a force constant of **1000 kJ/mol/nm²** to prevent large-scale protein conformational changes during equilibration while allowing side chains and solvent to relax.

---

### 3. NVT Equilibration

**Purpose**: Equilibrate at constant **N**umber of particles, **V**olume, and **T**emperature after fixing the box size from NPT.

| Parameter | Value |
|-----------|-------|
| Ensemble | NVT |
| Steps | 50,000 (100 ps) |
| Temperature ramp | 5 K → 300 K |
| Ramp schedule | +5 K every ~833 steps (60 increments) |
| Reporting interval | 5,000 steps (10 ps) |
| Constraints | Alpha-carbon (CA) position restraints (1000 kJ/mol/nm²) |

**Temperature Ramping Protocol**:
- Initial velocities set to 5 K
- Temperature increased gradually over 60 intervals
- Final temperature: 300 K

Alpha-carbon position restraints are maintained during temperature ramping to prevent protein unfolding during heating. This slow heating with backbone restraints helps prevent instabilities.

---

### 4. Production MD

**Purpose**: Generate an equilibrated trajectory for analysis.

| Parameter | Value |
|-----------|-------|
| Ensemble | NVT |
| Steps | 500,000 (1 ns) |
| Energy/force calculation interval | 10,000 steps (20 ps) |
| Constraints | None (for accurate force calculation) |

**Note**: HBonds constraints are removed during production to enable accurate force calculations on all atoms.

---

## Force and Energy Calculations

Forces are calculated as the **negative gradient of the potential energy** with respect to distance:

$$\vec{F} = -\nabla U = -\frac{dU}{dr}$$

### Van der Waals (Lennard-Jones) Interactions

The Lennard-Jones potential [1] describes van der Waals interactions:

#### Potential Energy

$$U_{LJ}(r) = 4\epsilon_{ij} \left[ \left(\frac{\sigma_{ij}}{r}\right)^{12} - \left(\frac{\sigma_{ij}}{r}\right)^{6} \right]$$

Where:
- $r$ = distance between atoms
- $\epsilon_{ij}$ = well depth (combined)
- $\sigma_{ij}$ = collision diameter (combined)

#### Combining Rules (Lorentz-Berthelot) [5, 6]

$$\sigma_{ij} = \frac{\sigma_i + \sigma_j}{2}$$

$$\epsilon_{ij} = \sqrt{\epsilon_i \cdot \epsilon_j}$$

#### Separated Energy Terms

| Component | Formula | Description |
|-----------|---------|-------------|
| **Repulsive** | $U_{rep} = 4\epsilon_{ij} \left(\frac{\sigma_{ij}}{r}\right)^{12}$ | Short-range Pauli repulsion |
| **Attractive** | $U_{att} = -4\epsilon_{ij} \left(\frac{\sigma_{ij}}{r}\right)^{6}$ | London dispersion |

#### Force Derivation

Taking the derivative of $U_{LJ}$ with respect to $r$:

$$F_{LJ}(r) = -\frac{dU_{LJ}}{dr} = \frac{24\epsilon_{ij}}{r} \left[ 2\left(\frac{\sigma_{ij}}{r}\right)^{12} - \left(\frac{\sigma_{ij}}{r}\right)^{6} \right]$$

#### Separated Force Terms

| Component | Formula | Sign Convention |
|-----------|---------|-----------------|
| **Repulsive** | $F_{rep} = \frac{48\epsilon_{ij}}{r} \left(\frac{\sigma_{ij}}{r}\right)^{12}$ | Positive (pushes apart) |
| **Attractive** | $F_{att} = -\frac{24\epsilon_{ij}}{r} \left(\frac{\sigma_{ij}}{r}\right)^{6}$ | Negative (pulls together) |

---

### Electrostatic (Coulomb) Interactions [7]

#### Potential Energy

$$U_{elec}(r) = \frac{k_e \cdot q_i \cdot q_j}{r}$$

Where:
- $k_e$ = Coulomb constant = 138.935456 kJ·nm/(mol·e²) in OpenMM units
- $q_i, q_j$ = partial charges in elementary charge units
- $r$ = distance between atoms

#### Force Derivation

$$F_{elec}(r) = -\frac{dU_{elec}}{dr} = \frac{k_e \cdot q_i \cdot q_j}{r^2}$$

#### Classification

| Condition | Energy | Force | Type |
|-----------|--------|-------|------|
| $q_i \cdot q_j > 0$ (like charges) | Positive | Positive | **Repulsive** |
| $q_i \cdot q_j < 0$ (opposite charges) | Negative | Negative | **Attractive** |

---

## Pairwise Residue Calculation

For each pair of protein residues $(i, j)$:

1. **Iterate over all atom pairs** $(a_i, a_j)$ where $a_i \in \text{residue } i$ and $a_j \in \text{residue } j$

2. **Calculate distance**: 
   $$r = \|\vec{r}_{a_j} - \vec{r}_{a_i}\|$$

3. **Skip if too close**: $r < 0.1$ nm (prevents singularities)

4. **Compute energies and forces** using formulas above

5. **Sum contributions** from all atom pairs

6. **Normalize** by the number of atom pairs:
   $$\bar{E}_{ij} = \frac{\sum_{a_i, a_j} E(a_i, a_j)}{n_{atoms,i} \times n_{atoms,j}}$$

7. **Store in upper triangle** of NxN matrix (symmetric)

---

## Output Matrices

The pipeline produces **8 NxN matrices** (where N = number of protein residues):

### Energy Matrices (units: kJ/mol)

| Matrix | Description |
|--------|-------------|
| `vdw_energy_attractive` | Attractive VdW energies ($r^{-6}$ term) |
| `vdw_energy_repulsive` | Repulsive VdW energies ($r^{-12}$ term) |
| `es_energy_attractive` | Attractive electrostatic (opposite charges) |
| `es_energy_repulsive` | Repulsive electrostatic (like charges) |

### Force Matrices (units: kJ/(mol·nm))

| Matrix | Description |
|--------|-------------|
| `vdw_force_attractive` | Attractive VdW forces |
| `vdw_force_repulsive` | Repulsive VdW forces |
| `es_force_attractive` | Attractive electrostatic forces |
| `es_force_repulsive` | Repulsive electrostatic forces |

### Matrix Properties

- **Dimensions**: N × N (protein residues only, excludes water/ions)
- **Storage**: Upper triangle only (matrices are symmetric)
- **Averaging**: Values averaged over all frames during production
- **Normalization**: Per atom-pair average for each residue pair

---

## Usage Example

```python
from proteogram.nonbonded_forces import NonBondedForceModel
import numpy as np

# Initialize model
model = NonBondedForceModel(
    pdb_path='protein.pdb',
    temperature=300.0,
    use_gpu=False
)

# Run full pipeline
(vdw_e_att, vdw_e_rep, es_e_att, es_e_rep,
 vdw_f_att, vdw_f_rep, es_f_att, es_f_rep, simulated_pdb_stream) = model.run_full_pipeline(
    npt_steps=50000,
    nvt_steps=50000,
    production_steps=500000,
    energy_calc_interval=10000,
    return_simulated_pdb=True
)

# Save the final production simulation structure
with open('production_pdb_file.pdb', 'w') as f:
    f.write(simulated_pdb_stream.read())

# Save VdW results
np.save('vdw_energy_attractive.npy', vdw_e_att)
np.save('vdw_force_attractive.npy', vdw_f_att)
# ... etc.
```

---

## Appendix

### Appendix A: Energy Monitoring and Validation

#### General Expectations

The "good" values for potential energy during MD simulations depend heavily on system size (number of atoms), but here is some general guidance:

##### NPT Equilibration
- **Potential energy should decrease and stabilize** over time
- Typical range: **-50,000 to -500,000 kJ/mol** for solvated proteins (depends on system size)
- Energy should fluctuate around a stable mean once equilibrated
- **Density** should converge to ~1.0 g/cm³ for aqueous systems

##### NVT Equilibration
- During temperature ramping (5K → 300K), energy will **increase gradually** with temperature (this is expected)
- Final energy should be similar to or slightly higher than NPT final energy
- Fluctuations should be smaller than NPT (fixed volume means less variability)

##### Production MD
- Energy should fluctuate around a **stable mean**
- No systematic drift (increasing or decreasing trend)
- Fluctuations typically 1-5% of mean energy

---

#### What to Watch For

| Indicator | Good | Bad |
|-----------|------|-----|
| **Energy trend** | Decreasing during equilibration, then stable | Continuously increasing or erratic |
| **Fluctuations** | Small, consistent | Large spikes or sudden jumps |
| **Magnitude** | Negative (bound system) | Positive (system exploding) |
| **Per-atom energy** | -10 to -20 kJ/mol/atom | > 0 or < -50 kJ/mol/atom |

#### Red Flags

1. **Positive total energy** → System is breaking apart ("exploding")
2. **Energy drops dramatically** → Atoms overlapping (bad geometry)
3. **Large oscillations that don't dampen** → Unstable integration (timestep too large?)
4. **Sudden energy spikes** → Steric clashes, bad contacts, or numerical instability

---

#### Rough Estimates by System Size

| System | Atoms (approx) | Typical Energy (kJ/mol) | Per-atom (kJ/mol) |
|--------|---------------|-------------------------|-------------------|
| Small protein + water | 10,000–30,000 | -100,000 to -200,000 | -10 to -15 |
| Medium protein + water | 50,000–100,000 | -300,000 to -600,000 | -10 to -15 |
| Large protein + water | 200,000+ | -1,000,000+ | -10 to -15 |

**Note**: These are rough estimates. Absolute values matter less than trends—track whether energy is stable, not whether it matches a specific number.

---

#### Energy Validation in Code

The pipeline includes automatic energy monitoring with warnings for:

- Positive potential energy (system instability)
- Large energy increases (>10% change between checks)
- Very large energy jumps (>100,000 kJ/mol)
- Unusual per-atom energies (positive or very negative)
- Large fluctuations during production (>5% of mean)

Example output:
```
Running NPT equilibration for 50000 steps...
  Initial potential energy: -150234.5 kJ/mol (-12.34 kJ/mol/atom)
  Final potential energy: -152456.7 kJ/mol (-12.52 kJ/mol/atom)
  Energy decreased by 2222.2 kJ/mol (good)
NPT equilibration complete.
```

---

### Appendix B: Alpha-Carbon Restraint Implementation

During energy minimization, NPT equilibration, and NVT equilibration, alpha-carbon (CA) atoms are restrained to their initial positions using OpenMM's [3] `CustomExternalForce` with a harmonic restraint potential that is periodic boundary condition (PBC) aware:

$$U_{restraint} = k \cdot d_{PBC}^2$$

Where:
- $k$ = 1000 kJ/mol/nm² (force constant)
- $d_{PBC}$ = minimum-image distance to reference position $(x_0, y_0, z_0)$

**Implementation**:
```python
# Get indices of CA atoms
ca_indices = [atom.index for atom in topology.atoms() if atom.name == 'CA']

# Custom external force to restrain CA atoms to their initial positions
restraint_force = CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")

# Add per-particle parameters for reference positions
restraint_force.addPerParticleParameter('x0') # reference x
restraint_force.addPerParticleParameter('y0') # reference y
restraint_force.addPerParticleParameter('z0') # reference z

# Add global parameter for force constant
restraint_force.addGlobalParameter('k', 1000.0 * kilojoules_per_mole / nanometer**2)

# Add CA atoms to the constraint force
for idx in ca_indices:
    restraint_force.addParticle(idx, positions[idx].value_in_unit(nanometers))
system.addForce(restraint_force)
```

### Appendix C: RESP Charges [8]

#### RESP Charges Lookup Table Approach

For standard amino acids, charges are **pre-computed and stored** in the force field XML files. OpenMM looks up these values based on residue name and atom name:

| Atom (Alanine) | Partial Charge (e) |
|----------------|-------------------|
| N | -0.4157 |
| H | +0.2719 |
| CA | +0.0337 |
| HA | +0.0823 |
| C | +0.5973 |
| O | -0.5679 |

#### RESP vs. Gasteiger

| Property | RESP (AMBER) | Gasteiger |
|----------|--------------|-----------|
| **Basis** | Quantum mechanical ESP | Electronegativity equalization |
| **Accuracy** | High (validated for MD) | Lower for polar groups |
| **Consistency** | Matched to force field VdW parameters | Standalone method |
| **Use case** | MD simulations | Cheminformatics, docking |

RESP charges are specifically parameterized to work synergistically with the other AMBER force field terms (VdW, bonds, angles, dihedrals), ensuring accurate reproduction of experimental properties like solvation free energies and protein folding thermodynamics [8, 9].

---

### Appendix D: Force and Energy Calculations - Detailed Derivations

This appendix provides detailed derivations and unit analysis for the force and energy calculations used in the pipeline.

#### Van der Waals (Lennard-Jones) Interactions [1]

##### Potential Energy

The Lennard-Jones potential is:

$$U_{LJ}(r) = 4\epsilon_{ij} \left[ \left(\frac{\sigma_{ij}}{r}\right)^{12} - \left(\frac{\sigma_{ij}}{r}\right)^{6} \right]$$

This is properly separated into:
- **Repulsive**: $U_{rep} = 4\epsilon_{ij} \left(\frac{\sigma_{ij}}{r}\right)^{12}$ (always positive)
- **Attractive**: $U_{att} = -4\epsilon_{ij} \left(\frac{\sigma_{ij}}{r}\right)^{6}$ (always negative)

##### Force Derivation

The force is the negative gradient of potential energy:

$$F = -\frac{dU}{dr}$$

Taking the derivative of $U_{LJ}$:

$$F_{LJ}(r) = -\frac{d}{dr}\left[4\epsilon_{ij} \left( \frac{\sigma_{ij}^{12}}{r^{12}} - \frac{\sigma_{ij}^{6}}{r^{6}} \right)\right]$$

$$F_{LJ}(r) = -4\epsilon_{ij} \left( -12\frac{\sigma_{ij}^{12}}{r^{13}} + 6\frac{\sigma_{ij}^{6}}{r^{7}} \right)$$

$$F_{LJ}(r) = \frac{24\epsilon_{ij}}{r} \left[ 2\left(\frac{\sigma_{ij}}{r}\right)^{12} - \left(\frac{\sigma_{ij}}{r}\right)^{6} \right]$$

##### Separated Force Terms

- **Repulsive**: $F_{rep} = \frac{48\epsilon_{ij}}{r} \left(\frac{\sigma_{ij}}{r}\right)^{12}$ (positive, pushes apart)
- **Attractive**: $F_{att} = -\frac{24\epsilon_{ij}}{r} \left(\frac{\sigma_{ij}}{r}\right)^{6}$ (negative, pulls together)

##### Combining Rules (Lorentz-Berthelot) [5, 6]

$$\sigma_{ij} = \frac{\sigma_i + \sigma_j}{2} \quad \text{(arithmetic mean)}$$

$$\epsilon_{ij} = \sqrt{\epsilon_i \cdot \epsilon_j} \quad \text{(geometric mean)}$$

---

#### Electrostatic (Coulomb) Interactions [7]

##### Potential Energy

$$U_{elec}(r) = \frac{k_e \cdot q_i \cdot q_j}{r}$$

Where $k_e = 138.935456$ kJ·nm/(mol·e²) is the Coulomb constant in OpenMM units [3].

##### Force Derivation

$$F_{elec}(r) = -\frac{dU_{elec}}{dr} = -\frac{d}{dr}\left(\frac{k_e q_i q_j}{r}\right) = \frac{k_e q_i q_j}{r^2}$$

The sign convention:
- Like charges ($q_i \cdot q_j > 0$): Force is **positive** (repulsive)
- Opposite charges ($q_i \cdot q_j < 0$): Force is **negative** (attractive)

---

#### Units Summary

| Quantity | OpenMM Units |
|----------|--------------|
| Distance $r$ | nanometers (nm) |
| Charge $q$ | elementary charge (e) |
| $\sigma$ | nanometers (nm) |
| $\epsilon$ | kJ/mol |
| Coulomb constant $k_e$ | kJ·nm/(mol·e²) |
| **Energy** | **kJ/mol** |
| **Force** | **kJ/(mol·nm)** |

##### Unit Verification for LJ Energy

$$[\epsilon] \cdot \left[\frac{\sigma}{r}\right]^6 = \frac{\text{kJ}}{\text{mol}} \cdot \left(\frac{\text{nm}}{\text{nm}}\right)^6 = \frac{\text{kJ}}{\text{mol}} \quad \checkmark$$

##### Unit Verification for LJ Force

$$\frac{[\epsilon]}{[r]} \cdot \left[\frac{\sigma}{r}\right]^{12} = \frac{\text{kJ/mol}}{\text{nm}} \cdot 1 = \frac{\text{kJ}}{\text{mol} \cdot \text{nm}} \quad \checkmark$$

##### Unit Verification for Coulomb Energy

$$[k_e] \cdot [q]^2 / [r] = \frac{\text{kJ} \cdot \text{nm}}{\text{mol} \cdot \text{e}^2} \cdot \text{e}^2 / \text{nm} = \frac{\text{kJ}}{\text{mol}} \quad \checkmark$$

##### Unit Verification for Coulomb Force

$$[k_e] \cdot [q]^2 / [r]^2 = \frac{\text{kJ} \cdot \text{nm}}{\text{mol} \cdot \text{e}^2} \cdot \text{e}^2 / \text{nm}^2 = \frac{\text{kJ}}{\text{mol} \cdot \text{nm}} \quad \checkmark$$

---

#### Summary Table

| Component | Energy Formula | Force Formula | Sign |
|-----------|----------------|---------------|------|
| **VdW Repulsive** | $4\epsilon_{ij} \left(\frac{\sigma_{ij}}{r}\right)^{12}$ | $\frac{48\epsilon_{ij}}{r} \left(\frac{\sigma_{ij}}{r}\right)^{12}$ | + (pushes) |
| **VdW Attractive** | $-4\epsilon_{ij} \left(\frac{\sigma_{ij}}{r}\right)^{6}$ | $-\frac{24\epsilon_{ij}}{r} \left(\frac{\sigma_{ij}}{r}\right)^{6}$ | - (pulls) |
| **ES Repulsive** | $\frac{k_e q_i q_j}{r}$ when $q_iq_j > 0$ | $\frac{k_e q_i q_j}{r^2}$ when $q_iq_j > 0$ | + (pushes) |
| **ES Attractive** | $\frac{k_e q_i q_j}{r}$ when $q_iq_j < 0$ | $\frac{k_e q_i q_j}{r^2}$ when $q_iq_j < 0$ | - (pulls) |

---

## References

1. Lennard-Jones, J. E. (1931). "Cohesion". Proceedings of the Physical Society. 43 (5): 461–482.
2. Case, D. A., et al. (2020). "AMBER 2020 Reference Manual".
3. Eastman, P., et al. (2017). "OpenMM 7: Rapid development of high performance algorithms for molecular dynamics". PLOS Computational Biology.
4. Tian, C., et al. (2020). "ff19SB: Amino-Acid-Specific Protein Backbone Parameters Trained against Quantum Mechanics Energy Surfaces in Solution". Journal of Chemical Theory and Computation. 16 (1): 528–552.
5. Lorentz, H. A. (1881). "Ueber die Anwendung des Satzes vom Virial in der kinetischen Theorie der Gase". Annalen der Physik. 248 (1): 127–136.
6. Berthelot, D. (1898). "Sur le mélange des gaz". Comptes Rendus. 126: 1703–1706.
7. Coulomb, C. A. (1785). "Premier mémoire sur l'électricité et le magnétisme". Histoire de l'Académie Royale des Sciences. 569–577.
8. Bayly, C. I., et al. (1993). "A well-behaved electrostatic potential based method using charge restraints for deriving atomic charges: the RESP model". The Journal of Physical Chemistry. 97 (40): 10269–10280.
9. Cornell, W. D., et al. (1995). "A Second Generation Force Field for the Simulation of Proteins, Nucleic Acids, and Organic Molecules". Journal of the American Chemical Society. 117 (19): 5179–5197.
