# Proteogram v2 — Overview of Changes

## What changed from v1

Proteogram v1 built a **symmetric** NxN 3‑channel image from static structural and chemical features:

1. Cα–Cα backbone distances (distogram)
2. Residue–residue hydrophobicity similarity
3. Residue–residue charge similarity

Proteogram v2 replaces those static maps with **physics‑based energies derived from a full molecular dynamics (MD) simulation**, and packs **six** pairwise maps into a single **asymmetric** NxN RGB image (three channels in the upper triangle, three in the lower triangle).

## High-level pipeline

```
PDB structure
   │
   ▼
MD simulation (OpenMM, AMBER ff19SB)
   energy minimize → NPT → NVT → 1 ns production
   (snapshots every 20 ps; GPU CUDA; solvent energies subtracted)
   │
   ▼
Pairwise residue maps (6 channels total)
   ┌──────────────────────────────┬──────────────────────────────┐
   │  Upper triangle (MD-derived) │  Lower triangle (complement) │
   │  R: VdW attractive (r⁻⁶)     │  R: Electrostatic attractive │
   │  G: VdW repulsive  (r⁻¹²)    │  G: Electrostatic repulsive  │
   │  B: Cα distogram             │  B: Hydrophobicity Δ         │
   └──────────────────────────────┴──────────────────────────────┘
   │
   ▼
Asymmetric NxN RGB Proteogram (each channel normalized to [0–255])
   │
   ▼
CNN / ResNet18 → embedding → cosine similarity search
```

## Channel definitions

### Upper triangle — MD-derived pairwise energies (kJ/mol, AMBER ff19SB, averaged over 1 ns production)

| Channel | Property | Notes |
|---------|----------|-------|
| R | Van der Waals attractive | London dispersion, r⁻⁶ term; 0.8 nm recording cutoff |
| G | Van der Waals repulsive | Pauli repulsion, r⁻¹² term; 0.8 nm recording cutoff |
| B | Cα pairwise distance | All-pairs distogram from production trajectory; no cutoff |

### Lower triangle — complementary energies and chemistry

| Channel | Property | Notes |
|---------|----------|-------|
| R | Electrostatic attractive | Opposite-charge residue pairs (qᵢ·qⱼ < 0); direct Coulomb, no cutoff |
| G | Electrostatic repulsive | Like-charge residue pairs (qᵢ·qⱼ > 0); direct Coulomb, no cutoff |
| B | Hydrophobicity Δ | Absolute hydrophobicity difference within 10 Å Cα cutoff |

All six maps are normalized to **[0–255]** before being combined into the final RGB image. Because upper and lower triangles encode different things, the v2 proteogram is **asymmetric** (v1 was symmetric).

## Key implementation pieces added in v2

- **`proteogram/v2/nonbonded_forces.py`** — `NonBondedForceModel` class wrapping the OpenMM MD pipeline (energy minimize → NPT → NVT → production) and producing four residue-residue energy matrices (VdW attractive/repulsive, electrostatic attractive/repulsive).
- **`proteogram/v2/proteogram.py`** — Combines the four MD-derived energy matrices, the Cα distogram, and the hydrophobicity-delta map into the final asymmetric RGB image.
- **`proteogram/v2/image_similarity.py`** — Image-embedding and similarity search adapted for the v2 format.
- **`scripts/v2/`** — Full v2 workflow: `create_v2_proteograms.py`, `measure_similarity_v2.py`, `train_multiple_models.py` (CNN from scratch or fine-tuned ResNet18), `evaluate_methods_v2.py` (vs GTalign and USalign), plus annotation/data-prep helpers.
- **Repo reorg (#5)** — `proteogram/` and `scripts/` split into `v1/` and `v2/` subfolders, with shared code in `proteogram/common/`.
- **PTM handling and eval updates (#6)** — Better treatment of post-translational modifications and additional evaluation patches.
- **Docker / GPU** — `Dockerfile.gpu` for CUDA 12 builds; `uv` extras `cuda12` to install `openmm-cuda-12` for GPU-accelerated MD.

## Default MD parameters (from `NonBondedForceModel`)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| Temperature | 311.75 K | Body-temperature sampling |
| Pressure | 1.0 atm | NPT phase |
| Water box padding | 1.0 nm | Around the protein |
| Timestep | 2.0 fs | Standard with HBond constraints |
| NPT steps | 50,000 | 100 ps equilibration |
| NVT steps | 50,000 | 100 ps equilibration |
| Production steps | 500,000 | 1 ns production |
| Energy snapshot interval | 10,000 steps | 50 frames over the production run |

## Resource usage (NVIDIA RTX 4090, CUDA 12.2 via OpenMM)

| Protein length | Max RAM | Max GPU VRAM | Approx. wall time |
|----------------|---------|--------------|-------------------|
| 50 residues    | ~900 MB | ~800 MB      | ~5 min            |
| 200 residues   | ~1 GB   | ~900 MB      | ~53 min           |

## Why this matters

The v1 representation is fast and purely geometric/physico-chemical. The v2 representation is much more expensive to compute (it runs a real MD trajectory per structure) but encodes **dynamic, physics-grounded interactions**: explicit-solvent equilibration, conformational sampling, and per-pair Lennard-Jones and Coulomb energetics. The hypothesis behind v2 is that downstream CNN embeddings should become more discriminative because each pixel now reflects how residues actually interact under the force field, not just how far apart they sit.

## References (from the v2 update)

- **OpenMM 7** — Eastman et al., *PLoS Comput. Biol.* 13(7):e1005659 (2017).
- **AMBER ff19SB** — Tian et al., *J. Chem. Theory Comput.* 16(1):528–552 (2020).
- **SCOPe 2.08** — Chandonia, Fox & Brenner, *J. Mol. Biol.* 429(3):348–355 (2017).
- **GTalign** — Margelevicius, *Nat. Commun.* 15:1261 (2024).
- **US-align** — Zhang et al., *Nat. Methods* 19:1109–1115 (2022).
- **ResNet** — He et al., *CVPR* 2016.

## Relevant v2 commits

- `559e4ff` Molecular dynamics simulation for new energy calculations — Proteogram v2 (#4)
- `f145c7e` Reorg repo for v1 and v2 approaches (#5)
- `8bb5ee0` Dealing better with PTMs, eval updates, and patches (#6)
