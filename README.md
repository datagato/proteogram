# Proteogram: an image embedding-based search approach to protein structure similarity

## Introduction

Proteogram is a novel approach to protein structure similarity search that represents protein structures as image data, enabling the use of computer vision models for efficient and accurate similarity detection. This repository leverages the SCOPe 2.08 protein structure dataset and classification hierarchy (https://scop.berkeley.edu) both to train as well as evaluate models.

### Proteogram v1: Distance, Hydrophobicity, and Charge Maps

The original Proteogram approach creates an NxN 3-channel image representation (where N is the residue length) by stacking three categories of residue-level information:

1. **Alpha-carbon backbone distances** - Pair-wise residue Cα distances (distogram)
2. **Hydrophobicity similarities** - Residue-residue hydrophobicity comparisons
3. **Charge similarities** - Residue-residue charge state comparisons

This representation captures both spatial similarity through distograms and physicochemical properties through hydrophobicity and charge maps. The resulting RGB image is inherently sequence-alignment independent and can be processed by standard computer vision models to generate embedding vectors for cosine-similarity-based search.

Example proteogram v1 (symmetric):

![](assets/3KFD_A.jpg)

### Proteogram v2: Incorporating MD Simulations

Proteogram v2 extends the original approach by incorporating molecular dynamics (MD) simulations to compute physics-based residue-residue interaction energies. Instead of using static distance and property maps, v2 runs a complete MD simulation pipeline using OpenMM with the AMBER ff19SB force field to calculate:

- **Van der Waals energies** - Attractive and repulsive Lennard-Jones interactions
- **Electrostatic energies** - Attractive and repulsive Coulomb interactions

The MD pipeline includes energy minimization, NPT and NVT equilibration, and production dynamics. The resulting 3-channel data (with 6 attributes in total) provides a richer representation of protein structure that accounts for dynamic conformational sampling and explicit solvent effects.

For detailed information on the MD simulation methodology, see the [MD Simulation Methodology documentation](docs/md_simulation_methodology.md).

The v2 Proteogram approach creates an NxN 3-channel image representation (where N is the residue length) by stacking three categories of physicochemical residue-level information in the upper triangle and three categories in the lower triangle, making v2 proteograms **asymmetric**:

**Upper triangle** — MD-derived pairwise energies (AMBER ff19SB, averaged over 1 ns production trajectory) and Cα distances:

| Channel | Property | Description |
|---------|----------|-------------|
| R | VdW attractive energy | London dispersion ($r^{-6}$ term), kJ/mol; atom pairs within 0.8 nm recording cutoff |
| G | VdW repulsive energy | Pauli repulsion ($r^{-12}$ term), kJ/mol; atom pairs within 0.8 nm recording cutoff |
| B | Cα pairwise distance | All-pairs distogram from production MD trajectory (no cutoff) |

**Lower triangle** — complementary MD-pairwise energies and a chemical property:

| Channel | Property | Description |
|---------|----------|-------------|
| R | Electrostatic attractive energy | Opposite-charge residue pairs ($q_i \cdot q_j < 0$), kJ/mol; direct Coulomb, no distance cutoff |
| G | Electrostatic repulsive energy | Like-charge residue pairs ($q_i \cdot q_j > 0$), kJ/mol; direct Coulomb, no distance cutoff |
| B | Hydrophobicity delta | Absolute difference in hydrophobicity between residue pairs within the 10 Å Cα distance cutoff |

All six maps are normalized to [0–255] before combining into the final RGB image.

Example Proteogram v2 (asymmetric):

![](assets/d3kfda_.jpg)

## Getting started with Proteogram v2

This repo uses Python 3.11+.

### System Requirements

**Operating System**
- Ubuntu 22.04.5 LTS or 24.04 LTS

**GPU (required for MD simulations and recommended for training/inference):**
- NVIDIA GPU with CUDA 12 support (e.g. RTX 3090, A100, H100)
- NVIDIA driver ≥ 525.x (required for CUDA 12)
- NVIDIA Container Toolkit (for Docker GPU workflows)

**CPU and RAM:**
- x86-64 CPU (AVX2 recommended for PyTorch performance)
- Minimum 32 GB system RAM; 64 GB recommended for large proteogram datasets (with creation run in parallel)

**Software:**
- Python 3.11+
- CUDA Toolkit 12.x (non-Docker GPU workflows)
- `uv` package manager (see [installation instructions](https://docs.astral.sh/uv/getting-started/installation/))

**MD simulation resource usage by protein length** (GPU-accelerated with an NVIDIA GeForce RTX 4090, Driver Version 535.288.01, CUDA Version 12.2, via OpenMM CUDA platform):

| Protein Length | Approx. Max RAM | Approx. Max GPU VRAM | Approx. Time (Minutes) |
|----------------|---------|--------------|--------------|
| 50 residues   |    900 MB     |      800 MB        |       5       |
| 200 residues   |    1 GB     |      900 MB        |      53        |

### Installing the package

This project uses [uv](https://docs.astral.sh/uv/) as the package manager. To install `uv`, follow the [installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

#### Create a virtual environment

Create and activate a uv-managed virtual environment:
```bash
uv venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate     # On Windows
```

#### CPU-only installation

For systems without a GPU or for development/testing on CPU:
```bash
uv sync
```

This installs OpenMM with CPU-only support.

#### GPU installation (CUDA 12)

For systems with NVIDIA GPUs, install with CUDA 12 support for accelerated MD simulations:
```bash
uv sync --extra cuda12
```

This uses the optional `cuda12` dependencies defined in `pyproject.toml` to install `openmm-cuda-12` and related CUDA packages.

> **Note:** Ensure you have compatible NVIDIA drivers and CUDA 12 toolkit installed. See the [OpenMM documentation](http://docs.openmm.org/latest/userguide/application/01_getting_started.html#installing-openmm) for GPU requirements.

#### [Optional] Adding dependencies

To add a package dependency:
```bash
uv add <packagename>
```

To add a development dependency:
```bash
uv add --dev <packagename>
```

### Set up configuration

1. Copy the example configuration file:
   ```bash
   cp scripts/v2/config.example.yml scripts/v2/config.yml
   ```

2. Edit `scripts/v2/config.yml` to configure:
   - `scope_structures_dir`: Path to your input PDB structure files (here we used SCOPe 2.08 structures)
   - `all_proteograms_dir`: Path where generated proteograms will be saved
   - `limit_file`: (Optional) Path to a file listing specific structures to process (one per line)

### Creating Proteograms

To create Proteograms for your protein structure(s), run the following from the `scripts/v2` folder:
```bash
cd scripts/v2
uv run python create_v2_proteograms.py
```

Optional arguments/flags:
- `--overwrite`: Recreate Proteograms even if they already exist
- `--verbose`: Enable verbose output and logging
- `--save_simulated_pdb`: Save the final MD simulation structure as a PDB file to a subfolder

### Measure similarity of a single domain against a database of Proteograms

> Coming soon

### Running an MD simulation (without creating a Proteogram)

The `NonBondedForceModel` module provides a complete pipeline for running molecular dynamics simulations by themselves and calculating residue-residue interaction energies (Van der Waals and electrostatics). Here's an example:

```python
from proteogram.v2 import NonBondedForceModel
import numpy as np

model = NonBondedForceModel(
    pdb_path='protein.pdb',
    temperature=311.75,   # Kelvin
    pressure=1.0,         # atmospheres
    padding=1.0,          # nanometers (water box padding around protein)
    timestep=2.0,         # femtoseconds
    use_gpu=False,
    output_dir='output'
)

# Full MD pipeline. Returns 4 matrices (vdw/es attractive/repulsive).
vdw_attractive, vdw_repulsive, es_attractive, es_repulsive = model.run_full_pipeline(
    npt_steps=50000,           # steps (50,000 × 2 fs = 100 ps NPT equilibration)
    nvt_steps=50000,           # steps (50,000 × 2 fs = 100 ps NVT equilibration)
    production_steps=500000,   # steps (500,000 × 2 fs = 1 ns production run)
    energy_calc_interval=10000, # steps between energy snapshots (10,000 × 2 fs = 20 ps; 50 frames total)
    return_simulated_pdb=False,
    subtract_solvent_energies=True,
    debug=True
)

print('VdW attractive matrix shape:', vdw_attractive.shape)
print('Electrostatic repulsive matrix shape:', es_repulsive.shape)

model.cleanup()
```

For detailed information on the MD simulation methodology, force calculations, and energy validation, see the [MD Simulation Methodology documentation](docs/md_simulation_methodology.md).

## Scripts reference

Scripts are organized into three subfolders under `scripts/`:

- `scripts/v2/` — Proteogram v2 pipeline (MD-based, recommended)
- `scripts/v1/` — Proteogram v1 pipeline (distance/hydrophobicity/charge maps)
- `scripts/utilities/` — Data preparation utilities

The `v1` and `v2` subfolders have their own `config.yml` (copy from the corresponding `config.example.yml`). The following table lists all scripts, their purpose, and the configuration variables or command-line arguments they use. It is recommended to have the main `data` folder directly under the `scripts` folder for common access.

### `scripts/v2/`

| Script | Purpose | Config Variables (`config.yml`) | Command-Line Arguments |
|--------|---------|--------------------------------|------------------------|
| `v2/create_v2_proteograms.py` | Create proteograms using MD-based nonbonded energy calculations, distances, and hydrophobicity deltas | `limit_file`, `scope_structures_dir`, `all_proteograms_dir` | `--max_workers/-w`, `--overwrite`, `--verbose`, `--debug`, `--memory-efficient`, `--save_simulated_pdb` |
| `v2/measure_similarity_v2.py` | Batch similarity search across all proteograms | `top_k`, `model_file`, `embed_file`, `proteogram_sim_results`, `proteograms_for_sim_dir`, `search_images_dir` | `--exclude_classes/-x`, `--overwrite`, `--embed/--no-embed` |
| `v2/train_multiple_models.py` | Train a from-scratch ConvNet or fine-tune ResNet18 for proteogram classification, with early stopping and per-class evaluation | `training_data_dir`, `num_epochs`, `learning_rate`, `batch_size`, `scope_level`, `model_file_prefix` | `--data_dir/-d` (overrides `training_data_dir`), `--epochs/-e`, `--batch_size/-b`, `--lr/-l`, `--model/-m` (`cnn`\|`resnet18`), `--level` (`class`\|`fold`\|`superfamily`\|`family`, default: `class`), `--tsv_file/-t`, `--patience`, `--val_size`, `--exclude_classes/-x`, `--overwrite/-o`, `--resize`, `--verbose/-v` |
| `v2/evaluate_methods_v2.py` | Evaluate proteogram approach vs GTalign and USalign | `top_k`, `scope_eval_set`, `proteogram_sim_results`, `gtalign_results_dir`, `usalign_results`, `search_images_dir`, `save_bad_searches_dir`, `save_good_searches_dir`, `scope_cla_file`, `scope_des_file`, `scope_hie_file` | `--overwrite`, `--exclude_classes/-x` |
| `v2/create_annotation_file.py` | Generate annotation lookup file from SCOPe/RCSB/PDBe | `limit_file`, `scope_structures_dir`, `annot_file`, `fasta_style_file`, `scope_cla_file`, `scope_des_file`, `scope_hie_file` | None |
| `v2/create_balanced_scope_train_eval_lists.py` | Create balanced train/eval splits from CD-HIT clustered results | None | `--lst-file/-l`, `--lookup-tsv/-t`, `--class-column/-c`, `--n-per-class/-n`, `--eval-fraction/-e`, `--train-output`, `--eval-output`, `--split-train`, `--seed` |

### `scripts/v1/`

| Script | Purpose | Config Variables (`config.yml`) | Command-Line Arguments |
|--------|---------|--------------------------------|------------------------|
| `v1/create_proteograms.py` | Create proteograms using distances, hydrophobicity deltas, and charge maps | `scope_structures_dir`, `eval_proteograms_dir`, `limit_file` | None |
| `v1/measure_similarity_single_domain.py` | Search a single structure against a proteogram database (query path hardcoded in script) | `top_k`, `model_file`, `embed_file`, `embed_file_exists`, `proteogram_sim_results`, `proteograms_dir_single_search` | None |
| `v1/measure_similarity.py` | Batch similarity search across all proteograms | `top_k`, `model_file`, `embed_file`, `proteogram_sim_results`, `proteograms_for_sim_dir`, `search_images_dir` | None |
| `v1/evaluate_methods.py` | Evaluate proteogram approach vs GTalign and USalign | `top_k`, `scope_eval_set`, `proteogram_sim_results`, `gtalign_results_dir`, `usalign_results`, `search_images_dir`, `save_bad_searches_dir`, `save_good_searches_dir`, `scope_cla_file`, `scope_des_file`, `scope_hie_file` | None |
| `v1/make_training_and_eval_data.py` | Create training/validation datasets with SCOPe annotations | `scope_eval_set`, `scope_structures_dir`, `scope_cla_file`, `scope_des_file`, `scope_hie_file`, `training_structures_dir`, `training_proteograms_dir`, `eval_structures_dir`, `eval_proteograms_dir`, `label_df_out` | None |
| `v1/make_training_data_exclude_eval.py` | Create training data excluding evaluation set proteins | `scope_eval_set`, `scope_structures_dir`, `scope_cla_file`, `scope_des_file`, `scope_hie_file`, `training_structures_dir`, `training_proteograms_dir`, `eval_structures_dir`, `eval_proteograms_dir`, `label_df_out`, `scope_level` | None |

### `scripts/utilities/`

| Script | Purpose | Config Variables | Command-Line Arguments |
|--------|---------|-----------------|------------------------|
| `utilities/copy_structures.py` | Copy structure files filtered by amino acid length | None (hardcoded paths in script) | None |
| `utilities/copy_structures_by_prefix.py` | Copy structure files matching a prefix list from a source to destination directory | None | `--prefix_file/-p`, `--src_dir/-s`, `--dst_dir/-d`, `--overwrite/-o` |
| `utilities/find_structures_in_scope.py` | Find PDB structures present in the SCOPe 2.08 database | None (hardcoded paths in script) | None |
| `utilities/get_structures_scope20840_list.py` | Download and parse PDB structures by chain from SCOPe 2.08 | None (hardcoded paths in script) | None |

> **Note:** Scripts with "None (hardcoded paths in script)" require editing the script directly to set file paths. See `config.example.yml` in the relevant subfolder for descriptions of all configuration variables.

## Workflow for paper where the Proteogram approach was compared to popular methods for structure alignment and search

### Overview of v1 approach

![](assets/Workflow-Structure-Compression.png)

#### Proteogram v1 generation

![](assets/proteogram_generation.png)

Example of resulting top 5 most similar Proteograms using the v1 approach:

![](assets/AF-A0A3M6TU40-F1-model_v4_A_top_sims.jpg)

## References

1. **GTalign** - Margelevicius, M. (2024). GTalign: High-performance protein structure alignment, superposition, and search. *Nature Communications*, 15, 1261. https://doi.org/10.1038/s41467-024-45653-4

2. **US-align** - Zhang, C., Shine, M., Pyle, A.M., & Zhang, Y. (2022). US-align: universal structure alignments of proteins, nucleic acids, and macromolecular complexes. *Nature Methods*, 19, 1109–1115. https://doi.org/10.1038/s41592-022-01585-1

3. **SCOPe 2.08** - Chandonia, J.M., Fox, N.K., & Brenner, S.E. (2017). SCOPe: Manual curation and artifact removal in the Structural Classification of Proteins - extended database. *Journal of Molecular Biology*, 429(3), 348-355. https://doi.org/10.1016/j.jmb.2016.11.023

4. **OpenMM** - Eastman, P., Swails, J., Chodera, J.D., McGibbon, R.T., Zhao, Y., Beauchamp, K.A., Wang, L.P., Simmonett, A.C., Harrigan, M.P., Stern, C.D., Wiewiora, R.P., Brooks, B.R., & Pande, V.S. (2017). OpenMM 7: Rapid development of high performance algorithms for molecular dynamics. *PLOS Computational Biology*, 13(7), e1005659. https://doi.org/10.1371/journal.pcbi.1005659

5. **AMBER ff19SB** - Tian, C., Kasavajhala, K., Belfon, K.A.A., Raguette, L., Huang, H., Migues, A.N., Bickel, J., Wang, Y., Pincay, J., Wu, Q., & Simmerling, C. (2020). ff19SB: Amino-Acid-Specific Protein Backbone Parameters Trained against Quantum Mechanics Energy Surfaces in Solution. *Journal of Chemical Theory and Computation*, 16(1), 528-552. https://doi.org/10.1021/acs.jctc.9b00591

6. **ResNet** - He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 770-778. https://doi.org/10.1109/CVPR.2016.90


# Docker Guide (uv based)

This repo is dockerized to run scripts under `/scripts` (e.g. measure_similarity.py, create_proteograms.py etc.)

## CPU vs GPU containers (important)

Docker images do **not** automatically get GPU access at build time.
GPU access is assigned **when you run the container**.

- Use a CPU image/container for CPU workflows.
- Use a GPU-capable image/container for GPU workflows.
- Start the GPU container with `--gpus ...` (or the equivalent in Compose/Kubernetes).

Also note: `--platform` (for example `linux/amd64` or `linux/arm64`) controls CPU architecture, **not** whether GPU is attached.

## Prerequisites
- Docker installed
- `uv.lock` present in repo root (recommended for reproducible builds)
- For GPU containers: NVIDIA driver + NVIDIA Container Toolkit installed on the host

## `uv.lock` usage (important for Docker builds)

Both Dockerfiles install dependencies with:

```bash
uv sync --active --frozen ...
```

`--frozen` means the build will fail if `uv.lock` is missing or out of sync with
`pyproject.toml`.

### When you change dependencies

If you edit `pyproject.toml` (or dependency extras), regenerate and commit the lockfile:

```bash
uv lock
uv sync --frozen
git add pyproject.toml uv.lock
```

### Common error: lockfile mismatch

If Docker build fails around `uv sync --frozen`, run:

```bash
uv lock
```

Then rebuild the image.

## Supported Docker platforms

- **CPU image (`Dockerfile`)**
  - Intended for standard Linux Docker platforms.
  - Commonly works on: `linux/amd64`, `linux/arm64`.

- **GPU image (`Dockerfile.gpu`)**
  - Intended for Linux hosts with NVIDIA GPU runtime support.
  - Primary supported platform: `linux/amd64`.

### Notes on platform vs GPU

- `--platform` selects CPU architecture (for example `linux/amd64`, `linux/arm64`).
- GPU access is assigned at runtime with `--gpus ...`.
- GPU use also depends on host setup (NVIDIA drivers + NVIDIA Container Toolkit).

---

## Build the Docker image

From the repo root (the folder that contains `Dockerfile`, `pyproject.toml`, `uv.lock`):

```
sudo docker build -t proteogram:dev .
```

For clarity, build/tag CPU and GPU images explicitly:

```bash
sudo docker build -t proteogram:cpu .
```

GPU image (uses `Dockerfile.gpu` and installs `cuda12` extra dependencies via uv):

```bash
sudo docker build -f Dockerfile.gpu -t proteogram:gpu .
```

> `Dockerfile` = CPU image; `Dockerfile.gpu` = GPU-capable Python environment.
> GPU access is still granted only at runtime with `--gpus ...`.

## Verify the image

Verify Python and package import
```
docker run --rm proteogram:dev python -c "import proteogram; print('import ok')"
```

Verify scripts inside the container
```
docker run --rm proteogram:dev python scripts/measure_similarity.py
```

## Run CPU container

Run normally (no GPU flags):

```bash
docker run --rm -it proteogram:cpu bash
```

## Run GPU container

Assign GPU at runtime with Docker's `--gpus` flag:

```bash
docker run --rm --gpus all -it proteogram:gpu bash
```

Use a specific GPU device (example GPU 0 only):

```bash
docker run --rm --gpus '"device=0"' -it proteogram:gpu bash
```

Verify OpenMM can see CUDA platform in GPU container:

```bash
docker run --rm --gpus all proteogram:gpu \
  python -c "from openmm import Platform; print([Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())])"
```

You should see `CUDA` in the printed platform list.

For CPU-only service, use `proteogram:cpu` and omit GPU device reservations.

Interactively login to container and inspect the contents to see expected files.
```
docker run --rm -it proteogram:dev bash
```

### Mount the datasets 
Note: `-v` bind mounts are applied **only at container run time**. The data is
not stored in the image and will not be present unless you start the container
with the `-v` flag.
```
sudo docker run --rm -it \
  -v "$(pwd)/scripts/data/pdbstyle-2.08:/app/scripts/data/pdbstyle-2.08" \
  proteogram:dev \
  bash
```