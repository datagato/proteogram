# Proteogram: an image embedding-based search approach to protein structure similarity

## Introduction

Proteogram is a novel approach to protein structure similarity search that represents protein structures as image data, enabling the use of computer vision models for efficient and accurate similarity detection.

### Proteogram v1: Distance, Hydrophobicity, and Charge Maps

The original Proteogram approach creates an NxN 3-channel image representation (where N is the residue length) by stacking three categories of residue-level information:

1. **Alpha-carbon backbone distances** - Pair-wise residue Cα distances (distogram)
2. **Hydrophobicity similarities** - Residue-residue hydrophobicity comparisons
3. **Charge similarities** - Residue-residue charge state comparisons

This representation captures both spatial similarity through distograms and physicochemical properties through hydrophobicity and charge maps. The resulting RGB image is inherently sequence-alignment independent and can be processed by standard computer vision models to generate embedding vectors for cosine-similarity-based search.

### Proteogram v2: Incorporating MD Simulations

Proteogram v2 extends the original approach by incorporating molecular dynamics (MD) simulations to compute physics-based residue-residue interaction energies. Instead of using static distance and property maps, v2 runs a complete MD simulation pipeline using OpenMM with the AMBER ff19SB force field to calculate:

- **Van der Waals energies** - Attractive and repulsive Lennard-Jones interactions
- **Electrostatic energies** - Attractive and repulsive Coulomb interactions

The MD pipeline includes energy minimization, NPT and NVT equilibration, and production dynamics with harmonic restraints on alpha-carbon atoms. The resulting 3-channel data (with 6 attributes total) provides a richer representation of protein structure that accounts for dynamic conformational sampling and explicit solvent effects.

For detailed information on the MD simulation methodology, see the [MD Simulation Methodology documentation](docs/md_simulation_methodology.md).

## Getting started

This repo uses Python 3.11+.

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
   cp scripts/config.example.yml scripts/config.yml
   ```

2. Edit `scripts/config.yml` to configure:
   - `scope_structures_dir`: Path to your input PDB structure files (here we used SCOPe 2.08 structures)
   - `all_proteograms_dir`: Path where generated proteograms will be saved
   - `limit_file`: (Optional) Path to a file listing specific structures to process

### Creating proteograms

To create proteograms for your protein structures, run the following from the `scripts` folder:
```bash
cd scripts
uv run python create_v2_proteograms.py
```

Optional arguments:
- `--overwrite`: Recreate proteograms even if they already exist
- `--verbose`: Enable verbose output and logging
- `--save_simulated_pdb`: Save the final MD simulation structure as a PDB file to a subfolder

### Measure similarity of a single domain to a database of proteograms

To compare a new structure against an existing database of proteograms and retrieve the top-k most similar proteins:

1. Ensure you have a database of pre-computed proteograms (supplied separately or generated using the step above)

2. Run the similarity search from the `scripts` folder:
   ```bash
   cd scripts
   uv run python measure_similarity_single_domain.py
   ```
   
   Modify the script to specify your query structure and the path to the proteogram database.

Example resulting search image (scores and proteogram files are also output):

![example set of 5 search hits](assets/AF-A0A3M6TU40-F1-model_v4_A_top_sims.jpg)

### Running an MD simulation

The `NonBondedForceModel` module provides a complete pipeline for running molecular dynamics simulations and calculating residue-residue interaction energies. Here's an example:

```python
from proteogram.nonbonded_forces import NonBondedForceModel
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

## Scripts Reference

The following table provides an overview of all scripts in the `scripts/` folder, their purpose, and the configuration variables or command-line arguments they use.

| Script | Purpose | Config Variables (`config.yml`) | Command-Line Arguments |
|--------|---------|--------------------------------|------------------------|
| `create_v2_proteograms.py` | Create proteograms using MD-based nonbonded force calculations | `limit_file`, `scope_structures_dir`, `all_proteograms_dir` | `--max_workers`, `--overwrite`, `--verbose`, `--save_simulated_pdb` |
| `create_proteograms.py` | Create proteograms using distance/hydrophobicity/charge maps (v1) | `scope_structures_dir`, `eval_proteograms_dir`, `limit_file` | None |
| `measure_similarity_single_domain.py` | Search a single structure against a proteogram database | `top_k`, `model_file`, `embed_file`, `embed_file_exists`, `proteogram_sim_results`, `proteograms_dir_single_search` | None |
| `measure_similarity.py` | Batch similarity search across all proteograms | `top_k`, `model_file`, `embed_file`, `proteogram_sim_results`, `proteograms_for_sim_dir`, `search_images_dir` | None |
| `train_resnet_model.py` | Fine-tune a ResNet18 model for proteogram classification | `training_data_dir`, `model_file`, `num_epochs`, `learning_rate`, `batch_size`, `pretrained` | None |
| `train_cnn_model.py` | Train an alternative CNN model for proteogram classification | `num_epochs_cnn`, `learning_rate_cnn`, `batch_size_cnn`, `cnn_model_file_prefix` | Uses `argparse` (see script) |
| `evaluate_methods.py` | Evaluate proteogram approach vs GTalign and USalign | `gtalign_results_dir`, `usalign_results`, `save_bad_searches_dir`, `save_good_searches_dir` | None |
| `make_training_and_eval_data.py` | Create training/validation datasets with SCOPe annotations | `scope_eval_set`, `scope_structures_dir`, `scope_cla_file`, `scope_des_file`, `scope_hie_file`, `training_structures_dir`, `training_proteograms_dir`, `eval_structures_dir`, `eval_proteograms_dir`, `label_df_out` | None |
| `make_training_data_exclude_eval.py` | Create training data excluding evaluation set proteins | `scope_eval_set`, `scope_structures_dir`, `scope_cla_file`, `scope_des_file`, `scope_hie_file`, `training_structures_dir`, `training_proteograms_dir`, `eval_structures_dir`, `eval_proteograms_dir`, `label_df_out`, `scope_level` | None |
| `create_annotation_file.py` | Generate annotation lookup file from SCOPe/RCSB/PDBe | `limit_file`, `scope_structures_dir`, `annot_file`, `fasta_style_file`, `scope_cla_file`, `scope_des_file`, `scope_hie_file` | None |
| `find_structures_in_scope.py` | Find PDB structures in SCOPe 2.08 database | None (hardcoded paths in script) | None |
| `get_structures_scope20840_list.py` | Download and parse PDB structures by chain | None (hardcoded paths in script) | None |
| `copy_structures.py` | Copy structure files filtered by amino acid length | None (hardcoded paths in script) | None |

> **Note:** Scripts with "None (hardcoded paths in script)" require editing the script directly to set file paths. See `config.example.yml` for descriptions of all configuration variables.

## Workflow for paper where the proteogram approach was compared to GTalign and USalign


### Overview of v1 approach

![](assets/Workflow-Structure-Compression.png)

### Proteogram v1 generation

![](assets/proteogram_generation.png)

## References

1. **GTalign** - Margelevicius, M. (2024). GTalign: High-performance protein structure alignment, superposition, and search. *Nature Communications*, 15, 1261. https://doi.org/10.1038/s41467-024-45653-4

2. **US-align** - Zhang, C., Shine, M., Pyle, A.M., & Zhang, Y. (2022). US-align: universal structure alignments of proteins, nucleic acids, and macromolecular complexes. *Nature Methods*, 19, 1109–1115. https://doi.org/10.1038/s41592-022-01585-1

3. **SCOPe 2.08** - Chandonia, J.M., Fox, N.K., & Brenner, S.E. (2017). SCOPe: Manual curation and artifact removal in the Structural Classification of Proteins - extended database. *Journal of Molecular Biology*, 429(3), 348-355. https://doi.org/10.1016/j.jmb.2016.11.023

4. **OpenMM** - Eastman, P., Swails, J., Chodera, J.D., McGibbon, R.T., Zhao, Y., Beauchamp, K.A., Wang, L.P., Simmonett, A.C., Harrigan, M.P., Stern, C.D., Wiewiora, R.P., Brooks, B.R., & Pande, V.S. (2017). OpenMM 7: Rapid development of high performance algorithms for molecular dynamics. *PLOS Computational Biology*, 13(7), e1005659. https://doi.org/10.1371/journal.pcbi.1005659

5. **AMBER ff19SB** - Tian, C., Kasavajhala, K., Belfon, K.A.A., Raguette, L., Huang, H., Migues, A.N., Bickel, J., Wang, Y., Pincay, J., Wu, Q., & Simmerling, C. (2020). ff19SB: Amino-Acid-Specific Protein Backbone Parameters Trained against Quantum Mechanics Energy Surfaces in Solution. *Journal of Chemical Theory and Computation*, 16(1), 528-552. https://doi.org/10.1021/acs.jctc.9b00591

6. **ResNet** - He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 770-778. https://doi.org/10.1109/CVPR.2016.90

7. **Foldseek** - van Kempen, M., Kim, S.S., Tumescheit, C., Mirdita, M., Lee, J., Gilchrist, C.L.M., Söding, J., & Steinegger, M. (2024). Fast and accurate protein structure search with Foldseek. *Nature Biotechnology*, 42, 243–246. https://doi.org/10.1038/s41587-023-01773-0


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