# Global Percentile Normalisation: Raw-Channel Capture Bug and Before/After Validation

## What This Document Covers

`docs/improvements_v2.md` (section 2) proposed global percentile normalisation
to replace per-protein min-max normalisation, with the goal of preserving
inter-protein energy scale in proteogram pixel values. This document covers
two things discovered and built after that design was implemented:

1. **A bug** in how the raw energy matrices feeding `compute_norm_stats.py`
   were captured, which meant the feature could not have been doing what its
   design doc claims.
2. **A validation tool** (`scripts/v2/validate_normalisation.py`) that
   measures, quantitatively, whether global normalisation actually preserves
   inter-protein scale — the claim the original design doc asserted but never
   measured.

No model needed to be retrained to find or fix this — it's entirely in the
data-preparation path (`create_v2_proteograms.py` → `.npy` matrices →
`compute_norm_stats.py` → `norm_stats.json`), before any image or model is
touched.

---

## Background: Why This Matters

Global normalisation only helps if the percentile bounds in `norm_stats.json`
are computed from **real physical-unit energy values** (kJ/mol for the energy
channels, Å for distance). If those bounds are instead computed from data
that has already been rescaled once, the corpus-level percentiles just
describe *that rescaling*, not the underlying physics — and the entire
justification for the feature ("a protein with 4× stronger packing should map
to visibly different pixel values than a weakly-packed one") silently stops
holding.

## The Bug

`ProteogramV2.calculate_proteogram()` (`proteogram/v2/proteogram.py`)
computes six genuinely distinct raw matrices before normalising them:

```python
vdw_e_att, vdw_e_rep, es_e_att, es_e_rep, disto_map  # from the MD pipeline
hydro_map                                             # derived from disto_map
```

These are exactly the values `compute_norm_stats.py` needs. But the method
only ever returned `final_data` — the RGB image *after* normalisation — and
these six arrays were deleted internally once the image was built.

`create_v2_proteograms.py`'s `--save_npy_matrices` flag exists specifically to
produce the `.npy` corpus that `compute_norm_stats.py` reads. Since the raw
arrays weren't accessible, its previous implementation faked them by slicing
the **already-normalised** `final_data` image instead:

```python
# Previous (buggy) implementation
channels = {
    'vdw_attractive':  final_data[:, :, 0],   # R upper triangle
    'vdw_repulsive':   final_data[:, :, 1],   # G upper triangle
    'distance':        final_data[:, :, 2],   # B upper triangle
    'es_attractive':   final_data[:, :, 0],   # R lower triangle (approx)
    'es_repulsive':    final_data[:, :, 1],   # G lower triangle (approx)
    'hydrophobicity':  final_data[:, :, 2],   # B lower triangle (approx)
}
```

Two independent problems with this:

1. **Wrong stage of the pipeline.** On a run without `--global_norm`,
   `final_data` was normalised per-protein (min-max to `[0, 255]`) *before*
   being sliced. `compute_norm_stats.py` then computed 1st/99th percentiles
   over a corpus of already-rescaled `[0, 255]` values, not physical-unit
   energies. Per-protein normalisation destroys inter-protein scale by
   design (every protein is independently stretched to use the full pixel
   range) — so the very information global normalisation is supposed to
   recover was already gone before `compute_norm_stats.py` ever saw the data.
2. **Duplicated channels.** `es_attractive`, `es_repulsive`, and
   `hydrophobicity` were literally copies of the `vdw_attractive`,
   `vdw_repulsive`, and `distance` arrays (marked `# (approx)` in the code) —
   not distinct physical quantities at all.

A code comment directly above this block said *"we skip this step"* — but
the code did not skip it; it proceeded to write the sliced-`final_data`
arrays regardless.

**Net effect:** any `norm_stats.json` built before this fix (and any imagery
generated with `--global_norm` using it) does not reflect what section 2 of
`improvements_v2.md` describes. It needs to be regenerated.

## The Fix

**`proteogram/v2/proteogram.py`** — `calculate_proteogram()` gained a
`return_raw_channels: bool = False` parameter. When set, it captures the six
raw arrays into a dict immediately after they're computed (and before
they're freed), and returns that dict as an additional element in its return
tuple:

```python
raw_channels = None
if return_raw_channels:
    raw_channels = {
        'vdw_attractive':  vdw_e_att,
        'vdw_repulsive':   vdw_e_rep,
        'es_attractive':   es_e_att,
        'es_repulsive':    es_e_rep,
        'distance':        disto_map,
        'hydrophobicity':  hydro_map,
    }
```

Default is `False`, so both existing call sites (`query_similar_proteins.py`,
and `create_v2_proteograms.py` when `--save_npy_matrices` isn't passed) are
unaffected — return-tuple shape is unchanged unless explicitly requested.

**`scripts/v2/create_v2_proteograms.py`** — when `--save_npy_matrices` is
passed, `calculate_proteogram(..., return_raw_channels=True)` is called and
the returned `raw_channels` dict is saved directly to `.npy` files, instead
of slicing `final_data`:

```python
if args.save_npy_matrices and raw_channels is not None:
    npy_dir = os.path.join(proteograms_output_dir, 'energy_matrices')
    os.makedirs(npy_dir, exist_ok=True)
    bname_base = os.path.splitext(os.path.basename(image_file))[0]
    for ch_name, ch_arr in raw_channels.items():
        npy_path = os.path.join(npy_dir, f'{bname_base}_{ch_name}.npy')
        np.save(npy_path, ch_arr.astype('float32'))
```

Now `compute_norm_stats.py` receives what its own docstring always claimed it
would: genuine pre-normalisation physical-unit values.

## The New Validation Tool

`scripts/v2/validate_normalisation.py` measures — quantitatively, on the
actual corpus — whether global normalisation does what section 2.1 of
`improvements_v2.md` claims. It needs no model and no retraining; it only
reads the `.npy` matrices and `norm_stats.json`.

For each of the 6 channels, per protein, it computes:

- **`raw_p99`** — the 99th percentile of that protein's own raw (physical-unit)
  values. A cheap, always-available proxy for "how energetic is this protein
  in this channel."
- **`perprotein_mean`** / **`global_mean`** — the mean normalised pixel value
  under each strategy.

From those, it reports per channel:

| Metric | What it tells you | Expected direction |
|---|---|---|
| Inter-protein **variance** of mean pixel value, global vs. per-protein (reported as a ratio) | Does global normalisation actually retain more between-protein signal than per-protein min-max destroys? | Ratio should be **> 1**; the higher, the more scale information global normalisation is preserving. |
| **Correlation** (`raw_p99` vs. normalised mean), per-protein strategy | Sanity check that per-protein normalisation destroys scale by design | Should be **~0**. A strong correlation here would be surprising and worth investigating. |
| **Correlation** (`raw_p99` vs. normalised mean), global strategy | Does the global mapping still track true physical scale? | Should be **close to 1**. A low value flags a broken or corpus-mismatched `norm_stats.json`. |
| **Saturation rate** (fraction of pixels clipped to 0 or 255 under global bounds) | Are the 1st/99th percentile bounds appropriate for this corpus? | Should be low (rule of thumb: **< 5–10%**). Higher means bounds are too tight — widen `--low_pct`/`--high_pct` in `compute_norm_stats.py`. |

It also writes a per-protein CSV and a scatter plot (`raw_p99` vs. normalised
mean, one panel per strategy per channel) for visual inspection.

This tool was sanity-checked against a synthetic corpus with a known,
injected per-protein energy scale (not committed to the repo) before being
used on real data — on that synthetic corpus it correctly reported a ~92×
variance ratio and a ~0.99 vs. ~-0.19 correlation for global vs. per-protein,
confirming the metrics discriminate real preserved-scale signal from noise
before being trusted on real proteograms.

---

## Steps to Run: Generating Before/After Impact Numbers

All commands run from `scripts/v2/`.

### Step 1 — Generate raw `.npy` matrices (post-fix)

Run (or re-run) proteogram creation with `--save_npy_matrices` on a
representative sample of the corpus. This uses per-protein normalisation for
the JPGs (the default), but now also writes the *real* raw matrices needed
for step 2:

```bash
uv run python create_v2_proteograms.py --save_npy_matrices
```

If a `norm_stats.json` or any `--global_norm` proteograms already exist from
before this fix, they were built on corrupted raw data — regenerate the
`.npy` matrices and `norm_stats.json` before trusting anything downstream.

### Step 2 — Compute global percentile bounds

```bash
uv run python compute_norm_stats.py \
    --npy_dir /path/to/proteograms/energy_matrices \
    --out_file /path/to/norm_stats.json \
    --low_pct 1.0 \
    --high_pct 99.0 \
    --max_samples 5000
```

### Step 3 — Measure the before/after impact (no retraining required)

```bash
uv run python validate_normalisation.py \
    --npy_dir /path/to/proteograms/energy_matrices \
    --norm_stats_file /path/to/norm_stats.json \
    --output_dir /path/to/normalisation_validation
```

Read the console table (or `normalisation_summary.csv`): confirm the
variance ratio is meaningfully `> 1`, the global correlation is close to 1,
per-protein correlation is close to 0, and saturation is low, **for every
channel** — including `es_attractive`, `es_repulsive`, and `hydrophobicity`,
since those are exactly the three channels the old buggy code silently
duplicated from others. If any channel looks like an outlier or shows a weak
global correlation, check `--low_pct`/`--high_pct` or the `.npy` sample size
before proceeding.

### Step 4 (optional, expensive) — Downstream task validation

The steps above validate the *data-preparation* claim in isolation. To
validate the *downstream* claim (does this actually improve the model), the
proteogram images themselves must be regenerated with `--global_norm` and a
second model trained on them, then both models compared with the existing
eval pipeline:

```bash
# Regenerate proteogram JPGs using global normalisation
uv run python create_v2_proteograms.py \
    --global_norm \
    --norm_stats_file /path/to/norm_stats.json \
    --overwrite

# Train a model on the globally-normalised proteograms with the SAME
# architecture, epochs, and learning rate as the existing per-protein-
# normalised baseline model — normalisation strategy must be the only
# variable that changes between the two runs, or the comparison is
# confounded and any MAP@K difference can't be attributed to normalisation.
uv run python train_multiple_models.py \
    --model resnet18 --level fold \
    --epochs <same as baseline> --batch_size <same as baseline> --lr <same as baseline>

# Then run the existing eval pipeline for each model in turn (point
# config.yml's model_file / proteogram_sim_results at each) and compare:
uv run python measure_similarity_v2.py --overwrite
uv run python evaluate_methods_v2.py --overwrite
```

Compare Precision@K / MAP@K / Recall@K between the per-protein-normalised
model and the global-normalised model at all four SCOP levels.
