# Energy-Decomposed Grad-CAM

> **Note:** this document previously also covered a physics-informed ListNet
> ranking loss (`proteogram/v2/ranking_loss.py`). That feature was removed
> from the codebase (see PR history) — its novelty/impact did not justify the
> added training-pipeline complexity. This document has been reduced to the
> energy-decomposed Grad-CAM explainability tool, which is independent of the
> training loss used and still works on any trained Proteogram model
> (including a plain CrossEntropy-only model).

## What This Document Covers

This document describes **Energy-Decomposed Grad-CAM**: an explainability
tool that reveals *which physical force* (Van der Waals, electrostatic, or
geometric distance) drove a similarity prediction between two proteins,
rather than only *where* in the image the model focused.

---

## Background: How Proteogram Works (No Domain Knowledge Required)

### The core idea

A protein is a chain of amino acids folded into a 3D shape. Whether two proteins have similar shapes matters enormously in biology — similar shapes often mean similar functions, and finding similar proteins helps understand disease mechanisms and design drugs.

Traditional tools compare protein shapes by literally trying to superimpose one structure on top of another (like fitting puzzle pieces together). Proteogram takes a completely different approach: it converts each protein into a **picture**, then uses image-recognition AI to find similar pictures.

### What the picture encodes

A proteogram is a square heatmap image. If a protein has N amino acids, the image is N×N pixels. Each pixel at position (row i, column j) encodes the **interaction between amino acid i and amino acid j** — how strongly they attract, repel, or are spaced apart.

The image has 3 colour channels (like any RGB photo), each encoding a different type of physical interaction:

| Channel | Colour | What it measures |
|---------|--------|-----------------|
| Channel 0 | Red (R) | Van der Waals (VdW) energy — the short-range "stickiness" between atoms when they are very close |
| Channel 1 | Green (G) | Electrostatic energy — attraction/repulsion between electrically charged amino acids |
| Channel 2 | Blue (B) | Geometric distance between amino acids / hydrophobicity similarity |

**Van der Waals forces** are what holds the hydrophobic core of a protein together — the oil-like interior that squeezes away from water. **Electrostatic forces** govern interactions between charged residues on the protein surface. **Geometric distance** is the raw spatial separation between amino acid pairs.

### How the AI model learns

A convolutional neural network (CNN) looks at thousands of these proteogram images and learns to produce a compact numerical "fingerprint" (embedding) for each protein. Two proteins with similar fingerprints are predicted to be structurally similar. The cosine similarity between two fingerprints — a number between -1 and 1 — is the similarity score.

---

## Background: What is Grad-CAM?

**Grad-CAM** (Gradient-weighted Class Activation Mapping) is an explanation technique for neural networks. When a model makes a prediction, Grad-CAM asks: "which part of the input image most influenced this prediction?" It does this by flowing the prediction signal backwards through the network and measuring which image regions caused the strongest gradients.

In the original Proteogram Grad-CAM implementation (already present before this work), the output is a single heatmap showing *where* in the proteogram image the model focused — which residue pairs drove the similarity score.

## The New Contribution: Channel Decomposition

The original heatmap answers **WHERE** the model focused. The new decomposed Grad-CAM answers **WHICH FORCE** drove the focus.

Instead of computing one combined heatmap over all three channels, we compute three separate attribution maps — one per input channel — using **Gradient × Input** saliency:

```
attribution_k(i, j) = |∂(cosine_sim) / ∂(input[k, i, j])| × |input[k, i, j]|
```

In plain English: for each residue pair (i, j) and each physical channel k, this measures "how much would the cosine similarity change if I slightly changed the VdW / electrostatic / distance value at this position?"

The multiplication by the actual input value is important — it means we weight the gradient by the magnitude of the signal. A large gradient at a pixel where the energy is near-zero doesn't matter; a large gradient at a pixel with strong VdW energy matters a lot.

### Output

The tool produces a **5-panel figure** per protein pair:

1. Original query proteogram (the input image)
2. Combined Grad-CAM heatmap (WHERE — existing capability)
3. VdW attribution map (Red channel — new)
4. Electrostatic attribution map (Green channel — new)
5. Distance/hydrophobicity attribution map (Blue channel — new)

### The key derived metric: Energy/Distance ratio

From the three channel attributions, we compute:

```
Energy/Distance ratio = (mean VdW attribution + mean Electrostatic attribution) / mean Distance attribution
```

This ratio measures how much the model relies on **physical energy terms** (VdW + electrostatic) relative to **pure geometry** (distance) when assessing similarity. A ratio of 1.0 means equal reliance; 2.0 means the model uses energy terms twice as intensively as distance.

### Files changed

| File | What changed |
|------|-------------|
| `proteogram/v2/gradcam.py` | Added `compute_decomposed()`, `compute_decomposed_from_paths()`, `save_decomposed_figure()`, `save_decomposed_npy()` methods to the `GradCAM` class. Channel labels and colourmap constants added. Fixed `tight_layout` warning by switching to `layout="constrained"`. |
| `proteogram/v2/image_similarity.py` | Added `gradcam_decomposed_similarity()` wrapper method to `Img2Vec`, making decomposed Grad-CAM accessible from the main API. |
| `scripts/v2/explain_energy_channels.py` | **New script.** Handles pair selection (manual via TSV or automatic via USalign categories), runs decomposed Grad-CAM on all pairs, saves figures, computes and saves channel attribution statistics CSV. |

### How to run

```bash
# From scripts/v2/
uv run python explain_energy_channels.py \
    --model_file /path/to/trained_model.pt \
    --proteograms_dir /path/to/proteograms/eval \
    --auto \
    --usalign_results /path/to/usalign_out.tsv \
    --annotations_tsv /path/to/ProteogramData_SCOP_RCSB_PDBe_AnnotationsLookup_AllSCOPe208.tsv \
    --output_dir /path/to/energy_explanations
```

`--model_file` accepts any trained Proteogram model — this tool has no
dependency on how the model was trained (CrossEntropy-only, triplet, or
otherwise).

The `--auto` flag selects three categories of protein pairs automatically:
- `same_fold_high_sim`: same SCOP fold, TM-score ≥ 0.7 (true positives — the model should get these right)
- `same_fold_low_sim`: same fold, TM-score ≤ 0.4 (hard cases — same fold but dissimilar)
- `diff_fold_low_sim`: different fold, TM-score ≤ 0.3 (true negatives — clearly unrelated)

---

## Example Output (Illustrative)

The table below is from an earlier internal run of `explain_energy_channels.py`
against a plain CrossEntropy-trained baseline model, with the default
`--top_k_auto 5` (5 pairs per category — a small, illustrative sample, not a
statistically powered result). It's included here only to show what the
tool's output looks like and how to read it, not as a validated finding:

| Category | VdW (R) | Electrostatic (G) | Distance (B) | E/D ratio |
|---|---|---|---|---|
| diff_fold_low_sim | 0.0099 | 0.0174 | 0.0250 | 1.084 |
| same_fold_low_sim | 0.0161 | 0.0161 | 0.0255 | 1.221 |
| same_fold_high_sim | 0.0266 | 0.0313 | 0.0345 | 1.767 |

**How to read this**: the E/D ratio rises from `diff_fold_low_sim` to
`same_fold_high_sim` — i.e., for pairs the model believes are genuinely
similar, it leans more on the VdW and electrostatic channels relative to raw
distance. Before treating a result like this as a finding rather than an
artifact of a 5-pair sample, increase `--top_k_auto` substantially (20+, or
all available pairs) and check variance across pairs, not just the mean.

---

## Recommended Next Steps

1. **Increase `--top_k_auto`** (try 20+, or use all available auto-selected pairs) for more robust statistics — 5 pairs per category is a small sample and the table above should not be over-interpreted.
2. **Report variance, not just the mean**, per category — `explain_energy_channels.py` already writes a per-pair CSV (`channel_attribution_stats.csv`); compute per-category std/CI from it rather than relying on the printed mean alone.
3. **Pick one case study pair** from `same_fold_high_sim` where VdW dominates and look up the known biology — if it is a membrane protein or a tightly-packed beta-barrel, the VdW dominance is physically expected and provides a compelling narrative example.

---

## Novelty Assessment — ML Research Perspective

### Explainability / XAI

The energy-decomposed attribution uses **Gradient × Input** saliency — one of the simplest gradient-based attribution methods (Baehrens et al. 2010, Simonyan et al. 2013). It is less theoretically rigorous than Integrated Gradients (Sundararajan et al. 2017) or SHAP, but has the advantage of requiring a single backward pass.

**What is new**: The input channels have a physically meaningful, predefined semantics (VdW / electrostatic / distance). Most Grad-CAM applications produce a single spatial heatmap over an undifferentiated RGB image. Here, the three channels are not arbitrary colour channels — they are **physical force channels** — so the per-channel attribution has a direct scientific interpretation that is absent in natural image settings.

The Energy/Distance ratio — `(mean(A_VdW) + mean(A_electrostatic)) / mean(A_distance)` — is a domain-specific aggregation of this attribution, similar in spirit to **feature group importance** (grouping correlated features and measuring their collective effect) but derived from spatial gradient attribution rather than permutation importance.

### Novelty rating and positioning

| Dimension | Rating | Justification |
|-----------|--------|---------------|
| **Method novelty** | Low-moderate | Gradient × Input is a well-established, simple saliency method; the channel decomposition itself is a straightforward per-channel application of it |
| **Application novelty** | Moderate | Attributing similarity predictions to physically named channels (VdW/electrostatic/distance) rather than arbitrary RGB channels gives the output a direct scientific reading not present in typical Grad-CAM use |
| **Engineering contribution** | Moderate | Clean, reusable implementation (`compute_decomposed`, `gradcam_decomposed_similarity`, `explain_energy_channels.py`) that works with any trained model |

### Current limitations in ML terms

| Limitation | Impact | Mitigation |
|------------|--------|-----------|
| 5 pairs per category by default in E/D ratio computation | Underpowered; high variance; not publishable as is | Increase to ≥ 20 pairs or use all available pairs |
| Gradient × Input vs Integrated Gradients | G×I attribution does not satisfy completeness axiom; can miss saturation effects at zero input | Run IG as ablation; expect similar E/D ratio trend |
| No variance/significance reporting in the script's console summary | A single mean per category can look like a clean trend even when driven by 1-2 outlier pairs | Compute per-category std/CI from `channel_attribution_stats.csv` before quoting a ratio |
