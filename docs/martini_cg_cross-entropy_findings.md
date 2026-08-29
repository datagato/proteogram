# CG Proteograms + Cross-Entropy Classification — Findings

*2026-08-01 — living doc; experiments in progress*

> This document tracks **cross-entropy (CE) classifiers** whose penultimate features are used for retrieval. Early evidence is that CE-at-a-fine-level is the *stronger* approach for fold/superfamily — this doc is where we explore and confirm this.

## Summary

A ResNet18 **CE classifier trained at the superfamily level**, using its
penultimate 512-dim avgpool features for retrieval, with **`--resize
--input_size 200`** (size-normalization), **beats GTalign, USalign, and Foldseek
at class, fold, and superfamily** — on P@5, MAP@5, and Recall@5 — on the shared
1603-protein test set. Family is essentially tied (the sparse metric on this
set). This is the first Proteogram configuration to win outright at the fine
levels.

Two ingredients made it: (1) the **objective** — a fine-level classifier is
forced to *maximally separate every superfamily*, producing sharply
discriminative features (whereas the triplet loss's graded margins deliberately
keep related folds close, which softened exactly the separation retrieval
needs); and (2) **`--resize`** — size-normalization, ported from the triplet
findings, which lifted CE fold/superfamily by **+0.06 / +0.08** over pad mode
(a bigger gain than it gave the triplet model).

## Setup

- **Data**: SCOPe 2.08, CD-HIT ≤40% sequence identity, classes a–g (h,i,j,k,l
  excluded). CG (Martini) proteograms.
- **Model**: ResNet18, cross-entropy loss (optionally focal), trained to
  classify at `--level <class|fold|superfamily|family>`. For retrieval, Img2Vec
  strips the classification head and uses the **512-dim avgpool** penultimate
  features (no tunable `embed_dim` unless a projection head is added).
- **Metric**: Precision@K (K=5), per SCOPe level, self-hit excluded.
- **Test set**: standardized on the **1603-protein list** from the superfamily
  run (`test_list_for_model__..._level-superfamily_acc93.1.lst`), applied via
  `--test_list` to all CE experiments. It already carries GTalign / USalign /
  Foldseek references, so every CE result stays directly comparable to alignment
  on one fixed set. The non-test remainder (6411) is split 5128 train / 1283 val.
  - **Fold**: robust — the list was filtered at `min_class_size=20` on
    *superfamily*, and folds (coarser) are at least as well-populated, so every
    query has ample same-fold neighbors.
  - **Family**: usable but a *lower-ceiling, noisier* metric on this set — some
    families have few/no same-family neighbors in the corpus, so those queries
    score 0 at family level regardless of model quality. This depresses the
    family average equally for *all* methods (so the comparison stays fair), but
    read family as sparse and lean on class/fold/superfamily as the robust
    signals. (For `--level family` *training*, small families filtered out by
    `min_class_size` may still appear in the test set — fine for retrieval, which
    uses the stripped embedding, not the classifier head; don't over-read the
    family model's training accuracy.)
  

## Headline: best CE model vs. alignment methods

**CE, `--level superfamily`, `--resize --input_size 200` (cutoff 300)** —
Pretrained ResNet18, retrieval on the stripped 512-dim avgpool features. Evaluated on the
shared **1603**-protein test set (same set the alignment methods are scored on).

**Precision@5** — Proteogram wins class / fold / superfamily; ties at family.

| Method | Class | Fold | Superfamily | Family |
|---|---|---|---|---|
| GTalign | 0.939 | 0.861 | 0.815 | **0.359** |
| USalign | 0.929 | 0.866 | 0.834 | 0.353 |
| Foldseek | 0.933 | 0.866 | 0.832 | 0.354 |
| **Proteogram (CE-sfam, resize, 200 grid)** | **0.960** | **0.897** | **0.876** | 0.356 |

**MAP@5** — same story, larger fold/superfamily margins.

| Method | Class | Fold | Superfamily | Family |
|---|---|---|---|---|
| GTalign | 0.933 | 0.877 | 0.837 | **0.428** |
| USalign | 0.922 | 0.884 | 0.859 | 0.425 |
| Foldseek | 0.926 | 0.884 | 0.858 | 0.427 |
| **Proteogram (CE-sfam, resize, 200 grid)** | **0.954** | **0.917** | **0.904** | 0.409 |

**Recall@5** — Proteogram wins fold/superfamily; slightly behind at family.

| Method | Class | Fold | Superfamily | Family |
|---|---|---|---|---|
| GTalign | 0.017 | 0.358 | 0.421 | **0.431** |
| USalign | 0.016 | 0.367 | 0.435 | 0.436 |
| Foldseek | 0.016 | 0.366 | 0.433 | 0.435 |
| **Proteogram (CE-sfam, resize, 200 grid)** | **0.018** | **0.383** | **0.463** | 0.417 |

Margins vs. the best alignment method: P@5 class +0.021, fold +0.031, superfamily
+0.042; family −0.004 (tie). Family is the sparse, lower-ceiling metric on this
set (see Setup) — the expected soft spot, not a real loss.

**Caveat**: this 1603 set is superfamily-stratified and well-populated at
fold/superfamily (it is the `acc93.1` split), so it is a *friendly* set for
demonstrating fold/superfamily retrieval — but friendly *equally* for the
alignment methods (their numbers are high too), so the win is fair on this set.
A harder / differently-composed set could show smaller margins.

### Prior baseline (pad mode) — for reference

Same model at `max_image_size=300`, **pad mode** (grid 300), on the same 1603
set: P@5 class 0.938 / fold 0.833 / superfamily 0.794 / family 0.322 — i.e.
within ~0.03 of alignment but not yet winning. `--resize --input_size 200` added
**+0.06 fold / +0.08 superfamily** on top, which is what tipped it past alignment.

### From-scratch control (no ImageNet) — the representation carries the signal

The best config retrained with `--no-pretrained` (random init, uniform LR, no
freezing) still **ties the alignment methods at fold and superfamily and wins at
class**, on the same 1603 set:

| | class | fold | superfamily | family |
|---|---|---|---|---|
| best (pretrained) | 0.960 | 0.897 | 0.876 | 0.356 |
| **from scratch** | 0.947 | 0.864 | 0.833 | 0.331 |
| best alignment | 0.939 | 0.866 | 0.834 | 0.359 |

A ResNet18 that has *never seen a natural image*, trained only on proteograms,
matches GTalign/USalign/Foldseek at the fine levels (fold 0.864 vs 0.866, sfam
0.833 vs 0.834 — dead even) and beats them at class. So the result is **not**
ImageNet features happening to work on these images: **the proteogram
representation alone gets you to alignment parity, and ImageNet pretraining adds
a modest ~+0.03 fold / +0.04 superfamily on top** that tips it from tied to
winning. (MAP@5 and Recall@5 show the same parity.)

## Experiment log

Held-out test P@5, all on the fixed **1603-protein** list (see Setup). All runs use `--max_image_size 300`.

| Config | Class | Fold | Sfam | Family | Notes |
|---|---|---|---|---|---|
| `--level superfamily`, pad, maxsz 300 | 0.938 | 0.833 | 0.794 | 0.322 | pad-mode baseline at "grid" 300 |
| `--level superfamily` + `--resize --input_size 300` (input size thus matching padded run) | 0.953 | 0.876 | 0.849 | 0.342 | resize at grid 300 — already beats alignment; isolates resize-vs-pad from grid size |
| `--level superfamily` + `--resize --input_size 200` | **0.960** | **0.897** | **0.876** | 0.356 | **BEST — beats all alignment at class/fold/sfam** |
| `--level fold` + `--resize --input_size 200` | 0.957 | 0.898 | 0.784 | 0.316 | best fold retriever, but superfamily drops *below alignment* — see nesting note |
| `--level family` (`--resize --input_size 200`, min_class_size 5) | 0.810 | 0.631 | 0.592 | 0.267 | **COLLAPSE** — worse at *every* level incl. family; superfamily model is the better family retriever |
| `--loss focal` (gamma 2.0) + `--level superfamily` + `--resize --input_size 200` | 0.955 | 0.888 | 0.862 | 0.343 | slightly *below* plain CE at every level — down-weighting easy examples hurts retrieval separation |
| best config, **from scratch** (`--no-pretrained`, no ImageNet) | 0.947 | 0.864 | 0.833 | 0.331 | **ties alignment at fold/sfam, wins class — with random init.** Proteogram carries the signal; ImageNet adds ~+0.03–0.04 |

## Levers to try (ported from the triplet findings)

- **`--resize` — VALIDATED, the win.** `--resize --input_size 200` added +0.06
  fold / +0.08 superfamily over pad mode and tipped CE past alignment (see
  headline). The grid-300 resize row is a clean ablation that separates the two
  effects: **pad→resize at the same grid 300** does most of the work (+0.043 fold
  / +0.055 sfam), and **resizing down to 200** adds a further +0.021 / +0.027 on
  top. So resize (size-normalization) is the dominant lever and a smaller grid
  helps more — normalization beats resolution. Resize-300 already beats alignment
  at class/fold/sfam; resize-200 just beats it by more.
- **Level choice — the key CE knob, and it has a Goldilocks optimum at
  superfamily.** Retrieval quality is *non-monotonic* in training level:

  ```
  class  →  fold  →  SUPERFAMILY  →  family
  (coarse:            (PEAK: best     (collapse:
   under-separates     at all 4        task too sparse
   fine levels)        levels)         to learn from)
  ```

  - **Superfamily is the sweet spot — best retriever at *every* SCOPe level.**
    Fine enough to induce discriminative features, coarse enough (~hundreds of
    ≥20-member classes) to be a learnable classification task.
  - **Nesting (partly):** the superfamily model is *tied* at fold (0.897 ≈ 0.898
    vs the fold model) *and* wins at superfamily (0.876 vs 0.784) — separating
    superfamilies also separates parent folds, not vice versa. The fold model is
    a *specialist* (best at fold, but superfamily 0.784 drops *below* alignment).
  - **Family collapses.** `--level family` is worse at *every* level, family
    included (family 0.267 vs the superfamily model's 0.356). Thousands of
    families with ~5 examples each is a sparse softmax the classifier can't fit,
    so it learns poor features → poor embeddings everywhere. It is not that
    family similarity is un-learnable — a thousand-way, 5-shot *classifier* just
    produces bad features. **The best family retriever is the superfamily
    model.**

  Takeaway: **use `--level superfamily`.** Don't go finer (collapses) or coarser
  (loses the fine levels).
- **`embed_dim` does NOT transfer.** CE retrieval uses the fixed 512-dim
  backbone avgpool; there is no `embed_dim` knob unless a projection head is
  added (a larger change).
- **Objective: plain CE wins; softer objectives lose.** A pattern across three
  losses — **plain CE** (full separation pressure on every class) beats **focal**
  (−0.01 at every level; down-weighting easy examples reduces separation of the
  well-classified points that retrieval still needs spread apart) and beats the
  **triplet graded-margin loss** (much worse). Focal helps *classification*
  imbalance, where easy-example geometry is irrelevant; retrieval cares about the
  whole embedding geometry, so its mechanism backfires. Keep plain CE.

## Methodological notes

- **Self-describing checkpoints now cover CE.** Newly trained CE/focal
  checkpoints record grid (`input_size`), `resize` mode, and residue cutoff in
  their `meta`, so `measure_similarity_v2.py` / `query_similar_proteins.py`
  match search-time preprocessing automatically. **Legacy CE checkpoints**
  (bare `state_dict`, e.g. the `acc93.1` model above) carry no meta → set
  `search_pad_size` / `search_resize` in `config.yml` to eval them correctly.

## Open questions / next steps

- ~~From-scratch control: is the win the proteogram or ImageNet?~~ **Answered
  (see From-scratch control section).** From scratch *ties* alignment at
  fold/superfamily and wins class — the proteogram representation carries the
  signal; ImageNet adds a ~+0.03–0.04 boost that tips it to winning.
- ~~Which training level is best for retrieval?~~ **Answered: superfamily**
  (Goldilocks optimum — see Level choice). fold = specialist; family = collapse.
- Family level: the best family *retriever* is the superfamily model (0.356), not
  the family model. Whether family is improvable *at all* beyond that likely
  needs a different lever than training level — it is the sparse-corpus ceiling
  shared by every method (~0.35), not something more CE-level tuning fixes.
- Does the CG channel limit (distance + physicochemical, weak on fold topology)
  bound CE too? — shared with the triplet doc; SSE/torsion enrichment is the
  representation-side lever, orthogonal to the loss.
