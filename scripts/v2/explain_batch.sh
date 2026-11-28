#!/bin/bash
# Batch run of the similarity-map explanation for one trained model.
#
# 1. select_explain_pairs.py re-embeds the model's eval corpus, checks the
#    ranking against the eval run, and samples query -> target pairs by stratum
#    (family top-1, remote superfamily, remote fold incl. every query all three
#    aligners miss, remote-fold misses, novel superfamily). Skipped when
#    pairs.tsv already exists (FORCE_SELECT=1 to redo).
# 2. explain_similarity_maps.py splits each pair's similarity over regions,
#    checks it against USalign, and (FAITHFULNESS=1) runs deletion/insertion.
#    Figures and PyMOL renders for the first MAX_FIGURES pairs; the aligner-miss
#    fold cases are listed first, so they get them.
#
# Run from anywhere; paths are relative to scripts/v2. Options (environment):
#   MODEL=...          checkpoint (default: run 3, seed 0)
#   EVAL_DIR=...       evaluate_methods_v2.py run for that model (default: run 3's)
#   TAG=run3           names the output folders
#   N=100              pairs per stratum (family top-1 gets N/2)
#   FAITHFULNESS=1     deletion/insertion test (0 to skip; ~3x faster)
#   CHANNELS=1         split every map over the six channels (exact Shapley,
#                      128 extra forward passes per pair; 0 to skip)
#   MAX_FIGURES=100    pairs with figure.png and PyMOL renders
#   MAX_PAIRS=         cap on pairs explained (e.g. 6 for a smoke test)
#   DEVICE=cuda|cpu    default: cuda if available
#   USALIGN=USalign    path or command
#   PYTHON=python
#
# Examples:
#   bash explain_batch.sh                                  # run 3, all strata
#   MAX_PAIRS=6 bash explain_batch.sh                      # smoke test
#   MODEL=../data/.../proteogram_model_..._seed1_..._acc68.4.pt \
#     EVAL_DIR=../data/.../eval_runs/eval_20261126-002519_superfamily_k10 TAG=run36 bash explain_batch.sh
#
# Outputs: ../data/.../evaluation/explain_runs/pairs_<TAG>/ (pairs.tsv, ranking check)
#          ../data/.../evaluation/explain_runs/explain_<TAG>_<time>/ (summary.tsv, report.txt, pairs/)
# Note: each structure comes from the crystal .ent when it is present and
# matches the proteogram, otherwise the CG production structure; summary.tsv
# records which (structure_query / structure_target).

cd "$(dirname "$0")" || exit 1

PYTHON=${PYTHON:-python}
DATA=../data/cdhits_protein_lists/scope2.08_all_40_pdb_filtered_cg
MODEL=${MODEL:-$DATA/proteogram_model_resnet18_lr0.001_bs32_e42_seed0_max_image_size500_input500_resize_min_train20_level-superfamily_lossce_aux-fold1-class0.25_es-rsf10_acc67.6.pt}
EVAL_DIR=${EVAL_DIR:-$DATA/evaluation/eval_runs/eval_20261125-225516_superfamily_k10}
TAG=${TAG:-run3}
N=${N:-100}
FAITHFULNESS=${FAITHFULNESS:-1}
CHANNELS=${CHANNELS:-1}
MAX_FIGURES=${MAX_FIGURES:-100}
USALIGN=${USALIGN:-USalign}
RUNS_DIR=$DATA/evaluation/explain_runs
PAIRS_DIR=$RUNS_DIR/pairs_$TAG
OUT=$RUNS_DIR/explain_${TAG}_$(date +%Y%m%d-%H%M%S)

for f in "$MODEL" "$EVAL_DIR"; do
  [ -e "$f" ] || { echo "Missing: $f" >&2; exit 1; }
done
DEV_ARGS=()
[ -n "$DEVICE" ] && DEV_ARGS=(--device "$DEVICE")

if [ ! -f "$PAIRS_DIR/pairs.tsv" ] || [ -n "$FORCE_SELECT" ]; then
  echo "== Selecting pairs -> $PAIRS_DIR"
  "$PYTHON" select_explain_pairs.py --model_file "$MODEL" --eval_dir "$EVAL_DIR" \
      --out_dir "$PAIRS_DIR" --n "$N" "${DEV_ARGS[@]}" || exit 1
else
  echo "== Using existing $PAIRS_DIR/pairs.tsv ($(($(wc -l < "$PAIRS_DIR/pairs.tsv") - 1)) pairs)"
fi

ARGS=(--model_file "$MODEL" --pairs_file "$PAIRS_DIR/pairs.tsv"
      --center_file "$PAIRS_DIR/corpus_mean_embedding.pt"
      --usalign "$USALIGN" --max_figures "$MAX_FIGURES" --out_dir "$OUT" "${DEV_ARGS[@]}")
[ "$FAITHFULNESS" = 1 ] && ARGS+=(--faithfulness)
[ "$CHANNELS" = 1 ] && ARGS+=(--channels)
[ -n "$MAX_PAIRS" ] && ARGS+=(--max_pairs "$MAX_PAIRS")
echo "== Explaining -> $OUT"
"$PYTHON" explain_similarity_maps.py "${ARGS[@]}"
