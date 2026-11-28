#!/bin/bash
# Search and evaluate every channel-ablation model, then draw the ablation figures.
#
# For each finished model in train_logs/ablation_summary.tsv (written by
# ablation_train.sh) this:
#   1. writes a per-model config (a copy of BASE_CONFIG, the config the full
#      model's eval used, so the corpus and the GTalign / US-align / Foldseek
#      results are identical) pointing model_file, embed_file,
#      proteogram_sim_results and search_images_dir into the model's own folder,
#      so no shared similarity file is overwritten;
#   2. runs measure_similarity_v2.py --config (the checkpoint meta tells it which
#      channel to blank);
#   3. runs evaluate_methods_v2.py --config with the same --train_list and
#      --query_types as the full model's eval;
#   4. records the eval folder in ablation_runs/index.tsv, together with the
#      full model's existing evals (runs 3, 36, 37) as the reference, and runs
#      plot_ablation.py on it.
# Steps 2-3 are skipped for a model whose eval already exists (FORCE=1 redoes them).
#
# Run from anywhere; paths are relative to scripts/v2. Options (environment):
#   SUMMARY=train_logs/ablation_summary.tsv
#   BASE_CONFIG=...    config to copy (default: run 3's eval copy)
#   REFERENCE="0:eval_... 1:eval_... 2:eval_..."   full-model evals (seed:folder in eval_runs/)
#   BOOTSTRAP=0        1 adds evaluate_methods_v2.py --bootstrap (Proteogram vs aligners;
#                      slow, and plot_ablation.py does not need it)
#   ONLY="distance"    restrict to these dropped channels
#   FORCE=1            redo search + eval for models that already have one
#   NO_PLOT=1          skip plot_ablation.py
#   PYTHON=python
#
# Outputs: ../data/.../evaluation/ablation_runs/
#   drop-<channel>_seed<s>/config.yml, embeddings.pkl, similarity_results.tsv,
#                          search_images/, eval/eval_<time>_superfamily_k10/
#   index.tsv              variant, drop_channels, seed, model, eval_dir (full model rows too)
#   figures/               plot_ablation.py output

cd "$(dirname "$0")" || exit 1

PYTHON=${PYTHON:-python}
DATA=../data/cdhits_protein_lists/scope2.08_all_40_pdb_filtered_cg
EVAL_RUNS=$DATA/evaluation/eval_runs
SUMMARY=${SUMMARY:-train_logs/ablation_summary.tsv}
BASE_CONFIG=${BASE_CONFIG:-$EVAL_RUNS/eval_20261125-225516_superfamily_k10/config.yml}
REFERENCE=${REFERENCE:-"0:eval_20261125-225516_superfamily_k10 1:eval_20261126-002519_superfamily_k10 2:eval_20261126-010935_superfamily_k10"}
SPLIT=level-superfamily_holdout-family_seed0_max_image_size500_clean_blast0.001_eval_min_train20_ood-superfamily0.3
TRAIN_LIST=$DATA/heldout_family_train_list_${SPLIT}_min_train20.lst
QUERY_TYPES=$DATA/heldout_family_query_types_${SPLIT}.tsv
OUT=$DATA/evaluation/ablation_runs
INDEX=$OUT/index.tsv

for f in "$SUMMARY" "$BASE_CONFIG" "$TRAIN_LIST" "$QUERY_TYPES"; do
  [ -f "$f" ] || { echo "Missing: $f" >&2; exit 1; }
done
mkdir -p "$OUT"
EVAL_ARGS=(--train_list "$TRAIN_LIST" --query_types "$QUERY_TYPES")
[ "${BOOTSTRAP:-0}" = 1 ] && EVAL_ARGS+=(--bootstrap)

# latest eval folder inside a directory (empty if none)
latest_eval() { ls -d "$1"/eval_*_k* 2>/dev/null | sort | tail -1; }

printf 'variant\tdrop_channels\tseed\tmodel\teval_dir\n' > "$INDEX.tmp"
for ref in $REFERENCE; do
  seed=${ref%%:*}
  dir=$EVAL_RUNS/${ref#*:}
  if [ -d "$dir" ]; then
    model=$(grep -m1 '^model_file:' "$dir/config.yml" | sed -E "s/^model_file: *'?([^' #]+).*/\1/")
    printf 'full\t\t%s\t%s\t%s\n' "$seed" "$model" "$dir" >> "$INDEX.tmp"
  else
    echo "Reference eval not found: $dir" >&2
  fi
done

# rows: drop_channels seed model (finished models only)
tail -n +2 "$SUMMARY" | awk -F'\t' '$3 == "done" {print $1 "\t" $2 "\t" $11}' |
while IFS=$'\t' read -r ch seed model; do
  if [ -n "$ONLY" ] && ! [[ " $ONLY " == *" $ch "* ]]; then
    continue
  fi
  name=drop-${ch}_seed${seed}
  dir=$OUT/$name
  mkdir -p "$dir/eval"
  if [ ! -f "$model" ]; then
    echo "[$name] model not found: $model" >&2
    continue
  fi
  eval_dir=$(latest_eval "$dir/eval")
  if [ -n "$eval_dir" ] && [ -z "$FORCE" ]; then
    echo "[$name] eval exists, skipping ($eval_dir)"
  else
    "$PYTHON" - "$BASE_CONFIG" "$dir/config.yml" "$model" "$dir" <<'PY' || exit 1
import sys, yaml
base, out, model, d = sys.argv[1:5]
cfg = yaml.safe_load(open(base))
cfg.update(model_file=model, embed_file=f'{d}/embeddings.pkl',
           proteogram_sim_results=f'{d}/similarity_results.tsv',
           search_images_dir=f'{d}/search_images')
with open(out, 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
    echo "[$name] $(date '+%Y-%m-%d %H:%M:%S') search"
    "$PYTHON" measure_similarity_v2.py --config "$dir/config.yml" --overwrite < /dev/null \
        > "$dir/similarity.log" 2>&1 ||
      { echo "[$name] search failed, see $dir/similarity.log" >&2; continue; }
    echo "[$name] $(date '+%Y-%m-%d %H:%M:%S') evaluate"
    "$PYTHON" evaluate_methods_v2.py --config "$dir/config.yml" --out_dir "$dir/eval" \
        "${EVAL_ARGS[@]}" < /dev/null > "$dir/eval.log" 2>&1 ||
      { echo "[$name] eval failed, see $dir/eval.log" >&2; continue; }
    eval_dir=$(latest_eval "$dir/eval")
    echo "[$name] done: $eval_dir"
  fi
  printf 'drop-%s\t%s\t%s\t%s\t%s\n' "$ch" "$ch" "$seed" "$model" "$eval_dir" >> "$INDEX.tmp"
done
mv "$INDEX.tmp" "$INDEX"
echo "Index: $INDEX ($(($(wc -l < "$INDEX") - 1)) evals)"

if [ -z "$NO_PLOT" ]; then
  "$PYTHON" plot_ablation.py --index "$INDEX" --out_dir "$OUT/figures"
fi
