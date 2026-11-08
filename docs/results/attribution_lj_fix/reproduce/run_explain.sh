#!/usr/bin/env bash
# End-to-end explainability on lj_fix test-set proteograms (run from this folder)
set -u
V=../../v2; PY="uv run --project ../../.. python"
M=$(grep '^model_file' config.yml | cut -d"'" -f2); P=corpus
CATS="same_sf_high_sim diff_fold_high_sim same_sf_low_sim diff_fold_low_sim"
echo "== Grad-CAM (selected pairs)"
for c in $CATS; do while IFS=$'\t' read q t; do
  $PY $V/explain_similarity.py -q $P/$q.jpg -t $P/$t.jpg -o results/gradcam/$c 2>&1 | grep -E "Saved:.*png|SKIP|Error|Traceback|preprocessing"
done < pairs_$c.tsv; done
echo "== Grad-CAM (25 worst queries, top-1 hit)"
$PY $V/explain_similarity.py --batch --n_worst 25 --top_k 1 -o results/gradcam/worst_queries 2>&1 | grep -E "Explaining|Done|Error|Traceback"
for c in $CATS; do
  echo "== energy channels: $c"
  $PY $V/explain_energy_channels.py -m "$M" -p $P --pairs_file pairs_$c.tsv --save_npy -o results/energy/$c 2>&1 | grep -viE "warning|warn\("
  echo "== faithfulness: $c"
  $PY $V/validate_attribution_faithfulness.py -m "$M" -p $P --pairs_file pairs_$c.tsv -o results/faithfulness/$c 2>&1 | grep -viE "warning|warn\(|it/s\]|s/it\]"
done
echo ALL_DONE
