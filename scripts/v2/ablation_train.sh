#!/bin/bash
# Channel ablation: retrain the final model six times, each time without one
# proteogram channel (leave-one-channel-out).
#
# Every model uses run 3's settings (the final configuration: ResNet18, input
# 500 resized, label smoothing 0.1, aux heads fold 1.0 / class 0.25, bs 32,
# lr 1e-3, natural sampling, early stopping on validation remote superfamily
# capped P@10) and differs only in --drop_channels, which blanks that channel
# (sets it to 0, "no interaction") in its half of every image:
#
#   vdw_attractive   upper R    elec_attractive  lower R
#   vdw_repulsive    upper G    elec_repulsive   lower G
#   distance         upper B    hydrophobicity   lower B
#
# The full model is not retrained: runs 3, 36 and 37 (seeds 0, 1, 2) are the
# reference. Each checkpoint records drop_channels in its meta, so
# measure_similarity_v2.py (and ablation_eval.sh) blank the same channel at
# search time.
#
# One log per model in train_logs/; the script ends with a table of all runs,
# written to train_logs/ablation_summary.tsv, which ablation_eval.sh reads.
#
# Run from anywhere; paths are relative to scripts/v2. Options (environment):
#   SEEDS="0"        seeds to train (e.g. "0 1 2" to match the full model's three)
#   CHANNELS="..."   channels to leave out (default: all six)
#   DRY_RUN=1        print the commands without running them
#   FORCE=1          rerun models whose log already shows a saved model
#   SUMMARY_ONLY=1   skip training and just rebuild the table from the logs
#   PYTHON=python
#
# Examples:
#   bash ablation_train.sh                              # six models, seed 0
#   SEEDS="1 2" bash ablation_train.sh                  # add seeds 1 and 2
#   CHANNELS="hydrophobicity distance" bash ablation_train.sh
#   CUDA_VISIBLE_DEVICES=1 CHANNELS="vdw_attractive vdw_repulsive" bash ablation_train.sh

cd "$(dirname "$0")" || exit 1

PYTHON=${PYTHON:-python}
SEEDS=${SEEDS:-0}
CHANNELS=${CHANNELS:-"vdw_attractive vdw_repulsive distance elec_attractive elec_repulsive hydrophobicity"}
LOG_DIR=train_logs
LOG_PREFIX=cg_resnet_ce-loss_aux-heads--ablation
SUMMARY=$LOG_DIR/ablation_summary.tsv

DATA=../data/cdhits_protein_lists/scope2.08_all_40_pdb_filtered_cg
SPLIT=level-superfamily_holdout-family_seed0_max_image_size500_clean_blast0.001_eval_min_train20_ood-superfamily0.3
TRAIN_LIST=$DATA/heldout_family_train_list_${SPLIT}_min_train20.lst
VAL_LIST=$DATA/heldout_family_val_list_${SPLIT}.lst
TEST_LIST=$DATA/heldout_family_test_list_${SPLIT}.lst
TSV=../data/annot_files/ProteogramData_RCSB_PDBe_AnnotationsLookup_scope2.08_all_40_pdb_filtered_cg_Chains-2.tsv

# Run 3's configuration (see tune_min_train20.sh).
FIXED_ARGS=(
  --model resnet18 --loss ce --level superfamily
  --max_image_size 500 --input_size 500 --resize
  --patience 10 --epochs 200
  --data_dir "$DATA/proteograms_v2/"
  --train_list "$TRAIN_LIST" --val_list "$VAL_LIST" --test_list "$TEST_LIST"
  --tsv_file "$TSV" --exclude_classes "h,i,j,k,l"
  --aux_heads fold,class --aux_weights 1.0,0.25
  --label_smoothing 0.1 --early_stop_metric remote_superfamily
  --batch_size 32 --lr 1e-3 --sampling natural
  --overwrite --verbose
)

if [ -z "$DRY_RUN" ] && [ -z "$SUMMARY_ONLY" ]; then
  for f in "$TRAIN_LIST" "$VAL_LIST" "$TEST_LIST" "$TSV"; do
    [ -f "$f" ] || { echo "Missing input: $f" >&2; exit 1; }
  done
fi
mkdir -p "$LOG_DIR"

if [ -z "$SUMMARY_ONLY" ]; then
  for seed in $SEEDS; do
    for ch in $CHANNELS; do
      log=$LOG_DIR/${LOG_PREFIX}-drop-${ch}_seed${seed}.log
      if [ -z "$FORCE" ] && [ -f "$log" ] && grep -q '^Saved model to' "$log"; then
        echo "[drop $ch, seed $seed] done already, skipping ($log)"
        continue
      fi
      cmd=("$PYTHON" -u train_multiple_models_randomized_eval.py "${FIXED_ARGS[@]}"
           --seed "$seed" --drop_channels "$ch")
      if [ -n "$DRY_RUN" ]; then
        echo "[drop $ch, seed $seed] ${cmd[*]} > $log 2>&1"
        continue
      fi
      echo "[drop $ch, seed $seed] $(date '+%Y-%m-%d %H:%M:%S') start"
      { echo "ABLATION_CONFIG drop_channels=$ch seed=$seed"
        echo "COMMAND ${cmd[*]}"; } > "$log"
      "${cmd[@]}" >> "$log" 2>&1
      echo "[drop $ch, seed $seed] $(date '+%Y-%m-%d %H:%M:%S') exit $?: $log"
    done
  done
fi

[ -n "$DRY_RUN" ] && exit 0

# ---- Summary table ----
"$PYTHON" - "$LOG_DIR" "$LOG_PREFIX" "$SUMMARY" <<'PY'
import glob, os, re, sys

log_dir, prefix, out = sys.argv[1:4]
NUM = r'[-+]?(?:\d+(?:\.\d*)?(?:[eE][-+]?\d+)?|nan|inf)'
rows = []
for path in sorted(glob.glob(os.path.join(log_dir, prefix + '-*.log'))):
    text = open(path, errors='replace').read()
    cfg = re.search(r'^ABLATION_CONFIG (.*)$', text, re.M)
    if not cfg:
        continue
    r = dict(re.findall(r'(\w+)=(\S+)', cfg.group(1)))
    vs = re.search(r'^VAL_SUMMARY (.*)$', text, re.M)
    if vs:
        r.update(re.findall(rf'\b(epoch|val_loss|val_retrieval_pooled|val_retrieval|val_bacc|val_acc)=({NUM})',
                            vs.group(1)))
    m = re.search(r'^Saved model to (\S+?\.pt)', text, re.M)
    r['model'] = m.group(1) if m else ''
    acc = re.search(r'_acc([\d.]+)\.pt$', r['model'])
    r['test_acc'] = acc.group(1) if acc else ''
    r['status'] = 'done' if r['model'] else ('FAILED' if 'Traceback' in text else 'running/incomplete')
    r['log'] = os.path.basename(path)
    rows.append(r)
if not rows:
    sys.exit(f'No ablation logs in {log_dir}.')
keys = ['drop_channels', 'seed', 'status', 'epoch', 'val_retrieval', 'val_retrieval_pooled',
        'val_acc', 'val_bacc', 'val_loss', 'test_acc', 'model', 'log']
rows.sort(key=lambda r: (r.get('seed', ''), r.get('drop_channels', '')))
with open(out, 'w') as f:
    f.write('\t'.join(keys) + '\n')
    for r in rows:
        f.write('\t'.join(str(r.get(k, '')) for k in keys) + '\n')

def fmt(v):
    try:
        return f'{float(v):.4f}'
    except ValueError:
        return v or '-'

print('\nChannel ablation (leave one out). val_rsf = validation remote superfamily capped P@10 at '
      'the kept epoch; for reference the full model (run 3, seed 0) scored 0.7575.')
print(f'{"dropped channel":18s} {"seed":>4s} {"status":10s} {"epoch":>5s} {"val_rsf":>8s} '
      f'{"val_acc":>8s} {"test%":>6s}')
for r in rows:
    print(f'{r.get("drop_channels", ""):18s} {r.get("seed", ""):>4s} {r["status"]:10s} '
          f'{r.get("epoch", "-"):>5s} {fmt(r.get("val_retrieval", "")):>8s} '
          f'{fmt(r.get("val_acc", "")):>8s} {r["test_acc"] or "-":>6s}')
print(f'\nTable written to {out}. Next: bash ablation_eval.sh')
PY
