#!/bin/bash
# Hyperparameter tuning on the validation set for the min_train20 model.
#
# Trains every combination of the grid below (2 x 4 x 2 x 2 = 32 runs, all
# stopping on validation retrieval) with train_multiple_models_randomized_eval.py,
# one log per run in train_logs/, then prints a table of all runs and a priority
# list of models to evaluate.
#
# Every run logs validation retrieval (remote superfamily capped P@10, averaged
# per superfamily, the selection metric) at the epoch it keeps. Each run keeps
# its best epoch on that metric, so the scores are optimistic in absolute terms
# but comparable with each other. Rank and select on validation only: test
# accuracy is shown for reference, not for choosing.
#
# Run from anywhere; paths are relative to scripts/v2. Options (environment):
#   RUNS="1-16"     run only these run numbers (e.g. "1-32" on one GPU and
#                   "33-64" on another, with CUDA_VISIBLE_DEVICES set per shell)
#   DRY_RUN=1       print the commands without running them
#   FORCE=1         rerun runs whose log already shows a saved model
#   SUMMARY_ONLY=1  skip training and just rebuild the table from the logs
#   TOP_N=5         how many models go on the priority list
#   PYTHON=python   interpreter to use
#
# Example:
#   bash tune_min_train20.sh                     # everything, sequentially
#   RUNS=1-32 CUDA_VISIBLE_DEVICES=0 bash tune_min_train20.sh
#   SUMMARY_ONLY=1 bash tune_min_train20.sh      # table only

cd "$(dirname "$0")" || exit 1

PYTHON=${PYTHON:-python}
TOP_N=${TOP_N:-5}
LOG_DIR=train_logs
LOG_PREFIX=cg_resnet_ce-loss_aux-heads_seed0--tune
SUMMARY=$LOG_DIR/tune_min_train20_summary.tsv

DATA=../data/cdhits_protein_lists/scope2.08_all_40_pdb_filtered_cg
SPLIT=level-superfamily_holdout-family_seed0_max_image_size500_clean_blast0.001_eval_min_train20_ood-superfamily0.3
TRAIN_LIST=$DATA/heldout_family_train_list_${SPLIT}_min_train20.lst
VAL_LIST=$DATA/heldout_family_val_list_${SPLIT}.lst
TEST_LIST=$DATA/heldout_family_test_list_${SPLIT}.lst
TSV=../data/annot_files/ProteogramData_RCSB_PDBe_AnnotationsLookup_scope2.08_all_40_pdb_filtered_cg_Chains-2.tsv

FIXED_ARGS=(
  --model resnet18 --loss ce --level superfamily
  --max_image_size 500 --input_size 500 --resize
  --seed 0 --patience 10 --epochs 200
  --data_dir "$DATA/proteograms_v2/"
  --train_list "$TRAIN_LIST" --val_list "$VAL_LIST" --test_list "$TEST_LIST"
  --tsv_file "$TSV" --exclude_classes "h,i,j,k,l"
  --aux_heads fold,class
  --overwrite --verbose
)

# Grid. Batch size and learning rate change together.
LABEL_SMOOTHING=(0.1 0.0)
AUX_WEIGHTS=(1.0,0.25 1.0,0 2.0,0.25 2.0,0)       # fold,class
EARLY_STOP=(remote_superfamily)                    # add val_loss to compare both
BS_LR=(8:3e-4 32:1e-3)
SAMPLING=(natural balanced)

for f in "$TRAIN_LIST" "$VAL_LIST" "$TEST_LIST" "$TSV"; do
  if [ ! -f "$f" ] && [ -z "$DRY_RUN" ] && [ -z "$SUMMARY_ONLY" ]; then
    echo "Missing input: $f" >&2
    exit 1
  fi
done
mkdir -p "$LOG_DIR"

# RUNS="a-b" (or a single number) limits which run numbers execute.
RUN_FIRST=1
RUN_LAST=999
if [ -n "$RUNS" ]; then
  RUN_FIRST=${RUNS%-*}
  RUN_LAST=${RUNS#*-}
fi

if [ -z "$SUMMARY_ONLY" ]; then
  n=0
  for ls in "${LABEL_SMOOTHING[@]}"; do
  for es in "${EARLY_STOP[@]}"; do
  for aw in "${AUX_WEIGHTS[@]}"; do
  for bslr in "${BS_LR[@]}"; do
  for samp in "${SAMPLING[@]}"; do
    n=$((n + 1))
    [ "$n" -lt "$RUN_FIRST" ] || [ "$n" -gt "$RUN_LAST" ] && continue
    bs=${bslr%%:*}
    lr=${bslr#*:}
    es_tag=valloss
    [ "$es" = remote_superfamily ] && es_tag=rsf10
    tag=$(printf '%02d' "$n")_ls${ls}_fold${aw%,*}-class${aw#*,}_es-${es_tag}_bs${bs}-lr${lr}_${samp}
    log=$LOG_DIR/${LOG_PREFIX}-${tag}.log

    if [ -z "$FORCE" ] && [ -f "$log" ] && grep -q '^Saved model to' "$log"; then
      echo "[$n] done already, skipping ($log)"
      continue
    fi
    cmd=("$PYTHON" train_multiple_models_randomized_eval.py "${FIXED_ARGS[@]}"
         --label_smoothing "$ls" --aux_weights "$aw" --early_stop_metric "$es"
         --batch_size "$bs" --lr "$lr" --sampling "$samp")
    if [ -n "$DRY_RUN" ]; then
      echo "[$n] ${cmd[*]} > $log 2>&1"
      continue
    fi
    echo "[$n] $(date '+%Y-%m-%d %H:%M:%S') start: $tag"
    { echo "TUNE_CONFIG run=$n label_smoothing=$ls aux_weights=$aw early_stop=$es" \
           "batch_size=$bs lr=$lr sampling=$samp"
      echo "COMMAND ${cmd[*]}"; } > "$log"
    "${cmd[@]}" >> "$log" 2>&1
    status=$?
    echo "[$n] $(date '+%Y-%m-%d %H:%M:%S') exit $status: $log"
  done; done; done; done; done
fi

[ -n "$DRY_RUN" ] && exit 0

# ---- Summary table and priority list ----
"$PYTHON" - "$LOG_DIR" "$LOG_PREFIX" "$SUMMARY" "$TOP_N" <<'PY'
import glob, os, re, sys

log_dir, prefix, out, top_n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
rows = []
for path in sorted(glob.glob(os.path.join(log_dir, prefix + '-*.log'))):
    text = open(path, errors='replace').read()
    cfg = re.search(r'^TUNE_CONFIG (.*)$', text, re.M)
    if not cfg:
        continue
    r = dict(kv.split('=', 1) for kv in cfg.group(1).split())
    r['log'] = os.path.basename(path)
    vs = re.search(r'^VAL_SUMMARY (.*)$', text, re.M)
    if vs:
        r.update(dict(kv.split('=', 1) for kv in vs.group(1).split()))
    m = re.search(r'^Saved model to (\S+?\.pt)', text, re.M)
    r['model'] = os.path.basename(m.group(1)) if m else ''
    acc = re.search(r'_acc([\d.]+)\.pt$', r['model'])
    r['test_acc'] = acc.group(1) if acc else ''
    for lvl in ('fold', 'class'):
        h = re.search(rf'^\s+{lvl}\s+\(aux[^)]*\)\s*:\s*([\d.]+) %', text, re.M)
        r[f'test_acc_{lvl}'] = h.group(1) if h else ''
    if r['model']:
        r['status'] = 'done'
    elif 'Traceback' in text:
        r['status'] = 'FAILED'
    else:
        r['status'] = 'running/incomplete'
    rows.append(r)

if not rows:
    sys.exit(f'No tuning logs found in {log_dir}.')

def num(r, k):
    """Float value, or -1 when missing, so unfinished runs sort last."""
    try:
        return float(r.get(k, ''))
    except ValueError:
        return -1.0

ranked = sorted(rows, key=lambda r: (r['status'] != 'done', -num(r, 'val_retrieval'),
                                     -num(r, 'val_retrieval_pooled')))
cols = [('rank', 4), ('run', 3), ('label_smoothing', 6), ('aux_weights', 9), ('early_stop', 18),
        ('batch_size', 4), ('lr', 6), ('sampling', 8), ('epoch', 5), ('val_retrieval', 8),
        ('val_retrieval_pooled', 8), ('val_loss', 8), ('test_acc', 6), ('test_acc_fold', 6),
        ('test_acc_class', 6), ('status', 8)]
heads = {'label_smoothing': 'ls', 'aux_weights': 'fold,cls', 'early_stop': 'early stop',
         'batch_size': 'bs', 'epoch': 'epoch', 'val_retrieval': 'val_rsf', 'val_retrieval_pooled': 'pooled',
         'val_loss': 'val_loss', 'test_acc': 'test%', 'test_acc_fold': 'fold%', 'test_acc_class': 'class%'}

def fmt(r, k):
    v = r.get(k, '')
    if k in ('val_retrieval', 'val_retrieval_pooled', 'val_loss') and v:
        return f'{float(v):.4f}'
    return str(v)

for i, r in enumerate(ranked, 1):
    r['rank'] = str(i) if r['status'] == 'done' and r.get('val_retrieval') else '-'

with open(out, 'w') as f:
    keys = [k for k, _ in cols] + ['model', 'log']
    f.write('\t'.join(keys) + '\n')
    for r in ranked:
        f.write('\t'.join(fmt(r, k) for k in keys) + '\n')

print('\nmin_train20 tuning on validation, ranked by val_rsf (remote superfamily capped P@10, '
      'per-superfamily mean, at the kept epoch)')
print('Reference: the earlier min_train20 model (layer1 frozen, augmentation on) scored 0.722 '
      'on this metric in eval_20261123-160819.\n')
print('  '.join(heads.get(k, k).ljust(w) for k, w in cols))
for r in ranked:
    print('  '.join(fmt(r, k).ljust(w) for k, w in cols))
done = [r for r in ranked if r['rank'] != '-']
print(f'\n{len(done)} of {len(rows)} logged runs finished. Table written to {out}')

if done:
    pick = done[:top_n]
    # Runs that stop on val retrieval keep their best epoch on this same metric,
    # so their val_rsf is optimistic. Make sure the best val-loss run is listed too.
    best_vl = next((r for r in done if r['early_stop'] == 'val_loss'), None)
    if best_vl and best_vl not in pick:
        pick.append(best_vl)
    print('\nPriority list to evaluate (embed the 714 val proteins, run evaluate_methods_v2.py, '
          'compare family MAP@10 too):')
    for i, r in enumerate(pick, 1):
        print(f'  {i}. run {r["run"]}: val_rsf {float(r["val_retrieval"]):.4f} '
              f'(pooled {float(r["val_retrieval_pooled"]):.4f}) | ls {r["label_smoothing"]}, '
              f'fold,class {r["aux_weights"]}, stop on {r["early_stop"]}, bs {r["batch_size"]}, '
              f'lr {r["lr"]}, {r["sampling"]}\n     {r["model"]}')
    stops = {r['early_stop'] for r in done}
    if len(stops) > 1:
        print('\nCaveat: runs that stop on remote_superfamily pick their epoch by this metric, so '
              'their val_rsf is biased upward relative to val_loss runs.')
    else:
        print('\nCaveat: every run picked its epoch by val_rsf, so the scores are optimistic in '
              'absolute terms (do not quote them as results) but comparable with each other.')
    print('Differences under ~0.03 are within noise for 44 scorable superfamilies; rerun the top '
          'two or three with 2 more seeds before choosing.')
PY
