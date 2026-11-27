#!/bin/bash
# Hyperparameter tuning on the validation set for the min_train20 model.
#
# Runs 1-32: every combination of the training grid below (2 x 4 x 2 x 2, all
# stopping on validation retrieval, input 500 with --resize, seed 0).
# Runs 33-37: the input grid. Run 3's settings (ls 0.1, fold 1.0, class 0.25,
# bs 32, lr 1e-3, natural) with pad 500, resize 224 and resize 352, plus run 3
# again with seeds 1 and 2 to measure seed-to-seed spread.
# Runs 38-39: run 35 (resize 352) with seeds 1 and 2.
#
# Each run trains with train_multiple_models_randomized_eval.py, one log per
# run in train_logs/, and the script ends with a table of all runs and a
# priority list of models to evaluate.
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
#   RUNS=33-37 bash tune_min_train20.sh          # input grid + run 3 seeds only
#   RUNS=36-37 bash tune_min_train20.sh          # just the run 3 seeds
#   RUNS=38-39 bash tune_min_train20.sh          # just the run 35 (352) seeds
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
  --max_image_size 500
  --patience 10 --epochs 200
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
# Runs 1-32 all use input 500, --resize, seed 0.

# Input grid (runs 33 on), each as input_size:mode:seed with run 3's settings.
# Pad mode needs input_size >= max_image_size (500).
INPUT_BASE="0.1 remote_superfamily 1.0,0.25 32:1e-3 natural"  # ls es aw bs:lr sampling
INPUT_GRID=(
  500:pad:0
  224:resize:0
  352:resize:0
  500:resize:1      # run 3, seed 1
  500:resize:2      # run 3, seed 2
  352:resize:1      # run 35, seed 1
  352:resize:2      # run 35, seed 2
)

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

# run_one N LS ES AW BS:LR SAMPLING INPUT_SIZE MODE SEED TAG_SUFFIX
run_one() {
  local n=$1 ls=$2 es=$3 aw=$4 bslr=$5 samp=$6 size=$7 mode=$8 seed=$9 suffix=${10}
  if [ "$n" -lt "$RUN_FIRST" ] || [ "$n" -gt "$RUN_LAST" ]; then
    return 0
  fi
  local bs=${bslr%%:*} lr=${bslr#*:} es_tag=valloss
  [ "$es" = remote_superfamily ] && es_tag=rsf10
  local tag
  tag=$(printf '%02d' "$n")_ls${ls}_fold${aw%,*}-class${aw#*,}_es-${es_tag}_bs${bs}-lr${lr}_${samp}${suffix}
  local log=$LOG_DIR/${LOG_PREFIX}-${tag}.log

  if [ -z "$FORCE" ] && [ -f "$log" ] && grep -q '^Saved model to' "$log"; then
    echo "[$n] done already, skipping ($log)"
    return 0
  fi
  local size_args=(--input_size "$size" --seed "$seed")
  [ "$mode" = resize ] && size_args+=(--resize)
  # -u (unbuffered) keeps stderr warnings from landing mid-line in the log.
  local cmd=("$PYTHON" -u train_multiple_models_randomized_eval.py "${FIXED_ARGS[@]}" "${size_args[@]}"
             --label_smoothing "$ls" --aux_weights "$aw" --early_stop_metric "$es"
             --batch_size "$bs" --lr "$lr" --sampling "$samp")
  if [ -n "$DRY_RUN" ]; then
    echo "[$n] ${cmd[*]} > $log 2>&1"
    return 0
  fi
  echo "[$n] $(date '+%Y-%m-%d %H:%M:%S') start: $tag"
  { echo "TUNE_CONFIG run=$n label_smoothing=$ls aux_weights=$aw early_stop=$es" \
         "batch_size=$bs lr=$lr sampling=$samp input_size=$size mode=$mode seed=$seed"
    echo "COMMAND ${cmd[*]}"; } > "$log"
  "${cmd[@]}" >> "$log" 2>&1
  echo "[$n] $(date '+%Y-%m-%d %H:%M:%S') exit $?: $log"
}

if [ -z "$SUMMARY_ONLY" ]; then
  # Runs 1-32. Their log names have no input/seed suffix, so existing logs still match.
  n=0
  for ls in "${LABEL_SMOOTHING[@]}"; do
  for es in "${EARLY_STOP[@]}"; do
  for aw in "${AUX_WEIGHTS[@]}"; do
  for bslr in "${BS_LR[@]}"; do
  for samp in "${SAMPLING[@]}"; do
    n=$((n + 1))
    run_one "$n" "$ls" "$es" "$aw" "$bslr" "$samp" 500 resize 0 ''
  done; done; done; done; done

  # Runs 33 on: input grid with run 3's settings.
  read -r b_ls b_es b_aw b_bslr b_samp <<< "$INPUT_BASE"
  for spec in "${INPUT_GRID[@]}"; do
    n=$((n + 1))
    IFS=: read -r size mode seed <<< "$spec"
    run_one "$n" "$b_ls" "$b_es" "$b_aw" "$b_bslr" "$b_samp" "$size" "$mode" "$seed" \
            "_input${size}-${mode}_seed${seed}"
  done
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
    # Logs from runs 1-32 predate these three fields; they used 500, resize, seed 0.
    r = {'input_size': '500', 'mode': 'resize', 'seed': '0'}
    r.update(re.findall(r'(\w+)=(\S+)', cfg.group(1)))
    r['log'] = os.path.basename(path)
    vs = re.search(r'^VAL_SUMMARY (.*)$', text, re.M)
    if vs:
        # Read only the four known fields. Older logs were written with stdout
        # buffered, so a stderr warning can start on this same line (e.g.
        # "val_retrieval_pooled=0.792199/home/.../UndefinedMetricWarning ...").
        line = vs.group(1)
        num_re = r'[-+]?(?:\d+(?:\.\d*)?(?:[eE][-+]?\d+)?|nan|inf)|None'
        fields = dict(re.findall(r'\b(epoch|val_loss|val_retrieval_pooled|val_retrieval)='
                                 rf'({num_re})', line))
        r.update(fields)
        if len(fields) < 4 or re.sub(rf'\b\w+=(?:{num_re})', '', line).strip():
            print(f'Note: extra text on the VAL_SUMMARY line of {r["log"]} was ignored; '
                  f'read {fields}', file=sys.stderr)
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
        ('batch_size', 4), ('lr', 6), ('sampling', 8), ('input_size', 5), ('mode', 6), ('seed', 4),
        ('epoch', 5), ('val_retrieval', 8),
        ('val_retrieval_pooled', 8), ('val_loss', 8), ('test_acc', 6), ('test_acc_fold', 6),
        ('test_acc_class', 6), ('status', 8)]
heads = {'label_smoothing': 'ls', 'aux_weights': 'fold,cls', 'early_stop': 'early stop',
         'batch_size': 'bs', 'input_size': 'input', 'epoch': 'epoch', 'val_retrieval': 'val_rsf', 'val_retrieval_pooled': 'pooled',
         'val_loss': 'val_loss', 'test_acc': 'test%', 'test_acc_fold': 'fold%', 'test_acc_class': 'class%'}

def fmt(r, k):
    v = r.get(k, '')
    if k in ('val_retrieval', 'val_retrieval_pooled', 'val_loss') and v:
        try:
            return f'{float(v):.4f}'
        except ValueError:
            return str(v)
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

# Settings run with more than one seed: mean and spread of val_rsf across seeds.
# The spread shows how large a difference between configs has to be to matter.
cfg_keys = ('label_smoothing', 'aux_weights', 'early_stop', 'batch_size', 'lr', 'sampling',
            'input_size', 'mode')
groups = {}
for r in done:
    groups.setdefault(tuple(r.get(k, '') for k in cfg_keys), []).append(r)
multi = {k: v for k, v in groups.items() if len({r['seed'] for r in v}) > 1}
if multi:
    print('\nSeed spread (val_rsf), for settings run with several seeds:')
    for k, v in multi.items():
        vals = [num(r, 'val_retrieval') for r in sorted(v, key=lambda r: r['seed'])]
        mean = sum(vals) / len(vals)
        sd = (sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5
        desc = ', '.join(f'{h} {x}'.strip() for h, x in zip(('ls', 'fold,class', 'stop', 'bs', 'lr', '',
                                                     'input', ''), k) if x)
        print(f'  {desc}: mean {mean:.4f}, sd {sd:.4f}, range {max(vals) - min(vals):.4f} '
              f'over seeds {", ".join(r["seed"] for r in sorted(v, key=lambda r: r["seed"]))} '
              f'(runs {", ".join(r["run"] for r in sorted(v, key=lambda r: r["seed"]))})')

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
              f'lr {r["lr"]}, {r["sampling"]}, input {r["input_size"]} {r["mode"]}, '
              f'seed {r["seed"]}\n     {r["model"]}')
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
