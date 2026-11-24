#!/bin/bash
# adopt_week4a.sh — after finalize_week4a: pick the variant by the val_talk rule (keyword detection >= base - 3 %p,
# lowest false alarms/h, margin 0), require seed 1 to agree on the ordering, then swap the .mem (archive old),
# regenerate golden vectors with the new weights and run the 4 verification steps. Writes snn/runs/adopt_week4a.json.
cd "$(dirname "$0")/../.."
PY=${PY:-python}
until grep -q FINALIZE4A_DONE snn/logs/finalize_week4a.log 2>/dev/null; do sleep 300; done
$PY - <<'PYEOF'
import json, os
R='snn/runs/vad_eval'
def ev(name):
    p=os.path.join(R,name); return json.load(open(p)) if os.path.exists(p) else None
base=ev('eval_M3k3e60_c12_q8_kd_base.json'); b=base['results']['val_talk']['margin_0']
rows=[]
for U in ('U1','U2','U3'):
    for s in (0,1):
        e=ev(f'eval_M3k3e60_{U}_q8_kd_s{s}_{U}.json')
        if e is None: continue
        v=e['results']['val_talk']['margin_0']; rows.append(dict(U=U, seed=s, det=v['det_rate'], fa=v['false_alarms_per_hour'], wrong=v['wrong_rate'], ok=v['det_rate']>=b['det_rate']-0.03))
print('base val_talk det %.3f FA/h %.1f'%(b['det_rate'], b['false_alarms_per_hour']))
for r in rows: print(r)
cand=[r for r in rows if r['seed']==0 and r['ok']]
pick=min(cand, key=lambda r: r['fa']) if cand else None
# seed-1 agreement: the picked variant must also satisfy the rule with seed 1 and have lower FA than base
agree = pick is not None and any(r['U']==pick['U'] and r['seed']==1 and r['ok'] and r['fa'] < b['false_alarms_per_hour'] for r in rows)
res=dict(base=dict(det=b['det_rate'], fa=b['false_alarms_per_hour']), rows=rows, pick=pick, seed1_agree=agree, adopted=(pick['U'] if (pick and agree and pick['fa'] < b['false_alarms_per_hour']) else None))
json.dump(res, open('snn/runs/adopt_week4a.json','w'), indent=1); print('ADOPT', res['adopted'])
PYEOF
U=$($PY -c "import json;print(json.load(open('snn/runs/adopt_week4a.json'))['adopted'] or '')")
if [ -n "$U" ]; then
  RUN=M3k3e60_${U}_q8_kd_s0; D=snn/data/feat_c12_m40_w480_$U.npz
  echo "adopting $RUN"
  # 1-2: numpy golden + regress (archives old m3k3 .mem, re-exports, clip/stream/regress vectors from the new run)
  # (run_regress.sh derives the U-variant dataset path itself)
  GPU=6 bash snn/tb/run_regress.sh $RUN m3k3 240 > snn/logs/adopt_regress.log 2>&1
  # 3: front-end tests are weight-independent; integrated test needs golden streams from the new weights
  P="--T_on 600 --T_off 360 --N_on 4 --H 30 --P 20 --L_min 25 --L_max 130"
  CUDA_VISIBLE_DEVICES="" $PY snn/golden/pipeline_int_ref.py --run snn/runs/$RUN --dump 4 --dur 8 $P --out snn/golden/vectors/pipeline_v3 > snn/logs/adopt_pipeline_golden.log 2>&1
  (cd snn/tb/pipeline && for i in 0 1 2 3; do ./run_pipeline.sh snn/golden/vectors/pipeline_v3/stream_$i 48; done 2>&1 | sed 's|vec=snn/golden/vectors/||' > snn/runs/pipeline_tb_v3.txt)
  for c in board talk quiet; do dur=20; [ $c == quiet ] && dur=60; CUDA_VISIBLE_DEVICES="" $PY snn/golden/pipeline_int_ref.py --run snn/runs/$RUN --dump 1 --dur $dur --cond $c $P --out snn/golden/vectors/activity_$c >> snn/logs/adopt_pipeline_golden.log 2>&1; done
  $PY snn/eval/confusion.py --snn $RUN --data $D > snn/logs/adopt_confusion.log 2>&1
  $PY snn/eval/build_report4a.py; $PY snn/docs/make_figures.py > /dev/null 2>&1
fi
echo ADOPT_DONE
