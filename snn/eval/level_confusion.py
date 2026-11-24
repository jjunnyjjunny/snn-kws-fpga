"""
level_confusion.py — stop <-> up confusion of the integer SNN (clip mode) vs input level, for several runs.
All Speech Commands test clips of 'stop' and 'up' (411 + 425) are attenuated by 0 / -12 / -20 / -30 dB (24-bit domain, same
as the dataset gain variants), passed through the integer front-end and run_int (98-frame clip, sum of output integrators).
Rows = true word, columns = predicted class (12). Also reports stop->up and up->stop counts per run and gain.
usage: python snn/eval/level_confusion.py [--runs M3k3e60_U2_q8_kd_s0,M3k3e60_U2c_q8_kd_s0,M3k3e60_U2c_q8_kd_s1] [--gains 0,-12,-20,-30]
"""
import os, sys, json, argparse
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')); sys.path.insert(0, ROOT)
from snn.frontend.frontend_ref import FrontendInt, wav_float_to_pcm24, apply_gain_pcm24
from snn.golden.snn_int_ref import run_int
from snn.data.make_dataset import KEYWORDS, load_item
CLASSES = KEYWORDS + ['silence', 'unknown']

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default='M3k3e60_U2_q8_kd_s0,M3k3e60_U2c_q8_kd_s0,M3k3e60_U2c_q8_kd_s1')
    ap.add_argument('--gains', default='0,-12,-20,-30'); ap.add_argument('--words', default='stop,up')
    ap.add_argument('--out', default=os.path.join(ROOT, 'snn', 'runs', 'board_capture'))
    a = ap.parse_args(); gains = [int(g) for g in a.gains.split(',')]; words = a.words.split(',')
    import torch; torch.backends.cuda.matmul.allow_tf32 = False
    from snn.golden.export_snn_mem import load_run, extract
    fl = json.load(open(os.path.join(ROOT, 'snn', 'data', 'filelist_c12.json')))['test']
    rels, ys = [], []
    for w in words:
        ci = KEYWORDS.index(w); r_ = [r for r, l in fl if l == ci]; rels += r_; ys += [ci] * len(r_)
    ys = np.array(ys); pcm0 = np.stack([wav_float_to_pcm24(load_item(r)) for r in rels]); fe = FrontendInt()
    feats = {}
    for g in gains:
        pcm = apply_gain_pcm24(pcm0, g) if g else pcm0
        feats[g] = np.concatenate([fe.run(pcm[i:i + 100]) for i in range(0, len(pcm), 100)])
    out = {}; L = []
    for run in a.runs.split(','):
        model, cfg, res = load_run(os.path.join(ROOT, 'snn', 'runs', run)); P = extract(model); out[run] = {}
        for g in gains:
            pred = run_int(P, feats[g], trace=False)['vout'].sum(1).argmax(-1)
            cm = {w: [int(((ys == KEYWORDS.index(w)) & (pred == c)).sum()) for c in range(12)] for w in words}
            out[run][g] = cm
            L.append(f"\n**{run}, {g:+d} dB** (행 = 발화, 열 = 판정; n: " + ', '.join(f"{w} {int((ys == KEYWORDS.index(w)).sum())}" for w in words) + ")\n")
            L.append("| 발화 \\ 판정 | " + ' | '.join(CLASSES) + " | 정답률 |"); L.append("|---|" + "---|" * 13)
            for w in words:
                n = sum(cm[w]); L.append(f"| {w} | " + ' | '.join(f"**{v}**" if CLASSES[c] == w else str(v) for c, v in enumerate(cm[w])) + f" | {cm[w][KEYWORDS.index(w)] / n:.3f} |")
            print(f"{run} {g:+d} dB: " + ' ; '.join(f"{w}: " + ', '.join(f"{CLASSES[c]} {v}" for c, v in sorted(enumerate(cm[w]), key=lambda kv: -kv[1]) if v) for w in words), flush=True)
    S = ["| 런 | 감쇠 dB | stop 정답 | stop→up | stop→unknown | stop→silence | up 정답 | up→stop | up→unknown | up→silence |", "|---|---|---|---|---|---|---|---|---|---|"]
    si, ui, uk, sl = KEYWORDS.index('stop'), KEYWORDS.index('up'), 11, 10
    for run in out:
        for g in gains:
            cm = out[run][g]; ns, nu = sum(cm['stop']), sum(cm['up'])
            S.append(f"| {run.replace('M3k3e60_', '').replace('_q8_kd', '')} | {g:+d} | {cm['stop'][si]} ({cm['stop'][si] / ns:.2f}) | {cm['stop'][ui]} | {cm['stop'][uk]} | {cm['stop'][sl]} | "
                     f"{cm['up'][ui]} ({cm['up'][ui] / nu:.2f}) | {cm['up'][si]} | {cm['up'][uk]} | {cm['up'][sl]} |")
    txt = "## 레벨별 stop ↔ up 혼동 (test 클립 전체: stop 411, up 425; 클립 모드, 마진 0)\n\n" + '\n'.join(S) + '\n' + '\n'.join(L) + '\n'
    os.makedirs(a.out, exist_ok=True); open(os.path.join(a.out, 'level_confusion.md'), 'w').write(txt)
    json.dump({r: {str(g): v for g, v in d.items()} for r, d in out.items()}, open(os.path.join(a.out, 'level_confusion.json'), 'w'), indent=1)
    print('\n'.join(S))

if __name__ == '__main__':
    main()
