"""
silence_e1_table.py — expected silence (quiet room) behaviour of the integer front-end + VAD for each codec gain
(BOARD_STEP1 §5 variants): noise floor x gain -> non-zero feature bands, E1 = sum of the 40 features per 10 ms frame,
frames above T_off (360) / T_on (600), VAD trigger (N_on = 4 consecutive frames > T_on), T0 front-end gating
(sum|x18| over the 480-sample window < 6694 -> features provably all 0).
Sources: the board quiet capture (snn/data/board_capture/quiet_01.csv, floor -76 dBFS, DC offset about -18 LSB18 (-1,170 LSB24)) as
recorded and with the DC removed (proxy for 'ADC HPF on'), plus synthetic white noise at -76 / -73 / -70 dBFS RMS.
usage: python snn/eval/silence_e1_table.py [--gains 0,6,12,20,26,32,38] [--out snn/runs/board_capture]
"""
import os, sys, json, argparse
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')); sys.path.insert(0, ROOT)
from snn.frontend.frontend_ref import FrontendInt, apply_gain_pcm24
from snn.eval.board_capture_check import read_capture
FS24 = 1 << 23; T_ON, T_OFF, N_ON, T0 = 600, 360, 4, 6694

def analyse(fe, pcm24):
    d = fe.run_debug(pcm24[None]); F = d['feat'][0].astype(np.int64); E = F.sum(1); nz = (F > 0).sum(1)
    x18 = d['frames'][0]                                      # [T, 480] after >>6 (the RTL's 18-bit domain)
    gated = (np.abs(x18).sum(1) < T0)
    run = best = 0
    for v in E:
        run = run + 1 if v > T_ON else 0; best = max(best, run)
    return dict(T=int(len(E)), bands_max=int(nz.max()), bands_mean=float(nz.mean()), E1_max=int(E.max()), E1_p99=float(np.percentile(E, 99)),
                E1_median=float(np.median(E)), frames_over_Toff=int((E > T_OFF).sum()), frames_over_Ton=int((E > T_ON).sum()),
                longest_run_over_Ton=int(best), vad_trigger=int(best >= N_ON), t0_gated_frac=float(gated.mean()),
                rms_dbfs=float(20 * np.log10(max(np.sqrt(np.mean(pcm24.astype(np.float64) ** 2)), 1e-9) / FS24)))

def verdict(r):
    if r['vad_trigger'] or r['E1_p99'] >= T_OFF: return '불합격'
    if r['E1_max'] >= T_OFF / 2: return '주의'
    return '합격'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gains', default='0,6,12,20,26,32,38'); ap.add_argument('--out', default=os.path.join(ROOT, 'snn', 'runs', 'board_capture'))
    ap.add_argument('--quiet', default=os.path.join(ROOT, 'snn', 'data', 'board_capture', 'quiet_01.csv'))
    a = ap.parse_args(); gains = [int(g) for g in a.gains.split(',')]
    fe = FrontendInt(); rng = np.random.default_rng(0); src = {}
    x18, _ = read_capture(a.quiet); q = x18 << 6
    src['quiet_01 그대로 (DC 포함)'] = q
    src['quiet_01 DC 제거 (HPF on 근사)'] = q - int(round(q.mean()))
    for lv in (-76, -73, -70):
        src[f'백색잡음 {lv} dBFS RMS'] = np.round(rng.standard_normal(2 * 16000) * (10 ** (lv / 20)) * FS24).astype(np.int64)
    rows = []
    for name, x in src.items():
        for g in gains:
            r = analyse(fe, apply_gain_pcm24(x, g) if g else x); r.update(source=name, gain_db=g, verdict=verdict(r)); rows.append(r)
            print(f"{name:28s} {g:+3d} dB rms {r['rms_dbfs']:6.1f} dBFS | bands>0 max {r['bands_max']:2d} mean {r['bands_mean']:5.2f} | E1 max {r['E1_max']:5d} p99 {r['E1_p99']:6.0f} med {r['E1_median']:5.0f} | "
                  f">T_off {r['frames_over_Toff']:3d}/{r['T']} >T_on {r['frames_over_Ton']:3d} run {r['longest_run_over_Ton']} VAD {r['vad_trigger']} | T0 게이팅 {r['t0_gated_frac']:.2f} | {r['verdict']}")
    os.makedirs(a.out, exist_ok=True); json.dump(rows, open(os.path.join(a.out, 'silence_e1.json'), 'w'), indent=1)
    L = [f"| 무음 소스 | 게인 dB | RMS dBFS | 0이 아닌 밴드 수 (최대 / 평균) | E1 max / p99 / 중앙값 | 프레임 > T_off {T_OFF} | 프레임 > T_on {T_ON} (최장 연속) | VAD 발동 | T0 게이팅 비율 | 판정 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['source']} | {r['gain_db']:+d} | {r['rms_dbfs']:.1f} | {r['bands_max']} / {r['bands_mean']:.2f} | {r['E1_max']} / {r['E1_p99']:.0f} / {r['E1_median']:.0f} | "
                 f"{r['frames_over_Toff']} / {r['T']} | {r['frames_over_Ton']} ({r['longest_run_over_Ton']}) | {'예' if r['vad_trigger'] else '아니오'} | {r['t0_gated_frac']:.2f} | **{r['verdict']}** |")
    L.append(f"\n판정 기준: 합격 = 무음 E1 최대 < {T_OFF // 2}(T_off의 절반) 이고 VAD 미발동; 주의 = E1 최대 {T_OFF // 2}~{T_OFF} 또는 p99 < {T_OFF}; 불합격 = p99 ≥ T_off {T_OFF} 또는 4프레임 연속 > T_on {T_ON}(VAD 발동).")
    open(os.path.join(a.out, 'silence_e1.md'), 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))

if __name__ == '__main__':
    main()
