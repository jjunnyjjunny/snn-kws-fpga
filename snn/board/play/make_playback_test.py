"""
make_playback_test.py — reproducible board accuracy test by loudspeaker playback (BOARD_STEP2 §1B).
  1. playback_test.wav   16 kHz mono 16-bit: Speech Commands *test* clips, one per 2.5 s slot (clip starts 0.5 s into the
                         slot), 3 s of digital silence lead-in, 2 s tail. Slot plan (seed 1, shuffled):
                         yes/no/stop/go x 25, up/down/left/right/on/off x 10, unknown x 40, silence x 10  -> 210 slots, 525 s.
  2. playback_answers.csv  slot, slot_start_s, clip_start_s, word, class_id, file   (the answer key for playback_score.py)
  3. cal_tone_1k_-6dBFS.wav (10 s, 1 kHz sine, peak -6 dBFS) and cal_ref_clips.wav (5 reference keyword clips at the
     dataset level, 1 s apart) for the speaker-volume calibration: play, capture with ILA mode 3, measure the peak with
     snn/eval/capture_level.py and adjust distance/volume until the capture peak is -6 ... -10 dBFS (dataset level).
  4. Golden upper bound: the same wav fed digitally (24-bit = int16 << 8) into the integer pipeline (front-end + VAD
     T_on 600/T_off 360/N_on 4/H 30/P 20/L 25-130 + SNN current ROM U2 + segment decision, margin 0) ->
     playback_golden_det.csv (same columns as an ILA det_valid capture, plus 'margin') and playback_golden_result.md
     (scored with playback_score.py). The board, at the same level, should reproduce this.
usage: python snn/board/play/make_playback_test.py [--out snn/board/play] [--run M3k3e60_U2_q8_kd_s0] [--no_golden]
"""
import os, sys, csv, json, argparse
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')); sys.path.insert(0, ROOT)
from snn.data.make_dataset import KEYWORDS, load_item
SR = 16000; CLIP = 16000; CADENCE = 2.5; CLIP_OFFSET = 0.5; LEAD = 3.0; TAIL = 2.0
PLAN = {'yes': 25, 'no': 25, 'stop': 25, 'go': 25, 'up': 10, 'down': 10, 'left': 10, 'right': 10, 'on': 10, 'off': 10, 'unknown': 40, 'silence': 10}
CLASSES = KEYWORDS + ['silence', 'unknown']
PRM = dict(T_on=600, T_off=360, N_on=4, H=30, P=20, L_min=25, L_max=130)

def write_wav(path, x16):
    import soundfile as sf; sf.write(path, x16.astype(np.int16), SR, subtype='PCM_16')

def build(out, seed):
    fl = json.load(open(os.path.join(ROOT, 'snn', 'data', 'filelist_c12.json')))['test']
    rng = np.random.default_rng(seed); slots = []
    for w, n in PLAN.items():
        ci = CLASSES.index(w); pool = [r for r, l in fl if l == ci]; sel = rng.choice(len(pool), n, replace=False)
        slots += [(w, ci, pool[i]) for i in sel]
    order = rng.permutation(len(slots)); slots = [slots[i] for i in order]
    total = int((LEAD + CADENCE * len(slots) + TAIL) * SR); x = np.zeros(total, np.float64); rows = []
    for s, (w, ci, rel) in enumerate(slots):
        t0 = LEAD + s * CADENCE; a = int(round((t0 + CLIP_OFFSET) * SR)); y = load_item(rel)
        x[a:a + CLIP] += y; rows.append(dict(slot=s, slot_start_s=f"{t0:.3f}", clip_start_s=f"{t0 + CLIP_OFFSET:.3f}", word=w, class_id=ci, file=rel))
    x16 = np.clip(np.round(x * 32768.0), -32768, 32767)
    write_wav(os.path.join(out, 'playback_test.wav'), x16)
    with open(os.path.join(out, 'playback_answers.csv'), 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    # calibration files
    t = np.arange(10 * SR) / SR; tone = np.sin(2 * np.pi * 1000 * t) * 10 ** (-6 / 20) * 32767
    write_wav(os.path.join(out, 'cal_tone_1k_-6dBFS.wav'), np.round(tone))
    # reference clips: 'yes' test clips whose peak is within 1 dB of the keyword-clip mean (so the reference is representative)
    cand = [r for r, l in fl if l == CLASSES.index('yes')]; kw_mean = np.mean([20 * np.log10(max(np.abs(load_item(rel)).max(), 1e-9)) for w, ci, rel in slots if w in KEYWORDS])
    refs = [r for r in cand if abs(20 * np.log10(max(np.abs(load_item(r)).max(), 1e-9)) - kw_mean) <= 1.0][:5]; z = np.zeros(int((1 + 2 * 5) * SR)); pk = []
    for i, r in enumerate(refs):
        y = load_item(r); a = int((1 + 2 * i) * SR); z[a:a + CLIP] = y; pk.append(20 * np.log10(max(np.abs(y).max(), 1e-9)))
    write_wav(os.path.join(out, 'cal_ref_clips.wav'), np.clip(np.round(z * 32768.0), -32768, 32767))
    peak_all = 20 * np.log10(np.abs(x16).max() / 32768.0)
    kw_pk = [20 * np.log10(max(np.abs(load_item(rel)).max(), 1e-9)) for w, ci, rel in slots if w in KEYWORDS]
    json.dump(dict(reference_clips=refs, note='cal_ref_clips.wav: 5 yes clips at 1, 3, 5, 7, 9 s, each peak within 1 dB of the keyword clip mean'), open(os.path.join(out, 'cal_ref_clips.json'), 'w'), indent=1)
    info = dict(n_slots=len(slots), duration_s=total / SR, cadence_s=CADENCE, clip_offset_s=CLIP_OFFSET, lead_s=LEAD, seed=seed, plan=PLAN,
                wav_peak_dbfs=float(peak_all), keyword_clip_peak_dbfs_mean=float(np.mean(kw_pk)), keyword_clip_peak_dbfs_p10_p90=[float(np.percentile(kw_pk, 10)), float(np.percentile(kw_pk, 90))],
                cal_ref_clip_peaks_dbfs=[float(v) for v in pk])
    json.dump(info, open(os.path.join(out, 'playback_info.json'), 'w'), indent=1)
    print(f"playback_test.wav: {len(slots)} slots, {total / SR:.1f} s, wav peak {peak_all:.1f} dBFS, keyword clip peak mean {np.mean(kw_pk):.1f} dBFS "
          f"(p10 {np.percentile(kw_pk, 10):.1f}, p90 {np.percentile(kw_pk, 90):.1f}); cal_ref_clips peaks " + ', '.join(f"{v:.1f}" for v in pk))
    return x16, rows, info

def golden(out, x16, run):
    """digital upper bound: wav -> 24-bit -> integer front-end (chunked) -> VAD + SNN + decision, margin 0 -> ILA-like det CSV."""
    import torch; torch.backends.cuda.matmul.allow_tf32 = False
    from snn.golden.export_snn_mem import load_run, extract
    from snn.frontend.frontend_ref import FrontendInt
    from snn.golden.snn_int_ref import SnnStepper
    from snn.eval.vad_eval import VAD, N_KEYWORD
    model, cfg, res = load_run(os.path.join(ROOT, 'snn', 'runs', run)); P = extract(model)
    pcm = x16.astype(np.int64) << 8; fe = FrontendInt(); HOP, WIN = 160, 480
    T = (len(pcm) - WIN) // HOP + 1; CH = 3000; feats = []
    for s in range(0, T, CH):
        n = min(CH, T - s); seg = pcm[s * HOP: s * HOP + (n - 1) * HOP + WIN]; feats.append(fe.run(seg[None])[0])
    feat = np.concatenate(feats); assert feat.shape[0] == T, (feat.shape, T)
    E = feat.astype(np.int64).sum(-1); vad = VAD(**PRM); snn = SnnStepper(P); seg_sum = None; dets = []; fed = 0
    for t in range(T):
        ev = vad.step(int(E[t]))
        if ev is not None and ev[0] == 'start':
            snn.reset(); seg_sum = np.zeros(len(snn.vo), np.int64)
            for tt in range(ev[1], t + 1): seg_sum += snn.step(feat[tt]); fed += 1
        elif ev is not None:
            seg_sum += snn.step(feat[t]); fed += 1
            if ev[1] >= PRM['L_min']:
                o = np.sort(seg_sum)[::-1]; c = int(seg_sum.argmax()); m = int(o[0] - o[1])
                dets.append(dict(t=t, cls=c, margin=m, valid=int(c < N_KEYWORD)))
        elif vad.active:
            seg_sum += snn.step(feat[t]); fed += 1
    # ILA-like CSV: one row per *valid* detection (det_valid pulses only for keyword classes at margin 0 on the board);
    # the 'margin' column lets playback_score.py emulate the margin switch (--margin 2838 ...)
    with open(os.path.join(out, 'playback_golden_det.csv'), 'w', newline='') as f:
        wr = csv.writer(f); wr.writerow(['user_logic_inst/dbg_det_class[3:0]', 'user_logic_inst/dbg_cnt_frames[23:0]', 'user_logic_inst/dbg_live_count[15:0]', 'user_logic_inst/dbg_cnt_snn_frames[23:0]', 'margin'])
        wr.writerow(['Radix - UNSIGNED', 'Radix - UNSIGNED', 'Radix - UNSIGNED', 'Radix - UNSIGNED', 'Radix - UNSIGNED'])
        k = 0
        for d in dets:
            if d['valid']: k += 1; wr.writerow([d['cls'], d['t'], k, fed, d['margin']])
    all_rows = [dict(t_s=d['t'] * 0.01, cls=d['cls'], name=CLASSES[d['cls']], margin=d['margin'], valid=d['valid']) for d in dets]
    json.dump(dict(run=run, prm=PRM, T=T, snn_frames=fed, duty=fed / T, dets=all_rows), open(os.path.join(out, 'playback_golden_all.json'), 'w'), indent=1)
    print(f"golden: {T} frames, SNN fed {fed} ({fed / T:.3f}), segments {len(dets)}, keyword detections {sum(d['valid'] for d in dets)} -> playback_golden_det.csv")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.dirname(os.path.abspath(__file__))); ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--run', default='M3k3e60_U2_q8_kd_s0'); ap.add_argument('--no_golden', action='store_true')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    x16, rows, info = build(a.out, a.seed)
    if not a.no_golden: golden(a.out, x16, a.run)

if __name__ == '__main__':
    main()
