"""
make_dataset.py — build integer log-mel feature datasets with the bit-exact front-end.

Outputs snn/data/feat_c{10|12}_m{n_mels}_w{win}[_tag].npz with
  X_train uint8 [N,T,M], y_train, copy_train (0 = clean, 1.. = augmented copies)
  X_val, y_val, X_test, y_test
  X_test_g{-20,-10,+6,+10} : test set with input gain applied (24-bit saturation) for robustness
  meta (json string): config, class names, counts
and snn/data/filelist_c{10|12}.json (deterministic split, incl. sampled unknown/silence).

Classes: 10 keywords (+ 'silence' = idx 10, 'unknown' = idx 11 when --classes 12).
Split: official validation_list.txt / testing_list.txt; unknown words use the same lists;
silence crops: background files, time-disjoint regions train [0,.8) / val [.8,.9) / test [.9,1).

Augmentation (train copies >= 1): time shift U(-100,+100) ms, background noise mix
(p=0.8, volume U(0,0.1) of a train-region noise crop), random gain U(-20,+10) dB with
24-bit saturation (disabled with --no_gain_aug).

usage: python snn/data/make_dataset.py --classes 12 --n_mels 40 --win 480 --aug_copies 2
"""
import os, sys, json, argparse, time
import numpy as np
import soundfile as sf
from multiprocessing import Pool
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
from snn.frontend import fe_config as C
from snn.frontend.frontend_ref import FrontendInt, wav_float_to_pcm24, apply_gain_pcm24
from scipy.signal import butter, lfilter, fftconvolve

DATA = os.path.join(ROOT, 'data', 'speech_commands_v0.02')
OUT_DIR = os.path.join(ROOT, 'snn', 'data')
KEYWORDS = ['yes', 'no', 'stop', 'go', 'up', 'down', 'left', 'right', 'on', 'off']
SILENCE, UNKNOWN = 'silence', 'unknown'
CLIP = 16000
SEED = 1234

# --------------------------------------------------------------------------- file lists
def build_filelist(n_classes, unknown_mult=1.0, silence_mult=1.0, silence_lowlevel=False, base=None):
    """base: existing filelist whose val/test are kept verbatim (only train is re-sampled with the multipliers)."""
    rng = np.random.default_rng(SEED + (1 if (unknown_mult != 1.0 or silence_mult != 1.0 or silence_lowlevel) else 0))
    val = set(l.strip() for l in open(os.path.join(DATA, 'validation_list.txt')) if l.strip())
    test = set(l.strip() for l in open(os.path.join(DATA, 'testing_list.txt')) if l.strip())
    words = sorted(d for d in os.listdir(DATA) if os.path.isdir(os.path.join(DATA, d)) and not d.startswith('_'))
    splits = {'train': [], 'val': [], 'test': []}
    def split_of(rel):
        return 'val' if rel in val else 'test' if rel in test else 'train'
    kw_count = {s: 0 for s in splits}
    others = {s: [] for s in splits}
    for w in words:
        for fn in sorted(os.listdir(os.path.join(DATA, w))):
            if not fn.endswith('.wav'):
                continue
            rel = f"{w}/{fn}"
            s = split_of(rel)
            if w in KEYWORDS:
                splits[s].append((rel, KEYWORDS.index(w)))
                kw_count[s] += 1
            else:
                others[s].append(rel)
    if n_classes == 12:
        for s in splits:
            if base is not None and s != 'train':
                splits[s] = [tuple(x) for x in base[s]]; continue
            n = int(round(kw_count[s] / 10))          # one class worth of unknown and silence
            n_unk = int(round(n * unknown_mult)) if s == 'train' else n
            n_sil = int(round(n * silence_mult)) if s == 'train' else n
            idx = rng.choice(len(others[s]), n_unk, replace=False)
            splits[s] += [(others[s][i], 11) for i in sorted(idx)]
            # silence: (file, start_sample, volume) crops from time-disjoint regions
            noise_files = sorted(f for f in os.listdir(os.path.join(DATA, '_background_noise_')) if f.endswith('.wav'))
            lo, hi = {'train': (0.0, 0.8), 'val': (0.8, 0.9), 'test': (0.9, 1.0)}[s]
            for i in range(n_sil):
                f = noise_files[i % len(noise_files)]
                info = sf.info(os.path.join(DATA, '_background_noise_', f))
                a, b = int(lo * info.frames), int(hi * info.frames) - CLIP
                start = int(rng.integers(a, max(a + 1, b)))
                if i % 20 == 0: vol = 0.0                                                   # 5 % pure digital silence
                elif silence_lowlevel and s == 'train' and (i % 2 == 1): vol = float(rng.uniform(0.0, 0.1))   # 50 % low-level (<= -20 dBFS)
                else: vol = float(rng.uniform(0.0, 1.0))
                splits[s].append((f"_background_noise_/{f}|{start}|{vol:.4f}", 10))
    for s in splits:
        rng.shuffle(splits[s])
    return splits

# --------------------------------------------------------------------------- loading
_noise_cache = {}
def load_noise(name):
    if name not in _noise_cache:
        x, sr = sf.read(os.path.join(DATA, '_background_noise_', name), dtype='float32')
        _noise_cache[name] = x
    return _noise_cache[name]

def load_item(rel):
    """-> float32 [16000] in [-1,1)"""
    if rel.startswith('_background_noise_'):
        f, start, vol = rel.split('|')
        x = load_noise(os.path.basename(f))[int(start): int(start) + CLIP] * float(vol)
    else:
        x, sr = sf.read(os.path.join(DATA, rel), dtype='float32')
        assert sr == 16000
    x = np.pad(x, (0, max(0, CLIP - len(x))))[:CLIP]
    return x.astype(np.float32)

def channel_aug(y, rng):
    """4-B input-channel robustness (board: mic/codec path level -15..-35 dB vs the dataset, noise floor about -73 dBFS,
    unknown mic response). y: float64 [16000] in 24-bit units. Applied before the gain stage:
      p 0.75 : random response = 2nd-order low-pass (cut-off U(2.5, 7) kHz)  or  high shelf (-U(0,12) dB above U(1.5, 4) kHz)
      p 0.30 : weak reverberation (exponential-decay noise IR, RT60 U(0.1, 0.3) s, wet/dry U(0.1, 0.3))"""
    r = rng.random()
    if r < 0.375:
        b, a_ = butter(2, float(rng.uniform(2500, 7000)) / (SR_ / 2)); y = lfilter(b, a_, y)
    elif r < 0.75:
        b, a_ = butter(2, float(rng.uniform(1500, 4000)) / (SR_ / 2)); lo = lfilter(b, a_, y)
        y = lo + 10 ** (-float(rng.uniform(0, 12)) / 20) * (y - lo)
    if rng.random() < 0.3:
        n = 3200; t = np.arange(n) / SR_; tau = float(rng.uniform(0.1, 0.3)) / 6.9
        ir = rng.standard_normal(n) * np.exp(-t / tau); ir /= np.sqrt((ir ** 2).sum())
        y = y + float(rng.uniform(0.1, 0.3)) * fftconvolve(y, ir)[:len(y)]
    return y
SR_ = 16000
def noise_floor(y, rng, lo_db=-80.0, hi_db=-62.0):
    """white noise at RMS U(lo, hi) dBFS (board quiet-room floor measured at about -73 dBFS), added after the gain."""
    return y + rng.standard_normal(len(y)) * (10 ** (float(rng.uniform(lo_db, hi_db)) / 20) * (1 << 23))

def augment(pcm, rng, noise_pool, gain_aug=True, chan=False):
    """pcm: int64 [16000] 24-bit. returns augmented int64 24-bit."""
    # time shift
    sh = int(rng.integers(-1600, 1601))
    y = np.zeros_like(pcm)
    if sh >= 0:
        y[sh:] = pcm[:CLIP - sh]
    else:
        y[:CLIP + sh] = pcm[-sh:]
    # background noise
    if rng.random() < 0.8:
        nf = noise_pool[int(rng.integers(len(noise_pool)))]
        st = int(rng.integers(0, int(0.8 * len(nf)) - CLIP))
        vol = rng.uniform(0.0, 0.1)
        y = y + np.round(nf[st: st + CLIP].astype(np.float64) * vol * (1 << 23)).astype(np.int64)
    if chan:
        y = np.round(channel_aug(y.astype(np.float64), rng)).astype(np.int64)
    # gain
    if gain_aug:
        y = apply_gain_pcm24(y, rng.uniform(-30.0 if chan else -20.0, 10.0))
    if chan:
        y = np.round(noise_floor(y.astype(np.float64), rng)).astype(np.int64)
    return np.clip(y, -(1 << 23), (1 << 23) - 1)

def test_variant(pcm, kind, rng):
    """fixed test-set variants (no randomness except the noise seed): 'lp4k' = 4th-order 4 kHz low-pass at 0 dB;
    'board' = -15 dB + white noise floor -73 dBFS (the 3 cm board capture condition)."""
    y = pcm.astype(np.float64)
    if kind == 'lp4k':
        b, a_ = butter(4, 4000 / (SR_ / 2)); y = lfilter(b, a_, y, axis=-1)
    elif kind == 'board':
        y = y * 10 ** (-15 / 20) + rng.standard_normal(y.shape) * (10 ** (-73 / 20) * (1 << 23))
    return np.clip(np.round(y), -(1 << 23), (1 << 23) - 1).astype(np.int64)

# --------------------------------------------------------------------------- worker
_G = {}
def _init(n_mels, win, gain_aug, chan=False):
    _G['fe'] = FrontendInt(n_mels=n_mels, win=win)
    _G['gain_aug'] = gain_aug; _G['chan'] = chan
    _G['noise'] = [load_noise(f) for f in sorted(os.listdir(os.path.join(DATA, '_background_noise_'))) if f.endswith('.wav')] if False else \
                  [load_noise(f) for f in sorted(os.listdir(os.path.join(DATA, '_background_noise_'))) if f.endswith('.wav')]

def _work(args):
    items, copy_id, gain_db, seed = args[:4]; variant = args[4] if len(args) > 4 else None
    rng = np.random.default_rng(seed)
    pcm = np.stack([wav_float_to_pcm24(load_item(r)) for r in items])
    if copy_id > 0:
        pcm = np.stack([augment(p, rng, _G['noise'], _G['gain_aug'], _G['chan']) for p in pcm])
    if gain_db != 0:
        pcm = apply_gain_pcm24(pcm, gain_db)
    if variant:
        pcm = test_variant(pcm, variant, rng)
    return _G['fe'].run(pcm)

def run_split(pool, rels, copy_id, gain_db, bs=128, variant=None):
    jobs = [(rels[i:i + bs], copy_id, gain_db, SEED * 1000 + copy_id * 100003 + i, variant) for i in range(0, len(rels), bs)]
    out = pool.map(_work, jobs)
    return np.concatenate(out, 0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--classes', type=int, default=12, choices=[10, 12])
    ap.add_argument('--n_mels', type=int, default=C.N_MELS)
    ap.add_argument('--win', type=int, default=C.WIN)
    ap.add_argument('--aug_copies', type=int, default=2)
    ap.add_argument('--no_gain_aug', action='store_true')
    ap.add_argument('--test_gains', default='-20,-10,6,10')
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--tag', default='')
    ap.add_argument('--unknown_mult', type=float, default=1.0); ap.add_argument('--silence_mult', type=float, default=1.0)
    ap.add_argument('--silence_lowlevel', action='store_true')
    ap.add_argument('--channel_aug', action='store_true', help='4-B: random input-channel response / reverb / noise floor, gain U(-30,+10) dB')
    ap.add_argument('--test_variants', default='lp4k,board', help='extra fixed test sets X_test_<kind> (lp4k, board)')
    a = ap.parse_args()
    t0 = time.time()
    variant = (a.unknown_mult != 1.0 or a.silence_mult != 1.0 or a.silence_lowlevel or a.channel_aug)
    fl_path = os.path.join(OUT_DIR, f'filelist_c{a.classes}{"_" + a.tag.strip("_") if (variant and a.tag) else ""}.json')
    if os.path.exists(fl_path):
        splits = json.load(open(fl_path))
    else:
        base = json.load(open(os.path.join(OUT_DIR, f'filelist_c{a.classes}.json'))) if variant else None
        splits = build_filelist(a.classes, a.unknown_mult, a.silence_mult, a.silence_lowlevel, base)
        json.dump(splits, open(fl_path, 'w'))
    names = KEYWORDS + ([SILENCE, UNKNOWN] if a.classes == 12 else [])
    for s in splits:
        cnt = np.bincount([y for _, y in splits[s]], minlength=a.classes)
        print(f"{s}: {len(splits[s])} clips, per class {cnt.tolist()}")
    out = {}
    with Pool(a.workers, initializer=_init, initargs=(a.n_mels, a.win, not a.no_gain_aug, a.channel_aug)) as pool:
        for s in ('train', 'val', 'test'):
            rels = [r for r, _ in splits[s]]
            ys = np.array([y for _, y in splits[s]], dtype=np.int64)
            X = run_split(pool, rels, 0, 0)
            cp = np.zeros(len(X), dtype=np.int8)
            if s == 'train':
                Xs, ys_, cps = [X], [ys], [cp]
                for c in range(1, a.aug_copies + 1):
                    Xs.append(run_split(pool, rels, c, 0)); ys_.append(ys); cps.append(np.full(len(X), c, np.int8))
                    print(f"  train aug copy {c} done ({time.time()-t0:.0f}s)")
                X, ys, cp = np.concatenate(Xs), np.concatenate(ys_), np.concatenate(cps)
            out[f'X_{s}'], out[f'y_{s}'] = X, ys
            if s == 'train':
                out['copy_train'] = cp
            if s == 'test':
                for g in [int(v) for v in a.test_gains.split(',') if v]:
                    key = f'X_test_g{g:+d}'
                    out[key] = run_split(pool, rels, 0, g)
                for kind in [v for v in a.test_variants.split(',') if v]:
                    out[f'X_test_{kind}'] = run_split(pool, rels, 0, 0, variant=kind)
            print(f"{s}: X {X.shape} ({time.time()-t0:.0f}s)")
    meta = dict(classes=names, n_mels=a.n_mels, win=a.win, hop=C.HOP, T=C.N_FRAMES, data_bits=C.FFT_DATA_BITS,
                log_offset=C.LOG_OFFSET, aug_copies=a.aug_copies, gain_aug=not a.no_gain_aug, channel_aug=a.channel_aug, seed=SEED,
                counts={s: len(splits[s]) for s in splits})
    tag = a.tag if a.tag else ('' if not a.no_gain_aug else '_nogain')
    path = os.path.join(OUT_DIR, f'feat_c{a.classes}_m{a.n_mels}_w{a.win}{tag}.npz')
    np.savez(path, meta=json.dumps(meta), **out)
    Xt = out['X_train']
    print(f"saved {path}  ({time.time()-t0:.0f}s)  train feat stats: mean {Xt.mean():.2f} min {Xt.min()} max {Xt.max()} "
          f"ratio==0 {(Xt==0).mean():.3f} ratio==255 {(Xt==255).mean():.5f}")

if __name__ == '__main__':
    main()
