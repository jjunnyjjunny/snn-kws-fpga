"""make_power_vectors.py — 3 s 24-bit sample files for the SAIF/activity scenarios: silence (background noise only),
keyword (one keyword at 1 s), talk (unknown + keyword utterances). Written to snn/board/pkg/power_vec/<name>.hex (and .npy)."""
import os, sys, numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')); sys.path.insert(0, ROOT)
from snn.eval import vad_eval as V
from snn.frontend.frontend_ref import wav_float_to_pcm24, apply_gain_pcm24
from snn.data.make_dataset import load_item
V._init(os.path.join(ROOT, 'snn', 'runs', 'M3k3e60_c12_q8_kd'))
out = os.path.join(ROOT, 'snn', 'board', 'pkg', 'power_vec'); os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(99); SR = 16000; n = 3 * SR; noise = V._G['noise']['test']; noise = np.tile(noise, 4)
files, labels = zip(*V._G['files']['test']); kw = [i for i, y in enumerate(labels) if y < 10]; unk = [i for i, y in enumerate(labels) if y >= 10]
def base(): st = int(rng.integers(0, len(noise) - n)); return np.round(noise[st:st + n].astype(np.float64) * 0.02 * (1 << 23)).astype(np.int64)
def put(wav, pos, i): wav[pos:pos + SR] += apply_gain_pcm24(wav_float_to_pcm24(load_item(files[i])), 0.0)
scen = {}
scen['silence'] = base()
w = base(); put(w, SR, int(rng.choice(kw))); scen['keyword'] = w
w = base(); put(w, int(0.2 * SR), int(rng.choice(unk))); put(w, int(1.3 * SR), int(rng.choice(kw))); scen['talk'] = w
for name, w in scen.items():
    w = np.clip(w, -(1 << 23), (1 << 23) - 1)
    with open(os.path.join(out, name + '.hex'), 'w') as f:
        for v in w: f.write(f"{int(v) & 0xffffff:06x}\n")
    np.save(os.path.join(out, name + '.npy'), w); print(name, len(w), 'samples, peak', int(np.abs(w).max()))
