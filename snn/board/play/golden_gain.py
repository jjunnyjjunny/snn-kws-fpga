"""
golden_gain.py — digital golden of the playback test at a different chain gain (BOARD_STEP2 §1B follow-up).
The board chain measured +1 dB (cal tone -6 -> -5 dBFS), so the wav is scaled by --gain_db (saturating at 16-bit full
scale like the codec ADC would) and run through the same integer pipeline as make_playback_test.golden().
usage: python snn/board/play/golden_gain.py --gain_db 1 [--run M3k3e60_U2_q8_kd_s0] [--out snn/board/play/golden_gain]
"""
import os, sys, argparse
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')); sys.path.insert(0, ROOT)
from snn.board.play.make_playback_test import golden, SR

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--gain_db', type=float, nargs='+', default=[1.0])
    ap.add_argument('--run', default='M3k3e60_U2_q8_kd_s0'); ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'golden_gain'))
    a = ap.parse_args()
    import soundfile as sf
    x, sr = sf.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'playback_test.wav'), dtype='int16'); assert sr == SR
    for g in a.gain_db:
        y = np.clip(np.round(x.astype(np.float64) * 10 ** (g / 20)), -32768, 32767).astype(np.int64)
        d = os.path.join(a.out, f"g{g:+.0f}dB"); os.makedirs(d, exist_ok=True)
        print(f"gain {g:+.1f} dB: saturated samples {int((np.abs(y) >= 32767).sum())}, peak {20 * np.log10(np.abs(y).max() / 32768):.2f} dBFS -> {d}")
        golden(d, y, a.run)

if __name__ == '__main__':
    main()
