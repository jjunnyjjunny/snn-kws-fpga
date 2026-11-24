"""
capture_level.py — speaker-volume calibration helper (BOARD_STEP2 §1B): peak / RMS / floor of an ILA mode-3 capture
(dbg_cap_sample, 18-bit = pcm[23:6]) in dBFS, and whether it is inside the target window (default -10 ... -6 dBFS peak,
the Speech Commands test clip level).
usage: python snn/eval/capture_level.py snn/data/board_capture/cal_01.csv [more.csv ...] [--lo -10 --hi -6]
"""
import os, sys, argparse
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')); sys.path.insert(0, ROOT)
from snn.eval.board_capture_check import read_capture, level_stats

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('csv', nargs='+'); ap.add_argument('--lo', type=float, default=-10.0); ap.add_argument('--hi', type=float, default=-6.0)
    a = ap.parse_args()
    print("| 캡처 | 길이 s | peak dBFS | RMS dBFS | 바닥 dBFS | DC (LSB18) | 판정 |"); print("|---|---|---|---|---|---|---|")
    for p in a.csv:
        x18, _ = read_capture(p); r = level_stats(x18 << 6); pk = r['peak_dbfs24']
        v = '목표 안' if a.lo <= pk <= a.hi else ('너무 큼 → 볼륨/거리 줄이기' if pk > a.hi else f'너무 작음 ({a.lo - pk:.1f} dB 더) → 볼륨 올리기/거리 줄이기')
        if pk > -1.0: v = '클리핑 위험 → 볼륨 줄이기'
        print(f"| {os.path.basename(p)} | {len(x18) / 16000:.2f} | {pk:.1f} | {r['rms_dbfs24']:.1f} | {r['floor_dbfs']:.1f} | {r['dc'] / 64:.0f} | {v} |")

if __name__ == '__main__':
    main()
