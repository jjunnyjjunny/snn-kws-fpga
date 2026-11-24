"""
power_rails.py — Vivado report_power 파일에서 "1.2 Power Supply Summary"(레일별 전압·전류)를 뽑아 빌드 × 시나리오 표를 만들고,
시나리오별 게이팅 전후(또는 임의 두 빌드) 레일 전류 차이와 보드 12 V 입력 환산(레귤레이터 효율 범위)을 계산한다. 전부 추정치(Vivado 모델).
입력: `report_power -file`(summary) 또는 `-hier all -file` 리포트 모두 가능(둘 다 1.1/1.2 절을 포함). 레일 전류는 칩 전체 값이므로
PS7·ILA·정적 전력이 섞여 있다 — 빌드/시나리오 사이의 **차이**만 설계 효과로 해석한다(공통 부분은 상쇄).
usage:
  python snn/eval/power_rails.py --rpt nogate/silence=reports_power_nogate/silence_saif_summary.rpt \
         --rpt gate_off/silence=reports_power/silence_saif_ce11_summary.rpt --rpt gate_ce00/silence=... --rpt gate_ce10/silence=... --rpt gate_ce01/silence=... \
         [--duty silence:0.0:0.0123 --duty keyword:0.0121:0.0124 --duty talk:0.0078:0.0123]   # d_snn:d_fe -> 'gate_on' 듀티 가중 빌드 생성
         --pairs gate_off-gate_on,nogate-gate_on --vin 12 --eff 0.8,0.9 --out snn/runs/power_rails
라벨 형식 BUILD/SCENARIO. 듀티 가중: I_on = I_ce00 + d_snn·(I_ce10 − I_ce00) + d_fe·(I_ce01 − I_ce00) (POWER_COMPARE.md 4절과 같은 식).
"""
import os, re, sys, json, argparse
RAILS_PL = ['Vccint', 'Vccaux', 'Vccbram', 'Vcco33', 'Vcco25', 'Vcco18', 'Vcco15', 'Vccaux_io']

def parse_rpt(path):
    """-> dict(rails={name: dict(v, total, dyn, static)}, components={name: W}, summary={...})"""
    txt = open(path, encoding='utf-8', errors='replace').read(); out = dict(rails={}, components={}, summary={})
    m = re.search(r'1\.2 Power Supply Summary\s*\n-+\n(.*?)\n\n', txt, re.S)
    if not m: raise ValueError(f'{path}: Power Supply Summary 없음')
    for line in m.group(1).splitlines():
        c = [x.strip() for x in line.strip().strip('|').split('|')]
        if len(c) >= 5 and re.match(r'^[A-Za-z]', c[0]) and c[0] != 'Source':
            try: out['rails'][c[0]] = dict(v=float(c[1]), total=float(c[2]), dyn=float(c[3]), static=float(c[4]))
            except ValueError: pass
    m = re.search(r'1\.1 On-Chip Components\s*\n-+\n(.*?)\n\n', txt, re.S)
    if m:
        for line in m.group(1).splitlines():
            c = [x.strip() for x in line.strip().strip('|').split('|')]
            if len(c) >= 2 and c[0] and c[0] not in ('On-Chip',) and not line.startswith('+'):
                v = c[1].replace('<', '')
                try: out['components'][c[0]] = float(v)
                except ValueError: pass
    for k in ('Total On-Chip Power (W)', 'Dynamic (W)', 'Device Static (W)', 'Design Nets Matched', 'Confidence Level'):
        mm = re.search(r'\|\s*' + re.escape(k) + r'\s*\|\s*([^|]+?)\s*\|', txt)
        if mm: out['summary'][k] = mm.group(1)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rpt', action='append', default=[], help='BUILD/SCENARIO=path')
    ap.add_argument('--duty', action='append', default=[], help='SCENARIO:d_snn:d_fe -> gate_on (ce00/ce10/ce01 필요)')
    ap.add_argument('--pairs', default='', help='예: gate_off-gate_on,nogate-gate_on (앞 − 뒤)')
    ap.add_argument('--vin', type=float, default=12.0); ap.add_argument('--eff', default='0.8,0.9'); ap.add_argument('--rails', default='Vccint,Vccaux,Vccbram')
    ap.add_argument('--out', default='')
    a = ap.parse_args(); effs = [float(x) for x in a.eff.split(',')]; rails = a.rails.split(',')
    data = {}
    for spec in a.rpt:
        lab, path = spec.split('=', 1); b, s = lab.split('/'); data.setdefault(b, {})[s] = parse_rpt(path)
    for spec in a.duty:
        s, ds, df = spec.split(':'); ds, df = float(ds), float(df)
        try: r00, r10, r01 = (data[b][s]['rails'] for b in ('gate_ce00', 'gate_ce10', 'gate_ce01'))
        except KeyError: print(f'duty {s}: gate_ce00/ce10/ce01 리포트가 없어 건너뜀'); continue
        est = {k: dict(v=r00[k]['v'], **{f: r00[k][f] + ds * (r10[k][f] - r00[k][f]) + df * (r01[k][f] - r00[k][f]) for f in ('total', 'dyn', 'static')}) for k in r00}
        data.setdefault('gate_on', {})[s] = dict(rails=est, components={}, summary={'note': f'duty-weighted d_snn={ds} d_fe={df}'})
    L = ['## 레일별 전압·전류 (Vivado report_power Power Supply Summary, 칩 전체, **추정**)', '',
         '| 빌드 | 시나리오 | ' + ' | '.join(f'{r} V / 총 mA / 동적 mA' for r in rails) + ' | 레일 합 동적 mW | 리포트 동적 W | 넷 매칭 |', '|---|---|' + '---|' * (len(rails) + 3)]
    rows = []
    for b in data:
        for s, d in data[b].items():
            R = d['rails']; cells = []; pdyn = 0.0
            for r in rails:
                if r in R: cells.append(f"{R[r]['v']:.2f} / {1e3 * R[r]['total']:.1f} / {1e3 * R[r]['dyn']:.1f}"); pdyn += R[r]['v'] * R[r]['dyn']
                else: cells.append('–')
            L.append(f"| {b} | {s} | " + ' | '.join(cells) + f" | {1e3 * pdyn:.1f} | {d['summary'].get('Dynamic (W)', '–')} | {d['summary'].get('Design Nets Matched', d['summary'].get('note', '–'))} |")
            rows.append(dict(build=b, scenario=s, rails={r: R.get(r) for r in rails}, pdyn_W=pdyn, summary=d['summary']))
    pairs = [p.split('-') for p in a.pairs.split(',') if p]; D = []
    if pairs:
        L += ['', f'## 빌드 간 레일 전류 차이 (앞 − 뒤; 12 V 입력 환산은 ΔP_rail / η, η = {", ".join(f"{e:.0%}" for e in effs)}; **추정**)', '',
              '| 쌍 | 시나리오 | ' + ' | '.join(f'Δ{r} mA' for r in rails) + ' | ΔP 레일 합 mW | ΔP 입력 mW | ΔI @ ' + f'{a.vin:.0f} V mA |', '|---|---|' + '---|' * (len(rails) + 3)]
        for x, y in pairs:
            for s in data.get(x, {}):
                if s not in data.get(y, {}): continue
                Rx, Ry = data[x][s]['rails'], data[y][s]['rails']; di = {}; dp = 0.0
                for r in rails:
                    if r in Rx and r in Ry: di[r] = 1e3 * (Rx[r]['total'] - Ry[r]['total']); dp += Rx[r]['v'] * (Rx[r]['total'] - Ry[r]['total'])
                pin = [dp / e for e in effs]
                L.append(f"| {x} − {y} | {s} | " + ' | '.join(f"{di[r]:+.1f}" if r in di else '–' for r in rails) + f" | {1e3 * dp:+.1f} | {' ~ '.join(f'{1e3 * p:+.1f}' for p in sorted(pin))} | {' ~ '.join(f'{1e3 * p / a.vin:+.2f}' for p in sorted(pin))} |")
                D.append(dict(pair=f'{x}-{y}', scenario=s, dI_mA=di, dP_rail_W=dp, dP_in_W=pin))
    L += ['', '주의: 레일 전류는 칩 전체(PS7·ILA·정적 포함) 값이므로 절대값이 아니라 빌드/시나리오 간 차이만 설계 효과로 읽는다. 리포트 해상도 1 mA·1 mW. 모든 값은 Vivado 모델 추정이며 보드 실측이 아니다.']
    txt = '\n'.join(L); print(txt)
    if a.out:
        open(a.out + '.md', 'w').write(txt + '\n'); json.dump(dict(rows=rows, diffs=D, vin=a.vin, eff=effs), open(a.out + '.json', 'w'), indent=1); print('->', a.out + '.md/.json')

if __name__ == '__main__':
    main()
