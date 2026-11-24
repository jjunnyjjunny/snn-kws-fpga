"""
export_snn_mem.py — export a QAT (int-mode) SNN checkpoint to integer parameters:
  snn/rom/<prefix>_<layer>_w.mem    weights, two's complement `bits` wide, address = pre*N_post + post
  snn/rom/<prefix>_<layer>_theta.mem per-neuron threshold, 16-bit unsigned
  snn/rom/<prefix>_<layer>_beta.mem  (ALIF) per-neuron adaptation increment, 16-bit unsigned
  snn/rom/<prefix>_params.json       dims, k (leak shift) per layer, in_shift, ka, m, bit widths, order
and returns the same as a python dict (used by snn_int_ref.py).

usage: python snn/golden/export_snn_mem.py --run snn/runs/M1_c12_q8 --prefix m1
"""
import os, sys, json, argparse
import numpy as np, torch
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
from snn.models.lif import set_mode, QLinear, LIFCell, ALIFCell, OutIntegrator, V_BITS_HIDDEN, V_BITS_OUT
from snn.models.snn_models import build
ROM = os.path.join(ROOT, 'snn', 'rom')

def load_run(run):
    cfg = json.load(open(os.path.join(run, 'config.json')))
    res = json.load(open(os.path.join(run, 'results.json')))
    n_in = {'m40': 40, 'm32': 32}[[t for t in os.path.basename(cfg['data']).split('_') if t.startswith('m')][0]]
    n_out = len(res['classes'])
    kw = dict(second_layer=cfg.get('second_layer', False), adapt=not cfg.get('no_adapt', False)) if cfg['model'] == 'M2' else {}
    model = build(cfg['model'], n_in, n_out, cfg['hidden'], cfg['bits'], cfg['k'], k_out=cfg.get('k_out'), K=cfg.get('K', 2),
                  dilations=[int(v) for v in cfg.get('dilations', '1,2,4,8').split(',')], theta_init=cfg['theta_init'], **kw)
    model.load_state_dict(torch.load(os.path.join(run, 'best.pt'), map_location='cpu'))
    set_mode(model, 'int')
    model.eval()
    return model, cfg, res

@torch.no_grad()
def extract(model):
    """-> dict with integer numpy params, in forward order."""
    P = dict(model=model.name, layers=[], v_bits_hidden=V_BITS_HIDDEN, v_bits_out=V_BITS_OUT)
    def lin(ql):
        W, s, sh = ql.export(); return W, s, sh
    def bias(ql):
        return ql.b_int().detach().numpy().astype('int64')
    def cell(c, q):
        d = dict(theta=c.theta_int(torch.tensor(q)).numpy().astype('int64'), k=c.k)
        if isinstance(c, ALIFCell):
            d.update(beta=c.beta_int(torch.tensor(q)).numpy().astype('int64'), ka=c.ka, m=c.m)
        return d
    if model.name == 'M1':
        for nm, ql, c in (('l1', model.fc1, model.lif1), ('l2', model.fc2, model.lif2)):
            W, s, sh = lin(ql); q = float(ql.q())
            P['layers'].append(dict(name=nm, type='lif', W=W, b=bias(ql), bits=ql.bits, in_shift=sh, q=q, **cell(c, q)))
        W, s, sh = lin(model.fc3)
        P['layers'].append(dict(name='out', type='out', W=W, b=bias(model.fc3), bits=model.fc3.bits, in_shift=0, q=float(model.fc3.q()), k=model.out.k))
    elif model.name == 'M2':
        W, s, sh = lin(model.fc1); q = float(model.fc1.q())
        Wr, rsh = model.export_rec()
        P['layers'].append(dict(name='l1', type='alif', W=W, b=bias(model.fc1), W_rec=Wr, rec_shift=rsh, bits=model.fc1.bits, in_shift=sh, q=q, **cell(model.alif, q)))
        if model.second:
            W, s, sh = lin(model.fc2); q2 = float(model.fc2.q())
            P['layers'].append(dict(name='l2', type='lif', W=W, b=bias(model.fc2), bits=model.fc2.bits, in_shift=0, q=q2, **cell(model.lif2, q2)))
        W, s, sh = lin(model.fc3)
        P['layers'].append(dict(name='out', type='out', W=W, b=bias(model.fc3), bits=model.fc3.bits, in_shift=0, q=float(model.fc3.q()), k=model.out.k))
    elif model.name == 'M3':
        for i, (ql, c, d) in enumerate(zip(model.convs, model.lifs, model.dil)):
            W, s, sh = lin(ql); q = float(ql.q())
            P['layers'].append(dict(name=f'tcn{i+1}', type='tcn', W=W, b=bias(ql), K=model.K, dilation=d, bits=ql.bits, in_shift=sh, q=q, **cell(c, q)))
        W, s, sh = lin(model.fc_out)
        P['layers'].append(dict(name='out', type='out', W=W, b=bias(model.fc_out), bits=model.fc_out.bits, in_shift=0, q=float(model.fc_out.q()), k=model.out.k))
    return P

def to_hex(v, bits):
    v = int(v)
    if v < 0: v += 1 << bits
    assert 0 <= v < (1 << bits)
    return f"{v:0{(bits + 3) // 4}x}"

def write_mem(path, vals, bits, comment):
    with open(path, 'w') as f:
        f.write(f"// {comment}\n// width={bits} depth={len(vals)}\n")
        for v in vals: f.write(to_hex(v, bits) + "\n")

def export(P, prefix):
    os.makedirs(ROM, exist_ok=True)
    meta = dict(model=P['model'], v_bits_hidden=P['v_bits_hidden'], v_bits_out=P['v_bits_out'], layers=[])
    files = []
    for L in P['layers']:
        W = L['W']                     # [n_post, n_pre] (torch layout) -> address = pre * n_post + post
        n_post, n_pre = W.shape
        fn = f"{prefix}_{L['name']}_w.mem"
        write_mem(os.path.join(ROM, fn), W.T.reshape(-1), L['bits'], f"{L['name']} weights [pre={n_pre}][post={n_post}], addr=pre*{n_post}+post, {L['bits']}-bit two's complement")
        files.append((fn, L['bits'], n_pre * n_post, f"addr = pre*{n_post} + post (pre = input/presynaptic index, post = neuron index)"))
        m = dict(name=L['name'], type=L['type'], n_pre=int(n_pre), n_post=int(n_post), bits=L['bits'], in_shift=L['in_shift'], k=L['k'], q_real_per_int=L['q'])
        fn = f"{prefix}_{L['name']}_b.mem"
        write_mem(os.path.join(ROM, fn), L['b'], 16, f"{L['name']} per-neuron bias current (16-bit two's complement), added every frame")
        files.append((fn, 16, len(L['b']), "addr = neuron index; signed constant current added to I every frame"))
        if 'W_rec' in L:
            fn = f"{prefix}_{L['name']}_wrec.mem"; Wr = L['W_rec']
            write_mem(os.path.join(ROM, fn), Wr.T.reshape(-1), L['bits'], f"{L['name']} recurrent weights [pre={Wr.shape[1]}][post={Wr.shape[0]}], I_rec = (sum) << {L['rec_shift']}")
            files.append((fn, L['bits'], Wr.size, f"addr = pre*{Wr.shape[0]} + post ; current of previous frame's spikes, result << rec_shift={L['rec_shift']}"))
            m['rec_shift'] = L['rec_shift']
        if 'theta' in L:
            fn = f"{prefix}_{L['name']}_theta.mem"
            write_mem(os.path.join(ROM, fn), L['theta'], 16, f"{L['name']} thresholds (unsigned)")
            files.append((fn, 16, len(L['theta']), "addr = neuron index; unsigned threshold in membrane units"))
            m['theta_min'], m['theta_max'] = int(L['theta'].min()), int(L['theta'].max())
        if 'beta' in L:
            fn = f"{prefix}_{L['name']}_beta.mem"
            write_mem(os.path.join(ROM, fn), L['beta'], 16, f"{L['name']} ALIF adaptation increment (unsigned)")
            files.append((fn, 16, len(L['beta']), "addr = neuron index; a += beta on spike; theta_eff = theta + (a >> m)"))
            m['ka'], m['m'] = L['ka'], L['m']
        if 'K' in L:
            m['K'], m['dilation'] = L['K'], L['dilation']
            m['tap_layout'] = "pre index = tap*n_in_channels + channel, tap j uses input at t - j*dilation"
        meta['layers'].append(m)
    json.dump(meta, open(os.path.join(ROM, f"{prefix}_params.json"), 'w'), indent=1)
    # append to README
    with open(os.path.join(ROM, 'README.md'), 'a') as f:
        f.write(f"\n## SNN `{prefix}` ({P['model']}) — generated by snn/golden/export_snn_mem.py\n\n")
        f.write("| file | width (bits) | depth | address / content rule |\n|---|---|---|---|\n")
        for fn, b, d, r in files:
            f.write(f"| `{fn}` | {b} | {d} | {r} |\n")
        f.write(f"\nParameters (k = leak shift per layer, in_shift = input-layer accumulator shift, bit widths): `{prefix}_params.json`.\n")
        f.write("Neuron update (per layer, per frame): `v = sat_s16(v - (v >>> k) + I)`; `spike = (v >= theta)`; `if spike: v -= theta`.\n")
        f.write("Input layer: `I = ((sum_i x[i] * W[i][j]) >>> in_shift) + b[j]` (x = 8-bit feature, arithmetic shift = floor).\n")
        f.write("Hidden layers: `I[j] = b[j] + sum over spiking pre i of W[i][j]` (no multiply). Output: `vo = sat_s24(vo - (vo >>> k) + I)`;\n")
        f.write("decision = argmax over classes of sum_t vo[t] (equivalently mean); alt: last step / max over t.\n")
        if P['model'] == 'M2':
            f.write("ALIF: `a = sat_u16(a - (a >>> ka) + beta * spike_prev)`, `theta_eff = theta + (a >>> m)`; recurrent current from previous frame's spikes added to I.\n")
        if P['model'] == 'M3':
            f.write("TCN: layer input vector = concat over taps j=0..K-1 of input at frame t - j*dilation (zeros for t<0). 1-bit spike history per layer.\n")
    return meta

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True); ap.add_argument('--prefix', required=True)
    a = ap.parse_args()
    model, cfg, res = load_run(a.run)
    P = extract(model)
    meta = export(P, a.prefix)
    print(json.dumps(meta, indent=1))
