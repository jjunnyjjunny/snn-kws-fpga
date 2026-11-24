"""
snn_int_ref.py — pure numpy int64 golden inference for exported SNN parameters, plus a
comparison against the PyTorch int-mode model on the whole test set (predictions must match 100 %).

usage: CUDA_VISIBLE_DEVICES=6 python snn/golden/snn_int_ref.py --run snn/runs/M1_c12_q8 --data snn/data/feat_c12_m40_w480.npz
"""
import os, sys, json, argparse
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

def sat(x, bits, signed=True):
    if signed:
        return np.clip(x, -(1 << (bits - 1)), (1 << (bits - 1)) - 1)
    return np.clip(x, 0, (1 << bits) - 1)

def lif_step(v, I, theta, k, v_bits):
    v = sat(v - (v >> k) + I, v_bits)
    s = (v >= theta).astype(np.int64)
    v = v - s * theta
    return v, s

def run_int(P, x, trace=False):
    """P: dict from export_snn_mem.extract ; x: uint8 [N, T, n_in]. Returns dict(pred_sum, pred_last, pred_max, vout[N,T,C], traces)"""
    x = np.asarray(x, dtype=np.int64)
    N, T, _ = x.shape
    VB, VO = P['v_bits_hidden'], P['v_bits_out']
    layers = P['layers']
    hid = [L for L in layers if L['type'] != 'out']
    out = [L for L in layers if L['type'] == 'out'][0]
    state = []
    for L in hid:
        n = L['W'].shape[0]
        st = dict(v=np.zeros((N, n), np.int64), s=np.zeros((N, n), np.int64))
        if L['type'] == 'alif': st['a'] = np.zeros((N, n), np.int64)
        if L['type'] == 'tcn': st['hist'] = []
        state.append(st)
    vo = np.zeros((N, out['W'].shape[0]), np.int64)
    vout = np.zeros((N, T, out['W'].shape[0]), np.int64)
    tr = {L['name']: dict(v=np.zeros((N, T, L['W'].shape[0]), np.int64), s=np.zeros((N, T, L['W'].shape[0]), np.int64)) for L in hid} if trace else None
    for t in range(T):
        inp = x[:, t]
        for L, st in zip(hid, state):
            if L['type'] == 'tcn':
                st['hist'].append(inp)
                taps = [st['hist'][t - j * L['dilation']] if t - j * L['dilation'] >= 0 else np.zeros_like(inp) for j in range(L['K'])]
                inp_vec = np.concatenate(taps, -1)
            else:
                inp_vec = inp
            I = inp_vec @ L['W'].T
            if L['in_shift']:
                I = I >> L['in_shift']
            I = I + L['b']
            if L['type'] == 'alif':
                s_prev = st['s']
                I = I + ((s_prev @ L['W_rec'].T) << L['rec_shift'])
                a = sat(st['a'] - (st['a'] >> L['ka']) + s_prev * L['beta'], 16, signed=False)
                theta = L['theta'] + (a >> L['m'])
                st['a'] = a
            else:
                theta = L['theta']
            st['v'], st['s'] = lif_step(st['v'], I, theta, L['k'], VB)
            if trace:
                tr[L['name']]['v'][:, t] = st['v']; tr[L['name']]['s'][:, t] = st['s']
            inp = st['s']
        vo = sat(vo - (vo >> out['k']) + inp @ out['W'].T + out['b'], VO)
        vout[:, t] = vo
    return dict(pred_sum=vout.sum(1).argmax(1), pred_last=vout[:, -1].argmax(1), pred_max=vout.max(1).argmax(1),
                vout=vout, traces=tr)

if __name__ == '__main__':
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    from snn.golden.export_snn_mem import load_run, extract
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True); ap.add_argument('--data', required=True)
    ap.add_argument('--n_trace', type=int, default=64)
    a = ap.parse_args()
    model, cfg, res = load_run(a.run)
    model = model.cuda()
    P = extract(model.cpu()); model = model.cuda()
    z = np.load(a.data); X, y = z['X_test'], z['y_test']
    # torch int-mode predictions (exact integer sums in float64)
    with torch.no_grad():
        preds_t, vout_t = [], []
        for i in range(0, len(X), 512):
            o = model(torch.from_numpy(X[i:i + 512]).cuda())
            preds_t.append(o['vout'].double().sum(1).argmax(1).cpu().numpy()); vout_t.append(o['vout'].cpu().numpy().astype(np.int64))
        preds_t = np.concatenate(preds_t); vout_t = np.concatenate(vout_t)
    g = run_int(P, X, trace=False)
    agree = float((g['pred_sum'] == preds_t).mean())
    vout_eq = float((g['vout'] == vout_t).mean())
    acc_np = float((g['pred_sum'] == y).mean())
    # trace check on a subset
    with torch.no_grad():
        o = model(torch.from_numpy(X[:a.n_trace]).cuda())
    gt = run_int(P, X[:a.n_trace], trace=True)
    trace_eq = {}
    for i, L in enumerate([L for L in P['layers'] if L['type'] != 'out']):
        trace_eq[L['name']] = float((o['spikes'][i].cpu().numpy().astype(np.int64) == gt['traces'][L['name']]['s']).mean())
    out = dict(run=a.run, n_test=len(X), pred_agreement=agree, vout_elementwise_equal=vout_eq, spike_trace_equal=trace_eq,
               acc_numpy_sum=acc_np, acc_numpy_last=float((g['pred_last'] == y).mean()), acc_numpy_max=float((g['pred_max'] == y).mean()),
               acc_torch=float((preds_t == y).mean()))
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(a.run, 'golden_check.json'), 'w'), indent=1)

# ----------------------------------------------------------------------------- frame-by-frame stepper (same math as run_int)
class SnnStepper:
    """Stateful integer SNN: reset() then step(x[n_in]) -> vo[C] per frame. Bit-exact with run_int."""
    def __init__(self, P):
        self.P = P; self.hid = [L for L in P['layers'] if L['type'] != 'out']; self.out = [L for L in P['layers'] if L['type'] == 'out'][0]
        self.reset()
    def reset(self):
        self.t = -1; self.state = []
        for L in self.hid:
            n = L['W'].shape[0]; st = dict(v=np.zeros(n, np.int64), s=np.zeros(n, np.int64))
            if L['type'] == 'alif': st['a'] = np.zeros(n, np.int64)
            if L['type'] == 'tcn': st['hist'] = []
            self.state.append(st)
        self.vo = np.zeros(self.out['W'].shape[0], np.int64)
    def step(self, x):
        P = self.P; VB, VO = P['v_bits_hidden'], P['v_bits_out']; self.t += 1; t = self.t
        inp = np.asarray(x, dtype=np.int64)
        for L, st in zip(self.hid, self.state):
            if L['type'] == 'tcn':
                st['hist'].append(inp)
                taps = [st['hist'][t - j * L['dilation']] if t - j * L['dilation'] >= 0 else np.zeros_like(inp) for j in range(L['K'])]
                inp_vec = np.concatenate(taps, -1)
            else: inp_vec = inp
            I = inp_vec @ L['W'].T
            if L['in_shift']: I = I >> L['in_shift']
            I = I + L['b']
            if L['type'] == 'alif':
                s_prev = st['s']; I = I + ((s_prev @ L['W_rec'].T) << L['rec_shift'])
                a = sat(st['a'] - (st['a'] >> L['ka']) + s_prev * L['beta'], 16, signed=False); theta = L['theta'] + (a >> L['m']); st['a'] = a
            else: theta = L['theta']
            st['v'], st['s'] = lif_step(st['v'], I, theta, L['k'], VB); inp = st['s']
        self.vo = sat(self.vo - (self.vo >> self.out['k']) + inp @ self.out['W'].T + self.out['b'], VO)
        return self.vo.copy()
