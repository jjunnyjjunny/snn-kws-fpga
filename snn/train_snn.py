"""
train_snn.py — train / QAT-finetune / evaluate SNN candidates (M1, M2, M3).

Examples
  float :  python snn/train_snn.py --model M1 --data snn/data/feat_c12_m40_w480.npz --name M1_c12_float --mode float --epochs 30
  QAT   :  python snn/train_snn.py --model M1 --data ... --name M1_c12_q8 --mode int --bits 8 --init snn/runs/M1_c12_float/best.pt --epochs 15
  KD    :  add --kd_logits snn/runs/dscnn_c12/teacher_logits_train.npy --kd_alpha 1.0 --kd_tau 4
Everything (config, per-epoch log, best checkpoint, test metrics incl. spike rates / ops / vmax)
is written to snn/runs/<name>/.
"""
import os, sys, json, time, argparse, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from snn.data.loader import FeatData
from snn.models.lif import set_mode, SpikeFn, QLinear, LIFCell, ALIFCell
from snn.models.snn_models import build

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision('highest')

def kd_loss(student, teacher, tau):
    return F.kl_div(F.log_softmax(student / tau, 1), F.softmax(teacher / tau, 1), reduction='batchmean') * tau * tau

@torch.no_grad()
def evaluate(model, X, y, bs=512, full=False):
    model.eval()
    n = len(X); correct = {'mean': 0, 'last': 0, 'max': 0}
    rates, vmax, ops_ac, logits_all = None, None, None, []
    for i in range(0, n, bs):
        o = model(X[i:i + bs]); yb = y[i:i + bs]
        correct['mean'] += (o['vout'].double().sum(1).argmax(1) == yb).sum().item()   # exact integer sum (argmax == mean)
        correct['last'] += (o['vout'][:, -1].argmax(1) == yb).sum().item()
        correct['max'] += (o['vout'].max(1).values.argmax(1) == yb).sum().item()
        logits_all.append(o['vout'].double().sum(1))
        if full:
            r = [s.mean().item() * len(yb) for s in o['spikes']]            # mean firing prob per neuron-step
            sp = [s.sum().item() for s in o['spikes']]                     # total spikes
            rates = r if rates is None else [a + b for a, b in zip(rates, r)]
            ops_ac = sp if ops_ac is None else [a + b for a, b in zip(ops_ac, sp)]
            vmax = o['vmax'] if vmax is None else [max(a, b) for a, b in zip(vmax, o['vmax'])]
    acc = {k: v / n for k, v in correct.items()}
    if not full:
        return acc, torch.cat(logits_all)
    return acc, torch.cat(logits_all), [r / n for r in rates], [s / n for s in ops_ac], vmax

def hw_cost(model, D, spikes_per_sample):
    """parameter / memory / op counts per sample and per frame (serial 1 MAC + 1 adder)."""
    T = D.T
    layers = []
    for nm, mod in model.named_modules():
        if isinstance(mod, QLinear):
            layers.append((nm, mod.W.shape[1], mod.W.shape[0], mod.bits, mod.is_input))
    n_params = sum(i * o for _, i, o, _, _ in layers)
    w_bits = sum(i * o * b for _, i, o, b, _ in layers)
    mac = sum(i * o for _, i, o, _, inp in layers if inp) * T               # input layer (dense, every frame)
    # accumulate ops: each hidden spike fans out to the next layer's neurons (+ recurrent / taps)
    fanout = {}
    if model.name == 'M1':
        fanout = {0: model.fc2.W.shape[0], 1: model.fc3.W.shape[0]}
    elif model.name == 'M2':
        fanout = {0: model.rec.W.shape[0] + (model.fc2.W.shape[0] if model.second else model.fc3.W.shape[0])}
        if model.second: fanout[1] = model.fc3.W.shape[0]
    elif model.name == 'M3':
        for i in range(len(model.dil)):
            fanout[i] = (model.convs[i + 1].W.shape[0] * model.K) if i + 1 < len(model.dil) else model.fc_out.W.shape[0]
    ac = sum(spikes_per_sample[i] * fanout[i] for i in range(len(spikes_per_sample)))
    n_neurons = sum(m.n for m in model.modules() if isinstance(m, LIFCell)) + model.out.n
    update = n_neurons * T * 2                                             # ~2 cycles per neuron update
    cyc_serial = (mac + ac + update) / T
    return dict(n_params=n_params, weight_bits=w_bits, weight_kB=w_bits / 8 / 1024, bram18k=math.ceil(w_bits / 18432),
                mac_per_sample=mac, ac_per_sample=ac, update_per_sample=update,
                cycles_per_frame_serial=cyc_serial, cycles_per_frame_overlap=(max(mac, ac) + update) / T,
                layers=[dict(name=n, n_in=i, n_out=o, bits=b, is_input=inp) for n, i, o, b, inp in layers])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['M1', 'M2', 'M3'])
    ap.add_argument('--data', required=True); ap.add_argument('--name', required=True)
    ap.add_argument('--mode', default='float', choices=['float', 'int'])
    ap.add_argument('--bits', type=int, default=8); ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--k', type=int, default=3); ap.add_argument('--theta_init', type=float, default=1.0)
    ap.add_argument('--second_layer', action='store_true', help='M2: add LIF layer after ALIF')
    ap.add_argument('--no_adapt', action='store_true', help='M2: plain recurrent LIF (beta = 0)')
    ap.add_argument('--epochs', type=int, default=30); ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--lr', type=float, default=2e-3); ap.add_argument('--wd', type=float, default=0.0)
    ap.add_argument('--sg', default='fastsig', choices=['fastsig', 'atan']); ap.add_argument('--sg_scale', type=float, default=4.0); ap.add_argument('--sg_gain', type=float, default=1.0)
    ap.add_argument('--init', default=''); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--kd_logits', default=''); ap.add_argument('--kd_alpha', type=float, default=0.0); ap.add_argument('--kd_tau', type=float, default=4.0)
    ap.add_argument('--clean_only', action='store_true'); ap.add_argument('--eval_only', action='store_true')
    ap.add_argument('--rate_reg', type=float, default=0.0, help='L2 penalty on mean firing rate above target')
    ap.add_argument('--rate_target', type=float, default=0.1)
    ap.add_argument('--k_out', type=int, default=None, help='output integrator leak shift (default = k)')
    ap.add_argument('--class_weight', default='', help='CE class weights, e.g. "10:2,11:3" (others 1)')
    ap.add_argument('--K', type=int, default=2, help='M3 kernel taps'); ap.add_argument('--dilations', default='1,2,4,8')
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    SpikeFn.sg_type, SpikeFn.sg_scale, SpikeFn.sg_gain = a.sg, a.sg_scale, a.sg_gain
    run = os.path.join(ROOT, 'snn', 'runs', a.name); os.makedirs(run, exist_ok=True)
    D = FeatData(a.data, clean_only=a.clean_only)
    kw = dict(second_layer=a.second_layer, adapt=not a.no_adapt) if a.model == 'M2' else {}
    model = build(a.model, D.n_mels, D.n_classes, a.hidden, a.bits, a.k, k_out=a.k_out, K=a.K,
                  dilations=[int(v) for v in a.dilations.split(',')], theta_init=a.theta_init, **kw).cuda()
    set_mode(model, a.mode)
    if a.init:
        sd = torch.load(a.init, map_location='cuda'); model.load_state_dict(sd)
    teacher = None
    if a.kd_logits:
        teacher = torch.from_numpy(np.load(a.kd_logits)).cuda()
        if a.clean_only: teacher = teacher[: len(D.X['train'])]
        assert len(teacher) == len(D.X['train'])
    json.dump(vars(a), open(os.path.join(run, 'config.json'), 'w'), indent=1)
    cw = torch.ones(D.n_classes, device='cuda')
    for kv in [x for x in a.class_weight.split(',') if x]: k, v = kv.split(':'); cw[int(k)] = float(v)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = max(1, a.epochs * ((len(D.X['train']) + a.bs - 1) // a.bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.15)
    gen = torch.Generator(device='cuda'); gen.manual_seed(a.seed)
    best, log = -1.0, []
    if not a.eval_only:
        for ep in range(a.epochs):
            model.train(); t0 = time.time(); tl = 0; n = 0; rate_sum = 0
            for idx, xb, yb in D.batches('train', a.bs, shuffle=True, gen=gen):
                o = model(xb)
                loss = F.cross_entropy(o['logits'], yb, weight=cw)
                if teacher is not None and a.kd_alpha > 0:
                    loss = loss + a.kd_alpha * kd_loss(o['logits'], teacher[idx], a.kd_tau)
                if a.rate_reg > 0:
                    rr = torch.stack([s.mean() for s in o['spikes']])
                    loss = loss + a.rate_reg * (F.relu(rr - a.rate_target) ** 2).sum()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step(); sched.step()
                tl += loss.item() * len(xb); n += len(xb); rate_sum += np.mean([s.mean().item() for s in o['spikes']]) * len(xb)
            va, _ = evaluate(model, D.X['val'], D.y['val'])
            log.append(dict(epoch=ep, loss=tl / n, val_acc=va, train_rate=rate_sum / n, time=time.time() - t0))
            print(f"ep {ep:3d} loss {tl/n:.4f} val mean/last/max {va['mean']:.4f}/{va['last']:.4f}/{va['max']:.4f} rate {rate_sum/n:.3f} ({time.time()-t0:.0f}s)", flush=True)
            if va['mean'] > best:
                best = va['mean']; torch.save(model.state_dict(), os.path.join(run, 'best.pt'))
        json.dump(log, open(os.path.join(run, 'log.json'), 'w'))
        model.load_state_dict(torch.load(os.path.join(run, 'best.pt')))
    elif not os.path.exists(os.path.join(run, 'best.pt')):
        torch.save(model.state_dict(), os.path.join(run, 'best.pt'))
    acc, logits, rates, spikes, vmax = evaluate(model, D.X['test'], D.y['test'], full=True)
    res = dict(model=a.model, mode=a.mode, bits=a.bits, hidden=a.hidden, k=a.k, best_val=best, test_acc=acc,
               layer_names=model.layer_names(), spike_rate_per_layer=rates, spikes_per_sample_per_layer=spikes,
               vmax_per_layer=vmax, v_bits_needed=[int(math.ceil(math.log2(max(v, 1) + 1)) + 1) for v in vmax],
               classes=D.classes, kd=bool(a.kd_logits and a.kd_alpha > 0))
    if a.model == 'M2':
        res['amax'] = None
    pred = logits.argmax(1)
    res['per_class_acc'] = {c: float((pred[D.y['test'] == i] == i).float().mean()) for i, c in enumerate(D.classes)}
    for g, Xg in D.gain_tests.items():
        res[f'test_acc_{g}'] = evaluate(model, Xg, D.y['test'])[0]['mean']
    res['hw'] = hw_cost(model, D, spikes)
    if a.mode == 'int':
        th = {}
        for nm, m in model.named_modules():
            if isinstance(m, LIFCell):
                pass
        res['in_shift'] = {nm: m.in_shift for nm, m in model.named_modules() if isinstance(m, QLinear) and m.is_input}
    json.dump(res, open(os.path.join(run, 'results.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ('per_class_acc', 'hw')}, indent=1))
    print('hw:', json.dumps({k: v for k, v in res['hw'].items() if k != 'layers'}))

if __name__ == '__main__':
    main()
