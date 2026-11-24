"""
train_dscnn.py — PyTorch port of 03_train.py DS-CNN (24 ch) trained on integer log-mel features.
Conv2d(10x4, stride 5x4) -> BN -> ReLU6 -> [Conv3x3 -> BN -> ReLU6] x2 -> GAP -> Dropout -> Linear
Used as (1) accuracy reference on the new features, (2) KD teacher (saves train-set logits).

usage: CUDA_VISIBLE_DEVICES=6 python snn/baseline/train_dscnn.py --data snn/data/feat_c12_m40_w480.npz --name dscnn_c12
"""
import os, sys, json, time, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
from snn.data.loader import FeatData

class DSCNN(nn.Module):
    def __init__(self, n_classes, ch=24, mean=0.0, std=1.0):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean)); self.register_buffer('std', torch.tensor(std))
        self.c1 = nn.Conv2d(1, ch, (10, 4), stride=(5, 4), padding=(4, 1), bias=False); self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False); self.b2 = nn.BatchNorm2d(ch)
        self.c3 = nn.Conv2d(ch, ch, 3, padding=1, bias=False); self.b3 = nn.BatchNorm2d(ch)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(ch, n_classes)
    def forward(self, x):                       # x [B,T,M] uint8
        x = ((x.float() - self.mean) / self.std).unsqueeze(1)
        x = self.drop(F.relu6(self.b1(self.c1(x))))
        x = F.relu6(self.b2(self.c2(x)))
        x = F.relu6(self.b3(self.c3(x)))
        x = x.mean((2, 3))
        return self.fc(self.drop(x))

def evaluate(model, X, y, bs=1024):
    model.eval(); correct = 0; logits = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            lo = model(X[i:i + bs]); logits.append(lo)
            correct += (lo.argmax(1) == y[i:i + bs]).sum().item()
    return correct / len(X), torch.cat(logits)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True); ap.add_argument('--name', required=True)
    ap.add_argument('--epochs', type=int, default=40); ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--lr', type=float, default=2e-3); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ch', type=int, default=24)
    ap.add_argument('--class_weight', default='', help='CE class weights, e.g. "10:2,11:3"')
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    run = os.path.join(ROOT, 'snn', 'runs', a.name); os.makedirs(run, exist_ok=True)
    D = FeatData(a.data)
    mean, std = D.X['train'].float().mean().item(), D.X['train'].float().std().item()
    model = DSCNN(D.n_classes, a.ch, mean, std).cuda()
    cw = torch.ones(D.n_classes, device='cuda')
    for kv in [x for x in a.class_weight.split(',') if x]: k, v = kv.split(':'); cw[int(k)] = float(v)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    steps = a.epochs * ((len(D.X['train']) + a.bs - 1) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    gen = torch.Generator(device='cuda'); gen.manual_seed(a.seed)
    best, log = 0.0, []
    json.dump(vars(a) | dict(n_params=n_params, mean=mean, std=std), open(os.path.join(run, 'config.json'), 'w'), indent=1)
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); tl = 0; n = 0
        for _, xb, yb in D.batches('train', a.bs, shuffle=True, gen=gen):
            loss = F.cross_entropy(model(xb), yb, weight=cw, label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tl += loss.item() * len(xb); n += len(xb)
        va, _ = evaluate(model, D.X['val'], D.y['val'])
        log.append(dict(epoch=ep, loss=tl / n, val_acc=va, time=time.time() - t0))
        print(f"ep {ep:3d} loss {tl/n:.4f} val {va:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if va > best:
            best = va; torch.save(model.state_dict(), os.path.join(run, 'best.pt'))
    model.load_state_dict(torch.load(os.path.join(run, 'best.pt')))
    te, te_logits = evaluate(model, D.X['test'], D.y['test'])
    res = dict(n_params=n_params, best_val=best, test_acc=te, classes=D.classes)
    for g, Xg in D.gain_tests.items():
        res[f'test_acc_{g}'] = evaluate(model, Xg, D.y['test'])[0]
    # per-class
    pred = te_logits.argmax(1)
    res['per_class_acc'] = {c: float((pred[D.y['test'] == i] == i).float().mean()) for i, c in enumerate(D.classes)}
    # teacher logits for KD (train incl. aug copies, in dataset order)
    _, tr_logits = evaluate(model, D.X['train'], D.y['train'])
    np.save(os.path.join(run, 'teacher_logits_train.npy'), tr_logits.cpu().numpy().astype(np.float32))
    json.dump(res, open(os.path.join(run, 'results.json'), 'w'), indent=1)
    json.dump(log, open(os.path.join(run, 'log.json'), 'w'))
    print(json.dumps(res, indent=1))

if __name__ == '__main__':
    main()
