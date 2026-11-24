"""confusion.py — B-7: confusion matrices / per-class accuracy of the final SNN (integer golden) and DS-CNN 24ch on the 12-class test set."""
import os, sys, json, argparse
import numpy as np, torch
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
torch.backends.cuda.matmul.allow_tf32 = False
from snn.golden.export_snn_mem import load_run, extract
from snn.golden.snn_int_ref import run_int
from snn.baseline.train_dscnn import DSCNN

def cm(y, p, n):
    m = np.zeros((n, n), np.int64)
    for a, b in zip(y, p): m[a, b] += 1
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snn', default='M3k3e60_c12_q8_kd'); ap.add_argument('--dscnn', default='dscnn_c12_e100')
    ap.add_argument('--data', default=os.path.join(ROOT, 'snn', 'data', 'feat_c12_m40_w480.npz')); ap.add_argument('--out', default=os.path.join(ROOT, 'snn', 'runs', 'confusion.json'))
    a = ap.parse_args()
    z = np.load(a.data); X, y = z['X_test'], z['y_test']; classes = json.loads(str(z['meta']))['classes']; n = len(classes)
    model, cfg, res = load_run(os.path.join(ROOT, 'snn', 'runs', a.snn)); P = extract(model)
    p_snn = run_int(P, X)['pred_sum']
    c = json.load(open(os.path.join(ROOT, 'snn', 'runs', a.dscnn, 'config.json')))
    d = DSCNN(n, c.get('ch', 24), c['mean'], c['std']).cuda(); d.load_state_dict(torch.load(os.path.join(ROOT, 'snn', 'runs', a.dscnn, 'best.pt'))); d.eval()
    with torch.no_grad():
        p_ds = torch.cat([d(torch.from_numpy(X[i:i+1024]).cuda()).argmax(1) for i in range(0, len(X), 1024)]).cpu().numpy()
    out = {}
    for name, p in (('snn', p_snn), ('dscnn', p_ds)):
        m = cm(y, p, n); out[name] = dict(acc=float((p == y).mean()), per_class={classes[i]: float(m[i, i] / m[i].sum()) for i in range(n)}, cm=m.tolist())
        print(f"== {name} acc {out[name]['acc']:.4f}")
        print("        " + " ".join(f"{cl[:5]:>5s}" for cl in classes))
        for i in range(n): print(f"{classes[i]:>7s} " + " ".join(f"{v:5d}" for v in m[i]) + f"  {m[i,i]/m[i].sum():.3f}")
    out['classes'] = classes
    json.dump(out, open(a.out, 'w'), indent=1)

if __name__ == '__main__':
    main()
