"""loader.py — load feat_*.npz into GPU tensors (uint8 features, int64 labels)."""
import os, json, numpy as np, torch
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def feat_path(classes, n_mels, win, tag=''):
    return os.path.join(ROOT, 'snn', 'data', f'feat_c{classes}_m{n_mels}_w{win}{tag}.npz')

class FeatData:
    def __init__(self, path, device='cuda', clean_only=False):
        z = np.load(path)
        self.meta = json.loads(str(z['meta']))
        self.classes = self.meta['classes']
        self.n_classes = len(self.classes)
        self.path = path
        Xtr, ytr, cp = z['X_train'], z['y_train'], z['copy_train']
        if clean_only:
            m = cp == 0
            Xtr, ytr, cp = Xtr[m], ytr[m], cp[m]
        self.copy_train = torch.from_numpy(cp.astype(np.int64)).to(device)
        self.X = {'train': torch.from_numpy(Xtr).to(device),
                  'val': torch.from_numpy(z['X_val']).to(device),
                  'test': torch.from_numpy(z['X_test']).to(device)}
        self.y = {'train': torch.from_numpy(ytr).to(device),
                  'val': torch.from_numpy(z['y_val']).to(device),
                  'test': torch.from_numpy(z['y_test']).to(device)}
        self.gain_tests = {}
        for k in z.files:
            if k.startswith('X_test_'):
                self.gain_tests[k[len('X_test_'):]] = torch.from_numpy(z[k]).to(device)
        self.T, self.n_mels = self.X['train'].shape[1:]
        self.device = device

    def batches(self, split, bs, shuffle=False, gen=None):
        X, y = self.X[split], self.y[split]
        n = len(X)
        idx = torch.randperm(n, device=self.device, generator=gen) if shuffle else torch.arange(n, device=self.device)
        for i in range(0, n, bs):
            j = idx[i:i + bs]
            yield j, X[j], y[j]
