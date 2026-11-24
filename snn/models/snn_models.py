"""
snn_models.py — candidate SNNs M1 / M2 / M3 built from lif.py blocks.
All models: forward(x) with x uint8/float [B, T, n_mels]; frame t = timestep t.
Returns dict(logits=[B,C] (mean over t of output membrane, real units), vout=[B,T,C] (raw units),
             spikes=list of [B,T,N] per hidden layer, vmax=list of max|v| per layer (raw units)).
"""
import math, torch, torch.nn as nn
from .lif import QLinear, LIFCell, ALIFCell, OutIntegrator, set_mode

class M1(nn.Module):
    """feed-forward: in -> LIF H -> LIF H -> out"""
    name = 'M1'
    def __init__(self, n_in, n_out, hidden=128, k=(3, 3, 3), bits=8, theta_init=1.0):
        super().__init__()
        self.fc1 = QLinear(n_in, hidden, bits, is_input=True); self.lif1 = LIFCell(hidden, k[0], theta_init)
        self.fc2 = QLinear(hidden, hidden, bits);              self.lif2 = LIFCell(hidden, k[1], theta_init)
        self.fc3 = QLinear(hidden, n_out, bits);               self.out = OutIntegrator(n_out, k[2])
        self.mode = 'float'
    def forward(self, x):
        B, T, _ = x.shape
        x = x.float()
        I1_all = self.fc1(x)                        # [B,T,H] (input layer = only multiplies)
        q1, q2 = self.fc1.q(), self.fc2.q()
        v1 = torch.zeros(B, self.lif1.n, device=x.device); v2 = torch.zeros(B, self.lif2.n, device=x.device)
        vo = torch.zeros(B, self.out.n, device=x.device)
        s1s, s2s, vos = [], [], []
        vmax = [0.0, 0.0, 0.0]
        for t in range(T):
            v1, s1 = self.lif1.step(v1, I1_all[:, t], q1)
            v2, s2 = self.lif2.step(v2, self.fc2(s1), q2)
            vo = self.out.step(vo, self.fc3(s2))
            s1s.append(s1); s2s.append(s2); vos.append(vo)
            vmax = [max(vmax[0], v1.abs().max().item()), max(vmax[1], v2.abs().max().item()), max(vmax[2], vo.abs().max().item())]
        vout = torch.stack(vos, 1)
        qo = self.fc3.q() if self.mode == 'int' else 1.0
        return dict(logits=vout.mean(1) * qo, vout=vout, spikes=[torch.stack(s1s, 1), torch.stack(s2s, 1)], vmax=vmax)
    def layer_names(self): return ['lif1', 'lif2', 'out']

class M2(nn.Module):
    """recurrent ALIF H (-> optional LIF H) -> out"""
    name = 'M2'
    def __init__(self, n_in, n_out, hidden=128, k=(3, 3, 3), bits=8, theta_init=1.0, second_layer=False, ka=5, m=2, adapt=True):
        super().__init__()
        self.fc1 = QLinear(n_in, hidden, bits, is_input=True)
        self.rec = QLinear(hidden, hidden, bits, init_scale=0.3)
        self.alif = ALIFCell(hidden, k[0], ka, m, theta_init, beta_init=0.5 * theta_init, adapt=adapt)
        self.second = second_layer
        if second_layer:
            self.fc2 = QLinear(hidden, hidden, bits); self.lif2 = LIFCell(hidden, k[1], theta_init)
            self.fc3 = QLinear(hidden, n_out, bits)
        else:
            self.fc3 = QLinear(hidden, n_out, bits)
        self.out = OutIntegrator(n_out, k[2])
        self.mode = 'float'
    def forward(self, x):
        B, T, _ = x.shape
        x = x.float()
        I1_all = self.fc1(x)
        q1 = self.fc1.q()
        H = self.alif.n
        v1 = torch.zeros(B, H, device=x.device); a1 = torch.zeros(B, H, device=x.device); s1 = torch.zeros(B, H, device=x.device)
        vo = torch.zeros(B, self.out.n, device=x.device)
        if self.second:
            v2 = torch.zeros(B, H, device=x.device); q2 = self.fc2.q()
        s1s, s2s, vos = [], [], []
        vmax = [0.0, 0.0, 0.0]; amax = 0.0
        for t in range(T):
            # recurrent and feed-forward currents must share one integer grid: the recurrent
            # weights are quantised with the SAME scale factor as the input layer's q (see rec_q)
            I = I1_all[:, t] + self.rec_current(s1, q1)
            v1, a1, s1 = self.alif.step_adapt(v1, a1, I, s1, q1)
            h = s1
            if self.second:
                v2, s2 = self.lif2.step(v2, self.fc2(s1), q2); h = s2; s2s.append(s2)
                vmax[1] = max(vmax[1], v2.abs().max().item())
            vo = self.out.step(vo, self.fc3(h))
            s1s.append(s1); vos.append(vo)
            vmax[0] = max(vmax[0], v1.abs().max().item()); vmax[2] = max(vmax[2], vo.abs().max().item()); amax = max(amax, a1.max().item())
        vout = torch.stack(vos, 1)
        qo = self.fc3.q() if self.mode == 'int' else 1.0
        sp = [torch.stack(s1s, 1)] + ([torch.stack(s2s, 1)] if self.second else [])
        return dict(logits=vout.mean(1) * qo, vout=vout, spikes=sp, vmax=vmax, amax=amax)
    def rec_quant(self, q1):
        """recurrent weights on the input layer's integer grid q1 with an extra left shift rsh so that
        the b-bit integer range is used: W_rec_int = round(W_rec / (q1 * 2^rsh)), I_rec = (s @ W_rec_int^T) << rsh."""
        from .lif import round_ste
        qmax = 2 ** (self.rec.bits - 1) - 1
        wmax = self.rec.W.detach().abs().max().clamp_min(1e-8)
        rsh = int(max(0, math.ceil(math.log2(float(wmax / q1 / qmax)))))
        W_int = torch.clamp(round_ste(self.rec.W.double() / (q1.double() * 2 ** rsh).expand_as(self.rec.W)), -qmax, qmax).to(self.rec.W.dtype)
        return W_int, rsh
    def rec_current(self, s_prev, q1):
        if self.mode == 'float':
            return torch.nn.functional.linear(s_prev, self.rec.W)
        W_int, rsh = self.rec_quant(q1)
        return torch.nn.functional.linear(s_prev, W_int) * (2 ** rsh)
    def export_rec(self):
        W_int, rsh = self.rec_quant(self.fc1.q())
        return W_int.detach().cpu().numpy().astype('int64'), rsh
    def layer_names(self): return ['alif1'] + (['lif2'] if self.second else []) + ['out']

class M3(nn.Module):
    """spiking temporal conv: causal dilated conv (kernel K) layers with dilations, 1-bit spike history."""
    name = 'M3'
    def __init__(self, n_in, n_out, hidden=64, k=(3, 3, 3, 3, 3), bits=8, theta_init=1.0, dilations=(1, 2, 4, 8), K=2):
        super().__init__()
        self.K, self.dil = K, list(dilations)
        self.convs = nn.ModuleList(); self.lifs = nn.ModuleList()
        n_prev = n_in
        for i, d in enumerate(self.dil):
            # one QLinear per tap so every tap shares the layer scale: stack taps along input dim
            self.convs.append(QLinear(n_prev * K, hidden, bits, is_input=(i == 0)))
            self.lifs.append(LIFCell(hidden, k[i], theta_init))
            n_prev = hidden
        self.fc_out = QLinear(hidden, n_out, bits)
        self.out = OutIntegrator(n_out, k[len(self.dil)])
        self.mode = 'float'
    def forward(self, x):
        B, T, _ = x.shape
        x = x.float()
        dev = x.device
        qs = [c.q() for c in self.convs]
        vs = [torch.zeros(B, l.n, device=dev) for l in self.lifs]
        hist = [[] for _ in self.dil]            # hist[i] = list of layer-i INPUT tensors over time
        vo = torch.zeros(B, self.out.n, device=dev)
        spk = [[] for _ in self.dil]; vos = []
        vmax = [0.0] * (len(self.dil) + 1)
        zero_in = [None] * len(self.dil)
        for t in range(T):
            inp = x[:, t]
            for i, (conv, lif, d) in enumerate(zip(self.convs, self.lifs, self.dil)):
                hist[i].append(inp)
                if zero_in[i] is None:
                    zero_in[i] = torch.zeros_like(inp)
                taps = [hist[i][t - j * d] if t - j * d >= 0 else zero_in[i] for j in range(self.K)]
                I = conv(torch.cat(taps, -1))
                vs[i], s = lif.step(vs[i], I, qs[i])
                vmax[i] = max(vmax[i], vs[i].abs().max().item())
                spk[i].append(s); inp = s
            vo = self.out.step(vo, self.fc_out(inp))
            vmax[-1] = max(vmax[-1], vo.abs().max().item()); vos.append(vo)
        vout = torch.stack(vos, 1)
        qo = self.fc_out.q() if self.mode == 'int' else 1.0
        return dict(logits=vout.mean(1) * qo, vout=vout, spikes=[torch.stack(s, 1) for s in spk], vmax=vmax)
    def layer_names(self): return [f'tcn{i+1}' for i in range(len(self.dil))] + ['out']

def build(name, n_in, n_out, hidden, bits, k, k_out=None, K=2, dilations=(1, 2, 4, 8), **kw):
    ko = k if k_out is None else k_out
    if name == 'M1': return M1(n_in, n_out, hidden, k=(k, k, ko), bits=bits, **kw)
    if name == 'M2': return M2(n_in, n_out, hidden, k=(k, k, ko), bits=bits, **kw)
    if name == 'M3': return M3(n_in, n_out, hidden, k=(k,) * len(dilations) + (ko,), bits=bits, dilations=tuple(dilations), K=K, **kw)
    raise ValueError(name)
