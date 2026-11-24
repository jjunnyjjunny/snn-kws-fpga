"""
lif.py — pure-PyTorch integer-friendly spiking layers (no SNN library).

Two execution modes share the same parameters:
  mode='float' : v = v*(1-2^-k) + I,           I = x @ W^T,  spike = v >= theta, v -= theta
  mode='int'   : (QAT, exact integer arithmetic emulated in float32 with STE)
                 W_int   = round(W / s),  s = max|W| / (2^(b-1)-1)   (per-layer symmetric)
                 I_int   = floor((x_int @ W_int^T) / 2^in_shift)      (input layer, x_int = 0..255)
                 I_int   = s_prev @ W_int^T                           (spike inputs, no multiply)
                 v       = clip16( v - floor(v / 2^k) + I_int )       (== v - (v >> k) + I  in RTL)
                 spike   = v >= theta_int,  v -= theta_int * spike    (subtractive reset)
                 theta_int = round(theta_real / q),  q = s * 2^in_shift / 256  (input) or s (hidden)
All integer values stay < 2^24 so float32 arithmetic is exact (TF32 must be disabled).
Surrogate gradient on u = (v - theta)/theta: fast-sigmoid 1/(1+a|u|)^2 or atan.
"""
import math, torch, torch.nn as nn, torch.nn.functional as F

X_SCALE = 1.0 / 256.0          # float-mode input scaling of the 8-bit feature
V_BITS_HIDDEN = 16
V_BITS_OUT = 24
IN_SHIFT = {8: 6, 6: 4, 4: 2}  # input-layer accumulator right shift per weight bit width

class _RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): return torch.floor(x + 0.5)   # round-half-up: deterministic across CPU/GPU (torch.round tie handling differed)
    @staticmethod
    def backward(ctx, g): return g

class _FloorSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): return torch.floor(x)
    @staticmethod
    def backward(ctx, g): return g

round_ste, floor_ste = _RoundSTE.apply, _FloorSTE.apply

class SpikeFn(torch.autograd.Function):
    """spike = heaviside(u), u = (v - theta)/theta ; surrogate grad on u."""
    sg_type, sg_scale, sg_gain = 'fastsig', 4.0, 1.0
    @staticmethod
    def forward(ctx, u):
        ctx.save_for_backward(u)
        return (u >= 0).to(u.dtype)
    @staticmethod
    def backward(ctx, g):
        (u,) = ctx.saved_tensors
        a = SpikeFn.sg_scale
        if SpikeFn.sg_type == 'atan':
            d = (a / 2) / (1 + (math.pi / 2 * a * u) ** 2)
        else:
            d = 1.0 / (1.0 + a * u.abs()) ** 2
        return g * d * SpikeFn.sg_gain

def quant_weight(W, bits):
    qmax = 2 ** (bits - 1) - 1
    # scale computed as a 1-element float64 tensor division (no scalar-reciprocal path) so that
    # CPU and GPU produce bit-identical s, W_int, theta_int, b_int
    s = (W.detach().abs().max().clamp_min(1e-8).double().reshape(1)
         / torch.full((1,), float(qmax), dtype=torch.float64, device=W.device))
    W_int = torch.clamp(round_ste(W.double() / s.expand_as(W)), -qmax, qmax).to(W.dtype)
    s = s.reshape(())
    return W_int, s

class QLinear(nn.Module):
    """y = x @ W^T (no bias). int mode returns exact integer currents."""
    def __init__(self, n_in, n_out, bits=8, is_input=False, init_scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n_out, n_in))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        with torch.no_grad(): self.W.mul_(init_scale)
        self.b = nn.Parameter(torch.zeros(n_out))      # per-neuron constant current (int: added each frame, 16-bit)
        self.bits, self.is_input = bits, is_input
        self.mode = 'float'
    @property
    def in_shift(self):
        return IN_SHIFT[self.bits] if self.is_input else 0
    def q(self):
        """real units per integer unit of the output current (int mode)."""
        _, s = quant_weight(self.W, self.bits)
        return s * (2 ** self.in_shift) * X_SCALE if self.is_input else s
    def b_int(self):
        return torch.clamp(round_ste(self.b.double() / self.q().double().expand_as(self.b)), -32768, 32767).to(self.b.dtype)
    def forward(self, x):
        if self.mode == 'float':
            return F.linear(x * X_SCALE if self.is_input else x, self.W, self.b)
        W_int, s = quant_weight(self.W, self.bits)
        I = F.linear(x, W_int)
        if self.is_input and self.in_shift > 0:
            I = floor_ste(I / 2 ** self.in_shift)
        return I + self.b_int()
    def export(self):
        W_int, s = quant_weight(self.W, self.bits)
        return W_int.detach().cpu().numpy().astype('int64'), float(s), self.in_shift

class LIFCell(nn.Module):
    """Leaky integrate-and-fire with shift leak k, per-neuron threshold, subtractive reset."""
    def __init__(self, n, k=3, theta_init=1.0, v_bits=V_BITS_HIDDEN):
        super().__init__()
        self.n, self.k, self.v_bits = n, k, v_bits
        self.theta = nn.Parameter(torch.full((n,), float(theta_init)))
        self.mode = 'float'
    def theta_int(self, q):
        q = torch.as_tensor(q, device=self.theta.device)
        return torch.clamp(round_ste(self.theta.clamp_min(1e-6).double() / q.double().expand_as(self.theta)), min=1.0).to(self.theta.dtype)
    def leak(self, v):
        if self.mode == 'float':
            return v - v * (2.0 ** -self.k)
        return v - floor_ste(v / 2 ** self.k)
    def clip(self, v):
        if self.mode == 'float':
            return v
        lim = 2 ** (self.v_bits - 1)
        return torch.clamp(v, -lim, lim - 1)
    def step(self, v, I, q, theta_eff=None):
        v = self.clip(self.leak(v) + I)
        th = (self.theta.clamp_min(0.05) if self.mode == 'float' else self.theta_int(q)) if theta_eff is None else theta_eff
        s = SpikeFn.apply((v - th) / th)
        v = v - s.detach() * th            # detach_reset
        return v, s

class ALIFCell(LIFCell):
    """adaptive threshold: a = a - a>>ka + beta*s ; theta_eff = theta + a>>m (m fixed)."""
    detach_adapt = True       # treat the adaptation variable as non-differentiable (stabilises BPTT)
    def __init__(self, n, k=3, ka=5, m=2, theta_init=1.0, beta_init=0.5, v_bits=V_BITS_HIDDEN, adapt=True):
        super().__init__(n, k, theta_init, v_bits)
        self.ka, self.m, self.adapt = ka, m, adapt
        self.beta = nn.Parameter(torch.full((n,), float(beta_init) if adapt else 0.0), requires_grad=adapt)
    def beta_int(self, q):
        q = torch.as_tensor(q, device=self.beta.device)
        return torch.clamp(round_ste(self.beta.clamp_min(0.0).double() / q.double().expand_as(self.beta)), min=0.0).to(self.beta.dtype)
    def step_adapt(self, v, a, I, s_prev, q):
        sp = s_prev.detach() if self.detach_adapt else s_prev
        if self.mode == 'float':
            a = a - a * (2.0 ** -self.ka) + sp * self.beta.clamp_min(0.0)
            th = self.theta.clamp_min(0.05) + (a.detach() if self.detach_adapt else a) * (2.0 ** -self.m)
        else:
            a = a - floor_ste(a / 2 ** self.ka) + sp * self.beta_int(q)
            a = torch.clamp(a, 0, 2 ** 16 - 1)
            th = self.theta_int(q) + floor_ste((a.detach() if self.detach_adapt else a) / 2 ** self.m)
        v, s = self.step(v, I, q, theta_eff=th)
        return v, a, s

class OutIntegrator(nn.Module):
    """non-spiking leaky integrator; logits = mean_t v (real units)."""
    def __init__(self, n, k=3, v_bits=V_BITS_OUT):
        super().__init__()
        self.n, self.k, self.v_bits = n, k, v_bits
        self.mode = 'float'
    def step(self, v, I):
        if self.mode == 'float':
            return v - v * (2.0 ** -self.k) + I
        lim = 2 ** (self.v_bits - 1)
        return torch.clamp(v - floor_ste(v / 2 ** self.k) + I, -lim, lim - 1)

def set_mode(model, mode):
    for m in model.modules():
        if hasattr(m, 'mode'):
            m.mode = mode
    model.mode = mode
