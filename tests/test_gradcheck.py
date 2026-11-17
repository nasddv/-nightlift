"""Finite-difference check of every autograd op and of the full model (float64)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

import lfs.autograd as ag
from lfs.autograd import Tensor

ag.DTYPE = np.float64  # exact checks
rng = np.random.default_rng(0)


def check(name, f, shapes, eps=1e-6, tol=1e-4):
    xs = [Tensor(rng.normal(size=s), requires_grad=True) for s in shapes]
    for x in xs:
        x.data = x.data.astype(np.float64)
    out = f(*xs)
    w = rng.normal(size=out.shape)
    loss = ag.mul(out, w)
    total = loss.data.sum()
    ag.Tensor.backward(ag_sum(loss))
    worst = 0.0
    for x in xs:
        ana = x.grad
        idx = [tuple(rng.integers(0, s) for s in x.shape) for _ in range(6)]
        for i in idx:
            old = x.data[i]
            x.data[i] = old + eps
            fp = (f(*xs).data * w).sum()
            x.data[i] = old - eps
            fm = (f(*xs).data * w).sum()
            x.data[i] = old
            num = (fp - fm) / (2 * eps)
            worst = max(worst, abs(num - ana[i]) / max(1.0, abs(num)))
    status = "ok" if worst < tol else "FAIL"
    print(f"{name:22s} max rel err {worst:.2e}  {status}")
    return worst < tol


def ag_sum(t):
    out = Tensor(t.data.sum(), (t,))
    out._back = lambda g: t._acc(np.full(t.shape, g))
    return out


ok = True
ok &= check("add/mul bcast", lambda a, b: ag.mul(ag.add(a, b), b), [(2, 3, 4, 4), (1, 3, 1, 1)])
ok &= check("sigmoid/tanh/silu", lambda a: ag.silu(ag.tanh(ag.sigmoid(a))), [(2, 3, 4, 4)])
ok &= check("conv k3 s1", lambda x, w, b: ag.conv2d(x, w, b, 1), [(2, 3, 6, 6), (4, 3, 3, 3), (4,)])
ok &= check("conv k3 s2", lambda x, w, b: ag.conv2d(x, w, b, 2), [(2, 3, 8, 8), (4, 3, 3, 3), (4,)])
ok &= check("conv k1", lambda x, w, b: ag.conv2d(x, w, b, 1), [(2, 3, 5, 5), (4, 3, 1, 1), (4,)])
ok &= check("upsample/concat", lambda a, b: ag.concat([ag.upsample2x(a), b]), [(1, 2, 3, 3), (1, 1, 6, 6)])
ok &= check("flip/swap", lambda a: ag.swap_hw(ag.flip(a, 3)), [(1, 2, 3, 5)])
ok &= check("selective scan", lambda u, a: ag.selective_scan_w(u, ag.sigmoid(a)), [(2, 3, 4, 7), (2, 3, 4, 7)])
ok &= check("fourier filter", lambda x, m: ag.fourier_filter(x, m), [(2, 3, 6, 5), (2, 3, 6, 5)])
tgt = rng.normal(size=(2, 3, 6, 6))
ok &= check("charbonnier", lambda p: ag.charbonnier(p, tgt), [(2, 3, 6, 6)])
ok &= check("fft amp l1 (even W)", lambda p: ag.fft_amp_l1(p, tgt), [(2, 3, 6, 6)])
tgt5 = rng.normal(size=(2, 3, 6, 5))
ok &= check("fft amp l1 (odd W)", lambda p: ag.fft_amp_l1(p, tgt5), [(2, 3, 6, 5)])

# full model, a few random parameters
from lfs.model import FreqSSMLite

net = FreqSSMLite(c=4, seed=1)
for p in net.params():
    p.data = p.data.astype(np.float64) + rng.normal(0, 0.05, p.shape)  # break zero/identity inits
for fb in (net.f2, net.fb):  # FreqBlock's gain input is stop-gradient by design: zero it for an exact check
    fb.gain.w.data[...] = 0.0
x = rng.uniform(size=(1, 3, 8, 8))
y = rng.uniform(size=(1, 3, 8, 8))
loss = ag.add(ag.charbonnier(net(Tensor(x)), y), ag.mul(ag.fft_amp_l1(net(Tensor(x)), y), 0.1))
loss.backward()
worst = 0.0
for name, p in list(net.named_params().items())[::3]:
    i = tuple(rng.integers(0, s) for s in p.shape)
    old = p.data[i]
    f = lambda: (ag.charbonnier(net(Tensor(x)), y).data + 0.1 * ag.fft_amp_l1(net(Tensor(x)), y).data)
    p.data[i] = old + 1e-6; fp = f()
    p.data[i] = old - 1e-6; fm = f()
    p.data[i] = old
    num = (fp - fm) / 2e-6
    worst = max(worst, abs(num - p.grad[i]) / max(1e-3, abs(num)))
print(f"{'full model':22s} max rel err {worst:.2e}  {'ok' if worst < 1e-3 else 'FAIL'}")
ok &= worst < 1e-3
print("ALL OK" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
