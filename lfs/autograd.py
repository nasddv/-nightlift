"""Minimal reverse-mode autograd on NumPy (float32), just enough for a small image CNN.

Tensor holds `data` (np.ndarray) and `grad`. Each op returns a new Tensor that remembers its
parents and a closure that pushes the output gradient back to them. Tensor.backward() runs the
closures in reverse topological order. All ops used by the model are here and are checked
against finite differences in tests/test_gradcheck.py.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

DTYPE = np.float32
NO_GRAD = False  # set by no_grad(): no graph is recorded, so intermediates are freed at once


class no_grad:
    def __enter__(self):
        global NO_GRAD
        self.prev, NO_GRAD = NO_GRAD, True

    def __exit__(self, *exc):
        global NO_GRAD
        NO_GRAD = self.prev


class Tensor:
    __slots__ = ("data", "grad", "parents", "_back", "requires_grad")

    def __init__(self, data, parents=(), back=None, requires_grad=False):
        self.data = np.asarray(data, dtype=DTYPE)
        self.grad = None
        if NO_GRAD:
            parents = ()
        self.parents = parents
        self._back = back
        self.requires_grad = requires_grad or any(p.requires_grad for p in parents)

    @property
    def shape(self):
        return self.data.shape

    def _acc(self, g):
        if not self.requires_grad:
            return
        self.grad = g if self.grad is None else self.grad + g

    def backward(self, grad=None):
        order, seen = [], set()

        def visit(t):
            stack = [(t, False)]
            while stack:
                node, done = stack.pop()
                if done:
                    order.append(node)
                    continue
                if id(node) in seen:
                    continue
                seen.add(id(node))
                stack.append((node, True))
                for p in node.parents:
                    if p.requires_grad and id(p) not in seen:
                        stack.append((p, False))

        visit(self)
        self.grad = np.ones_like(self.data) if grad is None else grad
        for node in reversed(order):
            if node._back is not None and node.grad is not None:
                node._back(node.grad)
                node.grad = None  # free intermediate memory (leaves have no _back and keep .grad)

    # operator sugar
    def __add__(self, o):
        return add(self, o)

    def __radd__(self, o):
        return add(self, o)

    def __mul__(self, o):
        return mul(self, o)

    def __rmul__(self, o):
        return mul(self, o)

    def __sub__(self, o):
        return add(self, mul(o, -1.0))


def as_t(x):
    return x if isinstance(x, Tensor) else Tensor(x)


def _unbroadcast(g, shape):
    while g.ndim > len(shape):
        g = g.sum(0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(i, keepdims=True)
    return g


def add(a, b):
    a, b = as_t(a), as_t(b)
    out = Tensor(a.data + b.data, (a, b))

    def back(g):
        a._acc(_unbroadcast(g, a.shape))
        b._acc(_unbroadcast(g, b.shape))

    out._back = back
    return out


def mul(a, b):
    a, b = as_t(a), as_t(b)
    out = Tensor(a.data * b.data, (a, b))

    def back(g):
        if a.requires_grad:
            a._acc(_unbroadcast(g * b.data, a.shape))
        if b.requires_grad:
            b._acc(_unbroadcast(g * a.data, b.shape))

    out._back = back
    return out


def sigmoid(x):
    s = 1.0 / (1.0 + np.exp(-np.clip(x.data, -30, 30)))
    out = Tensor(s, (x,))
    out._back = lambda g: x._acc(g * s * (1 - s))
    return out


def tanh(x):
    t = np.tanh(x.data)
    out = Tensor(t, (x,))
    out._back = lambda g: x._acc(g * (1 - t * t))
    return out


def silu(x):
    s = 1.0 / (1.0 + np.exp(-np.clip(x.data, -30, 30)))
    out = Tensor(x.data * s, (x,))
    out._back = lambda g: x._acc(g * (s * (1 + x.data * (1 - s))))
    return out


def leaky_relu(x, slope=0.1):
    m = np.where(x.data > 0, 1.0, slope).astype(DTYPE)
    out = Tensor(x.data * m, (x,))
    out._back = lambda g: x._acc(g * m)
    return out


def concat(ts, axis=1):
    out = Tensor(np.concatenate([t.data for t in ts], axis=axis), tuple(ts))
    sizes = np.cumsum([t.shape[axis] for t in ts])[:-1]

    def back(g):
        for t, gi in zip(ts, np.split(g, sizes, axis=axis)):
            t._acc(gi)

    out._back = back
    return out


def flip(x, axis):
    out = Tensor(np.flip(x.data, axis), (x,))
    out._back = lambda g: x._acc(np.flip(g, axis))
    return out


def swap_hw(x):
    out = Tensor(np.swapaxes(x.data, 2, 3), (x,))
    out._back = lambda g: x._acc(np.swapaxes(g, 2, 3))
    return out


def conv2d(x, w, b=None, stride=1):
    """'same'-style conv: kernel k (odd), zero padding k//2, stride 1 or 2. x (N,C,H,W), w (O,C,k,k).
    Internally channels-last im2col with (k, k, C) column order so col2im slices are contiguous."""
    N, C, H, W = x.shape
    O, _, k, _ = w.shape
    p = k // 2
    if NO_GRAD:
        return Tensor(_conv2d_nograd(x.data, w.data, None if b is None else b.data, stride))
    xh = x.data.transpose(0, 2, 3, 1)
    xp = np.pad(xh, ((0, 0), (p, p), (p, p), (0, 0))) if p else xh
    win = sliding_window_view(xp, (k, k), axis=(1, 2))[:, ::stride, ::stride]  # N,Ho,Wo,C,k,k
    Ho, Wo = win.shape[1], win.shape[2]
    cols = np.ascontiguousarray(win.transpose(0, 1, 2, 4, 5, 3)).reshape(N * Ho * Wo, k * k * C)
    wm = np.ascontiguousarray(w.data.transpose(0, 2, 3, 1)).reshape(O, -1)
    y = cols @ wm.T
    if b is not None:
        y += b.data
    y = y.reshape(N, Ho, Wo, O).transpose(0, 3, 1, 2)
    parents = (x, w) if b is None else (x, w, b)
    out = Tensor(np.ascontiguousarray(y), parents)

    def back(g):
        gm = np.ascontiguousarray(g.transpose(0, 2, 3, 1)).reshape(N * Ho * Wo, O)
        if w.requires_grad:
            w._acc((gm.T @ cols).reshape(O, k, k, C).transpose(0, 3, 1, 2))
        if b is not None and b.requires_grad:
            b._acc(gm.sum(0))
        if x.requires_grad:
            gc = (gm @ wm).reshape(N, Ho, Wo, k, k, C)
            gxp = np.zeros(xp.shape, DTYPE)
            for i in range(k):
                for j in range(k):
                    gxp[:, i:i + stride * Ho:stride, j:j + stride * Wo:stride] += gc[:, :, :, i, j]
            gxp = gxp[:, p:p + H, p:p + W] if p else gxp
            x._acc(np.ascontiguousarray(gxp.transpose(0, 3, 1, 2)))

    out._back = back
    return out


def _conv2d_nograd(x, w, b, stride, max_bytes=64 << 20):
    """Same result as conv2d's forward, computed in row blocks to bound the im2col memory."""
    N, C, H, W = x.shape
    O, _, k, _ = w.shape
    p = k // 2
    xp = np.pad(x.transpose(0, 2, 3, 1), ((0, 0), (p, p), (p, p), (0, 0))) if p else x.transpose(0, 2, 3, 1)
    Ho, Wo = (H + stride - 1) // stride, (W + stride - 1) // stride
    wm = np.ascontiguousarray(w.transpose(0, 2, 3, 1)).reshape(O, -1).T
    out = np.empty((N, O, Ho, Wo), DTYPE)
    rows = max(1, int(max_bytes // (4 * N * Wo * k * k * C)))
    for r0 in range(0, Ho, rows):
        r1 = min(Ho, r0 + rows)
        seg = xp[:, r0 * stride:(r1 - 1) * stride + k]
        win = sliding_window_view(seg, (k, k), axis=(1, 2))[:, ::stride, ::stride][:, :r1 - r0, :Wo]
        cols = np.ascontiguousarray(win.transpose(0, 1, 2, 4, 5, 3)).reshape(-1, k * k * C)
        y = cols @ wm
        if b is not None:
            y += b
        out[:, :, r0:r1] = y.reshape(N, r1 - r0, Wo, O).transpose(0, 3, 1, 2)
    return out


def upsample2x(x):
    out = Tensor(x.data.repeat(2, 2).repeat(2, 3), (x,))

    def back(g):
        N, C, H, W = g.shape
        x._acc(g.reshape(N, C, H // 2, 2, W // 2, 2).sum((3, 5)))

    out._back = back
    return out


def selective_scan_w(u, a):
    """Input-dependent linear recurrence along the last axis (Mamba-style S6 with a diagonal,
    per-channel state):   h_t = a_t * h_{t-1} + (1 - a_t) * u_t,   h_{-1} = 0.
    u, a: (N, C, H, W), a in (0, 1). Returns h with the same shape."""
    A, U = a.data, u.data
    L = U.shape[-1]
    h = np.empty_like(U)
    prev = np.zeros(U.shape[:-1], DTYPE)
    for t in range(L):
        prev = A[..., t] * prev + (1 - A[..., t]) * U[..., t]
        h[..., t] = prev
    out = Tensor(h, (u, a))

    def back(g):
        gu = np.empty_like(U)
        ga = np.empty_like(A)
        carry = np.zeros(U.shape[:-1], DTYPE)
        for t in range(L - 1, -1, -1):
            gh = g[..., t] + carry
            hp = h[..., t - 1] if t > 0 else 0.0
            gu[..., t] = gh * (1 - A[..., t])
            ga[..., t] = gh * (hp - U[..., t])
            carry = gh * A[..., t]
        u._acc(gu)
        a._acc(ga)

    out._back = back
    return out


def fourier_filter(x, m):
    """y = Re(IFFT2(FFT2(x) * m)), orthonormal FFT, m real with x's shape (a learned
    per-sample, per-channel, per-frequency gain). Gradients for both x and m."""
    X = np.fft.fft2(x.data, norm="ortho")
    y = np.fft.ifft2(X * m.data, norm="ortho").real
    out = Tensor(y, (x, m))

    def back(g):
        G = np.fft.fft2(g, norm="ortho")
        if x.requires_grad:
            x._acc(np.fft.ifft2(G * m.data, norm="ortho").real.astype(DTYPE))
        if m.requires_grad:
            m._acc((X * np.conj(G)).real.astype(DTYPE))

    out._back = back
    return out


def mean(x):
    out = Tensor(x.data.mean(), (x,))
    n = x.data.size
    out._back = lambda g: x._acc(np.full(x.shape, g / n, DTYPE))
    return out


def charbonnier(pred, target, eps=1e-3):
    d = pred.data - target
    r = np.sqrt(d * d + eps * eps)
    out = Tensor(r.mean(), (pred,))
    out._back = lambda g: pred._acc((g * d / r / d.size).astype(DTYPE))
    return out


def fft_amp_l1(pred, target):
    """L1 between FFT amplitude spectra (frequency-consistency loss), full orthonormal FFT."""
    P = np.fft.fft2(pred.data, norm="ortho")
    T = np.abs(np.fft.fft2(target, norm="ortho"))
    A = np.abs(P)
    out = Tensor(np.abs(A - T).mean(), (pred,))

    def back(g):
        v = np.sign(A - T) * (g / A.size) * P / np.maximum(A, 1e-12)
        pred._acc(np.fft.ifft2(v, norm="ortho").real.astype(DTYPE))

    out._back = back
    return out
