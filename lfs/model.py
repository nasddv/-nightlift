"""FreqSSM-Lite: a small, CPU-trainable low-light enhancer in the spirit of ExpoMamba.

Same ingredients as ExpoMamba (U-Net, frequency-domain processing, selective state-space
scan), scaled down so it trains on a laptop CPU with NumPy only:

  * U-Net encoder/decoder (3 scales) with residual conv blocks;
  * FreqBlock  - adaptive Fourier filter: the log-amplitude spectrum of the features drives a
                 1x1 conv that outputs a per-frequency gain m = 1 + 0.5*tanh(.), applied as
                 Re(IFFT(FFT(x) * m)). Initialised to the identity (m = 1);
  * SSMBlock   - 2-D selective scan (VMamba-style cross scan in 4 directions) with an
                 input-dependent decay a = sigmoid(Wa x) and gate silu(Wz x), i.e. a Mamba S6
                 recurrence with a diagonal per-channel state;
  * global residual: output = input + predicted correction.

The gain-map branch of FreqBlock treats the input log-amplitude as a constant (stop-gradient)
to keep the hand-written backward simple; gradients still reach its 1x1 conv and the features.
"""
import numpy as np

from .autograd import (Tensor, no_grad, add, concat, conv2d, flip, fourier_filter, leaky_relu, mul, sigmoid,
                       selective_scan_w, silu, swap_hw, tanh, upsample2x)


class Module:
    def params(self):
        out = []
        for v in self.__dict__.values():
            if isinstance(v, Tensor) and v.requires_grad:
                out.append(v)
            elif isinstance(v, Module):
                out += v.params()
            elif isinstance(v, (list, tuple)):
                for m in v:
                    if isinstance(m, Module):
                        out += m.params()
        return out

    def named_params(self, prefix=""):
        out = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Tensor) and v.requires_grad:
                out[prefix + k] = v
            elif isinstance(v, Module):
                out.update(v.named_params(prefix + k + "."))
            elif isinstance(v, (list, tuple)):
                for i, m in enumerate(v):
                    if isinstance(m, Module):
                        out.update(m.named_params(f"{prefix}{k}.{i}."))
        return out


class Conv(Module):
    def __init__(self, rng, cin, cout, k=3, stride=1, zero=False, bias_init=0.0):
        std = np.sqrt(2.0 / (cin * k * k))
        w = np.zeros((cout, cin, k, k)) if zero else rng.normal(0, std, (cout, cin, k, k))
        self.w = Tensor(w, requires_grad=True)
        self.b = Tensor(np.full(cout, bias_init), requires_grad=True)
        self.stride = stride

    def __call__(self, x):
        return conv2d(x, self.w, self.b, self.stride)


class ResBlock(Module):
    def __init__(self, rng, c):
        self.c1, self.c2 = Conv(rng, c, c), Conv(rng, c, c)
        self.c2.w.data *= 0.1  # small residual at init

    def __call__(self, x):
        return x + self.c2(leaky_relu(self.c1(x)))


class FreqBlock(Module):
    def __init__(self, rng, c):
        self.gain = Conv(rng, c, c, k=1, zero=True)  # zero -> m = 1 (identity) at init
        self.proj = Conv(rng, c, c, k=1)
        self.proj.w.data *= 0.1

    def __call__(self, x):
        logamp = Tensor(np.log1p(np.abs(np.fft.fft2(x.data, norm="ortho"))))  # constant input
        m = add(1.0, mul(0.5, tanh(self.gain(logamp))))
        return x + self.proj(fourier_filter(x, m))


class SSMBlock(Module):
    def __init__(self, rng, c):
        self.inp = Conv(rng, c, c, k=1)
        self.dec = Conv(rng, c, c, k=1, bias_init=2.0)  # a ~ sigmoid(2) = 0.88 -> long memory
        self.dec.w.data *= 0.1
        self.gate = Conv(rng, c, c, k=1)
        self.out = Conv(rng, c, c, k=1)
        self.out.w.data *= 0.1

    @staticmethod
    def _scan4(u, a):
        y = selective_scan_w(u, a)                                          # left -> right
        y = y + flip(selective_scan_w(flip(u, 3), flip(a, 3)), 3)           # right -> left
        ut, at = swap_hw(u), swap_hw(a)
        y = y + swap_hw(selective_scan_w(ut, at))                           # top -> bottom
        y = y + swap_hw(flip(selective_scan_w(flip(ut, 3), flip(at, 3)), 3))  # bottom -> top
        return y

    def __call__(self, x):
        u = self.inp(x)
        a = sigmoid(self.dec(x))
        y = mul(self._scan4(u, a), 0.25)
        return x + self.out(mul(y, silu(self.gate(x))))


class FreqSSMLite(Module):
    """Input (N,3,H,W) in [0,1], H and W divisible by 4 (use enhance() for any size)."""

    def __init__(self, c=16, seed=0):
        rng = np.random.default_rng(seed)
        self.inc = Conv(rng, 3, c)
        self.e1 = ResBlock(rng, c)
        self.down1 = Conv(rng, c, 2 * c, stride=2)
        self.e2 = ResBlock(rng, 2 * c)
        self.f2 = FreqBlock(rng, 2 * c)
        self.down2 = Conv(rng, 2 * c, 4 * c, stride=2)
        self.b1 = ResBlock(rng, 4 * c)
        self.ssm = SSMBlock(rng, 4 * c)
        self.fb = FreqBlock(rng, 4 * c)
        self.b2 = ResBlock(rng, 4 * c)
        self.up2 = Conv(rng, 6 * c, 2 * c)
        self.d2 = ResBlock(rng, 2 * c)
        self.up1 = Conv(rng, 3 * c, c)
        self.d1 = ResBlock(rng, c)
        self.outc = Conv(rng, c, 3)
        self.outc.w.data *= 0.1

    def __call__(self, x):
        e1 = self.e1(leaky_relu(self.inc(x)))
        e2 = self.f2(self.e2(leaky_relu(self.down1(e1))))
        b = leaky_relu(self.down2(e2))
        b = self.b2(self.fb(self.ssm(self.b1(b))))
        d2 = self.d2(leaky_relu(self.up2(concat([upsample2x(b), e2]))))
        d1 = self.d1(leaky_relu(self.up1(concat([upsample2x(d2), e1]))))
        return x + self.outc(d1)

    def enhance(self, img):
        """img: (N,3,H,W) float array in [0,1], any size -> enhanced array, clipped to [0,1]."""
        N, C, H, W = img.shape
        ph, pw = (-H) % 4, (-W) % 4
        x = np.pad(img, ((0, 0), (0, 0), (0, ph), (0, pw)), mode="reflect") if ph or pw else img
        with no_grad():
            y = self(Tensor(x)).data[:, :, :H, :W]
        return np.clip(y, 0, 1)

    def state_dict(self):
        return {k: v.data.copy() for k, v in self.named_params().items()}

    def load_state_dict(self, sd):
        for k, v in self.named_params().items():
            v.data[...] = sd[k]
