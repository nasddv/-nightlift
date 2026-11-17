"""PSNR and SSIM on [0,1] RGB images (N,3,H,W); SSIM = Wang et al. 2004, Gaussian 11x11, sigma 1.5,
averaged over channels (same convention as skimage with channel_axis, data_range=1)."""
import numpy as np


def psnr(x, y):
    mse = ((x.astype(np.float64) - y) ** 2).mean(axis=(1, 2, 3))
    return 10 * np.log10(1.0 / np.maximum(mse, 1e-12))


def _gauss(size=11, sigma=1.5):
    t = np.arange(size) - size // 2
    g = np.exp(-t * t / (2 * sigma * sigma))
    return g / g.sum()


def _filt(a, g):
    """Separable 'valid' Gaussian filter over the last two axes."""
    k = len(g)
    a = np.lib.stride_tricks.sliding_window_view(a, k, axis=-1) @ g
    a = np.lib.stride_tricks.sliding_window_view(a, k, axis=-2) @ g
    return a


def ssim(x, y):
    x, y = x.astype(np.float64), y.astype(np.float64)
    g = _gauss()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mx, my = _filt(x, g), _filt(y, g)
    sxx = _filt(x * x, g) - mx * mx
    syy = _filt(y * y, g) - my * my
    sxy = _filt(x * y, g) - mx * my
    m = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return m.mean(axis=(1, 2, 3))
