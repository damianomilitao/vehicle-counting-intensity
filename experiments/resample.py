"""Exact linear-weight versions of cv2.resize (INTER_AREA / INTER_LINEAR) restricted to a sub-window.
The ROI mean at a processing size (w,h) is mean_{(i,j) in mask} O[i,j], with O = Wy @ G @ Wx^T."""
import numpy as np, math

def area_weights(n_src, n_dst):
    scale = n_src / n_dst
    W = np.zeros((n_dst, n_src), np.float64)
    if abs(scale - round(scale)) < 1e-9:            # integer factor: plain box average (resizeAreaFast)
        s = int(round(scale))
        for d in range(n_dst):
            W[d, d * s:(d + 1) * s] = 1.0 / s
        return W
    for d in range(n_dst):
        fs1 = d * scale; fs2 = fs1 + scale
        cell = min(scale, n_src - fs1)
        s1 = math.ceil(fs1); s2 = math.floor(fs2)
        s2 = min(s2, n_src - 1) if s2 > n_src - 1 else s2
        if s1 - fs1 > 1e-3:
            W[d, s1 - 1] += (s1 - fs1) / cell
        for s in range(s1, min(s2, n_src)):
            W[d, s] += 1.0 / cell
        if fs2 - s2 > 1e-3 and s2 < n_src:
            W[d, s2] += min(min(fs2 - s2, 1.0), cell) / cell
    W /= W.sum(1, keepdims=True)
    return W

def linear_weights(n_src, n_dst):
    scale = n_src / n_dst
    W = np.zeros((n_dst, n_src), np.float64)
    for d in range(n_dst):
        fx = (d + 0.5) * scale - 0.5
        sx = math.floor(fx); fx -= sx
        if sx < 0:
            fx, sx = 0.0, 0
        if sx >= n_src - 1:
            fx, sx = 0.0, n_src - 1
        a0 = round((1 - fx) * 2048) / 2048.0; a1 = 1.0 - a0       # OpenCV fixed-point coefficients
        W[d, sx] += a0
        if sx + 1 < n_src:
            W[d, sx + 1] += a1
        else:
            W[d, sx] += a1
    return W

def weights(n_src, n_dst, interp):
    return area_weights(n_src, n_dst) if interp == 'area' else linear_weights(n_src, n_dst)
