"""Lightweight image corruption suite (numpy/scipy/matplotlib only, no cv2).

Loosely modeled on Hendrycks & Dietterich's ImageNet-C corruptions, simplified
for a fast, dependency-light validation suite. Each function takes an HWC
float32 array in [0, 1] plus a severity level in {1, 2, 3, 4, 5} (increasing
severity) and returns a corrupted HWC float32 array in [0, 1].
"""
from __future__ import annotations

import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from scipy.ndimage import convolve

_RNG = np.random.default_rng(0)


def gaussian_noise(img: np.ndarray, severity: int) -> np.ndarray:
    std = [0.02, 0.04, 0.06, 0.09, 0.12][severity - 1]
    noise = _RNG.normal(loc=0.0, scale=std, size=img.shape).astype(np.float32)
    return np.clip(img + noise, 0.0, 1.0)


def _motion_kernel(length: int) -> np.ndarray:
    """Horizontal linear motion-blur kernel of given length, normalized to sum to 1."""
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    return kernel / kernel.sum()


def motion_blur(img: np.ndarray, severity: int) -> np.ndarray:
    length = [3, 5, 9, 13, 17][severity - 1]
    kernel = _motion_kernel(length)
    out = np.stack([convolve(img[..., c], kernel, mode="reflect") for c in range(img.shape[-1])], axis=-1)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def brightness(img: np.ndarray, severity: int) -> np.ndarray:
    delta = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    hsv = rgb_to_hsv(np.clip(img, 0.0, 1.0))
    hsv[..., 2] = np.clip(hsv[..., 2] + delta, 0.0, 1.0)
    return np.clip(hsv_to_rgb(hsv), 0.0, 1.0).astype(np.float32)


def fog(img: np.ndarray, severity: int) -> np.ndarray:
    alpha, contrast = [(0.2, 0.95), (0.35, 0.9), (0.5, 0.85), (0.65, 0.8), (0.8, 0.75)][severity - 1]
    fog_color = np.ones_like(img)
    blended = (1 - alpha) * img + alpha * fog_color
    mean = blended.mean(axis=(0, 1), keepdims=True)
    contrasted = (blended - mean) * contrast + mean
    return np.clip(contrasted, 0.0, 1.0).astype(np.float32)


CORRUPTIONS = {
    "gaussian_noise": gaussian_noise,
    "motion_blur": motion_blur,
    "brightness": brightness,
    "fog": fog,
}
