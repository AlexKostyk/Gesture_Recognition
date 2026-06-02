from __future__ import annotations
from dataclasses import dataclass
import numpy as np
@dataclass

class AugmentConfig:
    enable_xyz: bool = True
    p_random_interpolation: float = 0.7
    max_crop_frac: float = 0.15
    p_affine: float = 0.7
    translate: float = 0.02
    scale: float = 0.08
    rotate_deg: float = 10.0
    p_replace: float = 0.0
    enable_masking: bool = True
    p_time_mask: float = 0.30
    time_mask_max_frac: float = 0.10
    p_point_mask: float = 0.30
    point_mask_max_frac: float = 0.10

def _rand_uniform(rng: np.random.Generator, a: float, b: float) -> float:
    return float(rng.uniform(a, b))

def _rotate2d(xy: np.ndarray, angle_rad: float) -> np.ndarray:
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    return xy @ R.T

def random_interpolation_xyz(
    xyz: np.ndarray, mask: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig
) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() >= cfg.p_random_interpolation:
        return xyz, mask
    T, P, C = xyz.shape
    if T < 4:
        return xyz, mask
    max_crop = int(round(T * cfg.max_crop_frac))
    crop_l = int(rng.integers(0, max_crop + 1)) if max_crop > 0 else 0
    crop_r = int(rng.integers(0, max_crop + 1)) if max_crop > 0 else 0
    if crop_l + crop_r >= T - 2:
        return xyz, mask
    xyz_c = xyz[crop_l:T - crop_r]
    mask_c = mask[crop_l:T - crop_r]
    Tc = xyz_c.shape[0]
    idx = np.linspace(0, Tc - 1, T)
    idx0 = np.floor(idx).astype(np.int32)
    idx1 = np.clip(idx0 + 1, 0, Tc - 1)
    w = (idx - idx0).astype(np.float32)
    out_xyz = (1.0 - w)[:, None, None] * xyz_c[idx0] + w[:, None, None] * xyz_c[idx1]
    out_mask = (1.0 - w)[:, None] * mask_c[idx0] + w[:, None] * mask_c[idx1]
    out_mask = (out_mask > 0.5).astype(np.float32)
    return out_xyz.astype(np.float32), out_mask.astype(np.float32)

def random_affine_xyz(
    xyz: np.ndarray, mask: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig
) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() >= cfg.p_affine:
        return xyz, mask
    out = xyz.copy().astype(np.float32)
    valid = mask > 0.5
    if valid.any():
        center = out[:, :, :2][valid].mean(axis=0)
    else:
        center = np.array([0.5, 0.5], dtype=np.float32)
    ang = np.deg2rad(_rand_uniform(rng, -cfg.rotate_deg, cfg.rotate_deg))
    sc = 1.0 + _rand_uniform(rng, -cfg.scale, cfg.scale)
    tx = _rand_uniform(rng, -cfg.translate, cfg.translate)
    ty = _rand_uniform(rng, -cfg.translate, cfg.translate)
    xy = out[:, :, :2].reshape(-1, 2)
    xy = xy - center
    xy = _rotate2d(xy, ang)
    xy = xy * sc
    xy = xy + center + np.array([tx, ty], dtype=np.float32)
    T, P, _ = out.shape
    out[:, :, 0] = xy[:, 0].reshape(T, P)
    out[:, :, 1] = xy[:, 1].reshape(T, P)
    out[:, :, 2] = out[:, :, 2] * sc
    out[:, :, :2] = np.clip(out[:, :, :2], -0.5, 1.5)
    return out, mask

def time_and_point_masking_image(
    x: np.ndarray,
    m: np.ndarray,
    rng: np.random.Generator,
    cfg: AugmentConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if not cfg.enable_masking:
        return x, m
    x = x.copy().astype(np.float32)
    m = m.copy().astype(np.float32)
    _, T, P = x.shape
    if rng.random() < cfg.p_time_mask:
        max_len = max(1, int(round(T * cfg.time_mask_max_frac)))
        t_len = int(rng.integers(1, max_len + 1))
        t0 = int(rng.integers(0, max(1, T - t_len + 1)))
        x[:, t0:t0 + t_len, :] = 0.0
        m[:, t0:t0 + t_len, :] = 0.0
    if rng.random() < cfg.p_point_mask:
        max_len = max(1, int(round(P * cfg.point_mask_max_frac)))
        p_len = int(rng.integers(1, max_len + 1))
        p0 = int(rng.integers(0, max(1, P - p_len + 1)))
        x[:, :, p0:p0 + p_len] = 0.0
        m[:, :, p0:p0 + p_len] = 0.0
    return x, m
