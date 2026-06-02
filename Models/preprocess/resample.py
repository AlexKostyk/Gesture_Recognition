from __future__ import annotations
import numpy as np

def resample_time_nearest(x: np.ndarray, target_T: int) -> np.ndarray:
    T0 = x.shape[0]
    if T0 == target_T:
        return x
    if T0 <= 0:
        raise ValueError("Empty sequence")
    idx = np.linspace(0, T0 - 1, target_T)
    idx = np.rint(idx).astype(np.int64)
    idx = np.clip(idx, 0, T0 - 1)
    return x[idx]

def resample_mask_nearest(m: np.ndarray, target_T: int) -> np.ndarray:
    T0 = m.shape[0]
    if T0 == target_T:
        return m
    idx = np.linspace(0, T0 - 1, target_T)
    idx = np.rint(idx).astype(np.int64)
    idx = np.clip(idx, 0, T0 - 1)
    return m[idx]
