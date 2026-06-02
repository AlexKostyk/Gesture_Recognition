from __future__ import annotations
from pathlib import Path
import json
import numpy as np

def save_npz(path: Path, xyz: np.ndarray, mask: np.ndarray, label: int, meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        xyz=xyz.astype(np.float32),
        mask=mask.astype(np.uint8),
        label=np.int64(label),
        meta=json.dumps(meta, ensure_ascii=False),
    )

def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
