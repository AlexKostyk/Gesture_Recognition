from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
MODEL_ROOT = Path(__file__).resolve().parents[1]
@dataclass(frozen=True)

class DatasetConfig:
    data_dir: Path = MODEL_ROOT / "keypoints_out"
    split: str = "train"
    nan_to_num: float = 0.0
    per_sample_normalize: bool = True

class SlovoNPZDataset(Dataset):
    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        self.split = cfg.split
        self.samples_dir = cfg.data_dir / "samples"
        self.splits_csv = cfg.data_dir / "splits.csv"
        self.labels_json = cfg.data_dir / "labels.json"
        df = pd.read_csv(self.splits_csv)
        df = df[df["split"] == cfg.split].reset_index(drop=True)
        if len(df) == 0:
            raise RuntimeError(f"No samples for split='{cfg.split}'")
        self.df = df
        with open(self.labels_json, "r", encoding="utf-8") as f:
            labels = json.load(f)
        self.label2id = labels["label2id"]
        if isinstance(labels["id2label"], list):
            self.id2label = labels["id2label"]
        else:
            self.id2label = {int(k): v for k, v in labels["id2label"].items()}
    def __len__(self) -> int:
        return len(self.df)
    def _load_npz(self, attachment_id: str) -> tuple[np.ndarray, np.ndarray]:
        npz_path = self.samples_dir / f"{attachment_id}.npz"
        data = np.load(npz_path, allow_pickle=True)
        xyz = data["xyz"]
        mask = data["mask"]
        xyz = np.nan_to_num(xyz, nan=self.cfg.nan_to_num).astype(np.float32)
        mask = mask.astype(np.float32)
        return xyz, mask
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        attachment_id = row["attachment_id"]
        label = int(row["label"])
        xyz, mask = self._load_npz(attachment_id)
        x = torch.from_numpy(xyz).float()
        m = torch.from_numpy(mask).float()
        if self.cfg.per_sample_normalize:
            mean = x.mean()
            std = x.std().clamp_min(1e-6)
            x = (x - mean) / std
        x = x.permute(2, 0, 1).contiguous()
        m = m.unsqueeze(0)
        y = torch.tensor(label, dtype=torch.long)
        return x, m, y
