from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
from sklearn.model_selection import train_test_split
@dataclass(frozen=True)

class SplitRow:
    attachment_id: str
    label: int
@dataclass(frozen=True)

class SplitResult:
    train_ids: set[str]
    val_ids: set[str]
    test_ids: set[str]

def make_stratified_split(
    rows: Iterable[SplitRow],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> SplitResult:
    fr_sum = train_frac + val_frac + test_frac
    if abs(fr_sum - 1.0) > 1e-6:
        raise ValueError(f"Fractions must sum to 1.0, got {fr_sum}")
    rows = list(rows)
    ids = [r.attachment_id for r in rows]
    y = [r.label for r in rows]
    ids_trainval, ids_test, y_trainval, _ = train_test_split(
        ids,
        y,
        test_size=test_frac,
        random_state=seed,
        shuffle=True,
        stratify=y,
    )
    val_size_inside = val_frac / (train_frac + val_frac)
    ids_train, ids_val, _, _ = train_test_split(
        ids_trainval,
        y_trainval,
        test_size=val_size_inside,
        random_state=seed,
        shuffle=True,
        stratify=y_trainval,
    )
    return SplitResult(
        train_ids=set(ids_train),
        val_ids=set(ids_val),
        test_ids=set(ids_test),
    )
