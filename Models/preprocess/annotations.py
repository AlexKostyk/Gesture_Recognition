from dataclasses import dataclass
from pathlib import Path
import pandas as pd
@dataclass(frozen=True)

class AnnRow:
    attachment_id: str
    text: str
    user_id: str
    height: int
    width: int
    length: float
    train: bool

def read_annotations(csv_path: Path) -> list[AnnRow]:
    df = pd.read_csv(csv_path, sep="\t", engine="python")
    df.columns = df.columns.str.strip()
    df.rename(columns=lambda c: c.replace("\ufeff", ""), inplace=True)
    required = ["attachment_id", "text", "user_id", "height", "width", "length", "train"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"annotations.csv missing columns: {missing}\n"
            f"Columns found: {df.columns.tolist()}"
        )
    rows: list[AnnRow] = []
    for _, r in df.iterrows():
        train_val = r["train"]
        if isinstance(train_val, str):
            train_flag = train_val.strip().lower() in ("true", "1", "yes")
        else:
            train_flag = bool(train_val)
        rows.append(
            AnnRow(
                attachment_id=str(r["attachment_id"]).strip(),
                text=str(r["text"]).strip(),
                user_id=str(r["user_id"]).strip(),
                height=int(r["height"]),
                width=int(r["width"]),
                length=float(r["length"]),
                train=train_flag,
            )
        )
    return rows
