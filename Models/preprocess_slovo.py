from __future__ import annotations
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any
import argparse
import numpy as np
from tqdm import tqdm
import cv2
import json
from preprocess.config import PreprocessConfig
from preprocess.annotations import read_annotations
from preprocess.split import make_stratified_split, SplitRow
from preprocess.resample import resample_time_nearest, resample_mask_nearest
from preprocess.io_npz import save_npz, save_json
from preprocess.mp_extract import MPOptions, MediaPipeHolisticExtractor
_WORKER_CFG: PreprocessConfig | None = None
_WORKER_LABEL2ID: dict[str, int] | None = None
_WORKER_SPLIT_MAP: dict[str, str] | None = None
_WORKER_EXTRACTOR: MediaPipeHolisticExtractor | None = None
MODELS_ROOT = Path(__file__).resolve().parent

def _worker_init(cfg: PreprocessConfig, label2id: dict[str, int], split_map: dict[str, str]) -> None:
    global _WORKER_CFG, _WORKER_LABEL2ID, _WORKER_SPLIT_MAP, _WORKER_EXTRACTOR
    _WORKER_CFG = cfg
    _WORKER_LABEL2ID = label2id
    _WORKER_SPLIT_MAP = split_map
    _WORKER_EXTRACTOR = MediaPipeHolisticExtractor(
        use_hands=cfg.use_hands,
        use_pose=cfg.use_pose,
        opts=MPOptions(
            model_complexity=cfg.mp_model_complexity,
            min_detection_conf=cfg.mp_min_detection_conf,
            min_tracking_conf=cfg.mp_min_tracking_conf,
        ),
    )

def _read_all_frames(video_path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames

def _load_existing_splits(splits_csv: Path) -> dict[str, tuple]:
    if not splits_csv.exists():
        return {}
    import pandas as pd
    df = pd.read_csv(splits_csv)
    out: dict[str, tuple] = {}
    if len(df) == 0:
        return out
    for _, r in df.iterrows():
        aid = str(r["attachment_id"])
        out[aid] = (
            aid,
            str(r["split"]),
            int(r["label"]),
            str(r["text"]),
            str(r["user_id"]),
            int(r["orig_len_frames"]) if "orig_len_frames" in df.columns else -1,
        )
    return out

def _load_meta_from_npz(npz_path: Path) -> dict | None:
    try:
        data = np.load(npz_path, allow_pickle=True)
        if "meta" not in data:
            return None
        meta_raw = data["meta"]
        if isinstance(meta_raw, np.ndarray):
            meta_raw = meta_raw.item()
        if isinstance(meta_raw, bytes):
            meta_raw = meta_raw.decode("utf-8")
        if isinstance(meta_raw, str):
            return json.loads(meta_raw)
        return None
    except Exception:
        return None

def _process_one_row(row_dict: dict[str, Any]) -> tuple[bool, tuple | str]:
    global _WORKER_CFG, _WORKER_LABEL2ID, _WORKER_SPLIT_MAP, _WORKER_EXTRACTOR
    assert _WORKER_CFG is not None
    assert _WORKER_LABEL2ID is not None
    assert _WORKER_SPLIT_MAP is not None
    assert _WORKER_EXTRACTOR is not None
    cfg = _WORKER_CFG
    attachment_id = row_dict["attachment_id"]
    text = row_dict["text"]
    user_id = row_dict["user_id"]
    video_path = cfg.videos_dir / f"{attachment_id}.mp4"
    if not video_path.exists():
        return (False, f"missing_video:{attachment_id}")
    frames = _read_all_frames(video_path)
    if len(frames) < cfg.min_segment_len:
        return (False, f"too_short:{attachment_id}:{len(frames)}")
    xyz, mask = _WORKER_EXTRACTOR.extract_from_bgr_frames(frames)
    xyz_rs = resample_time_nearest(xyz, cfg.T)
    mask_rs = resample_mask_nearest(mask, cfg.T)
    split_name = _WORKER_SPLIT_MAP.get(attachment_id, "train")
    label = _WORKER_LABEL2ID[text]
    out_path = cfg.out_dir / "samples" / f"{attachment_id}.npz"
    meta = {
        "attachment_id": attachment_id,
        "user_id": user_id,
        "text": text,
        "orig_len_frames": len(frames),
        "fixed_T": cfg.T,
        "split": split_name,
    }
    save_npz(out_path, xyz_rs, mask_rs, label, meta)
    return (True, (attachment_id, split_name, label, text, user_id, len(frames)))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("hands", "holistic"), default="hands")
    return parser.parse_args()

def build_config(args: argparse.Namespace) -> PreprocessConfig:
    base = PreprocessConfig()
    if args.mode == "holistic":
        base = replace(
            base,
            out_dir=MODELS_ROOT / "Model_holistic" / "keypoints_out",
            use_hands=True,
            use_pose=True,
        )
    else:
        base = replace(
            base,
            out_dir=MODELS_ROOT / "Model_hands" / "keypoints_out",
            use_hands=True,
            use_pose=False,
        )
    return base

def main():
    cfg = build_config(parse_args())
    if not cfg.annotations_csv.exists():
        raise FileNotFoundError(f"annotations.csv not found: {cfg.annotations_csv}")
    if not cfg.videos_dir.exists():
        raise FileNotFoundError(f"videos_dir not found: {cfg.videos_dir}")
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "samples").mkdir(parents=True, exist_ok=True)
    rows = read_annotations(cfg.annotations_csv)
    if len(rows) == 0:
        raise RuntimeError("annotations.csv is empty")
    texts = sorted({r.text for r in rows})
    label2id = {t: i for i, t in enumerate(texts)}
    id2label = {i: t for t, i in label2id.items()}
    save_json(cfg.out_dir / "labels.json", {"label2id": label2id, "id2label": id2label})
    split_input = [SplitRow(attachment_id=r.attachment_id, label=label2id[r.text]) for r in rows]
    split = make_stratified_split(
        split_input,
        train_frac=cfg.train_frac,
        val_frac=cfg.val_frac,
        test_frac=cfg.test_frac,
        seed=cfg.split_seed,
    )
    split_map: dict[str, str] = {}
    for aid in split.train_ids:
        split_map[aid] = "train"
    for aid in split.val_ids:
        split_map[aid] = "val"
    for aid in split.test_ids:
        split_map[aid] = "test"
    existing_splits = _load_existing_splits(cfg.out_dir / "splits.csv")
    samples_dir = cfg.out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    results_rows = []
    seen_ids = set()
    skipped_existing = 0
    work_items = []
    for r in rows:
        attachment_id = r.attachment_id
        npz_path = samples_dir / f"{attachment_id}.npz"
        if npz_path.exists():
            if attachment_id in existing_splits:
                results_rows.append(existing_splits[attachment_id])
                seen_ids.add(attachment_id)
                skipped_existing += 1
                continue
            meta = _load_meta_from_npz(npz_path)
            if meta is not None:
                text = str(meta.get("text", r.text))
                user_id = str(meta.get("user_id", r.user_id))
                split_name = str(meta.get("split", split_map.get(attachment_id, "train")))
                orig_len = int(meta.get("orig_len_frames", -1))
                label = int(label2id.get(text, -1))
                results_rows.append((attachment_id, split_name, label, text, user_id, orig_len))
                seen_ids.add(attachment_id)
                skipped_existing += 1
                continue
        work_items.append(
            {"attachment_id": r.attachment_id, "text": r.text, "user_id": r.user_id}
        )
    ctx = mp.get_context("spawn")
    num_workers = int(cfg.num_workers)
    print(f"Using num_workers = {num_workers}")
    skipped = 0
    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=ctx,
        initializer=_worker_init,
        initargs=(cfg, label2id, split_map),
    ) as ex:
        futures = [ex.submit(_process_one_row, item) for item in work_items]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Preprocess SLOVO"):
            ok, payload = fut.result()
            if ok:
                if payload[0] not in seen_ids:
                    results_rows.append(payload)
                    seen_ids.add(payload[0])
            else:
                skipped += 1
                print(payload)
    import pandas as pd
    df = pd.DataFrame(
        results_rows,
        columns=["attachment_id", "split", "label", "text", "user_id", "orig_len_frames"],
    )
    df.to_csv(cfg.out_dir / "splits.csv", index=False, encoding="utf-8")
    print(f"Done. Saved to: {cfg.out_dir}")
    print(f"Samples: {len(results_rows)}, skipped_existing: {skipped_existing}, skipped: {skipped}")
    print("Files written: labels.json, splits.csv, samples/*.npz")

if __name__ == "__main__":
    main()
