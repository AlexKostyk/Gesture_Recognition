from __future__ import annotations
import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from .model_effnet import SlovoEfficientNetB0, SlovoEfficientNetV2S
APP_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = APP_ROOT / "models"
if not MODEL_ROOT.exists():
    MODEL_ROOT = APP_ROOT / "model"
FRONTEND_DIR = APP_ROOT / "frontend"
IMAGES_DIR = APP_ROOT / "imgs"
PROJECT_ROOT = APP_ROOT.parent
DATASET_ROOT = PROJECT_ROOT / "Dataset" / "slovo"
DATASET_VIDEO_DIR = DATASET_ROOT / "data"
ANNOTATIONS_PATH = DATASET_ROOT / "annotations.csv"
T_DEFAULT = 70
NAN_TO_NUM = 0.0
PER_SAMPLE_NORMALIZE = True
EFFNET_INPUT_SIZE = 384
MODE_HANDS = "hands"
MODE_HOLISTIC = "holistic"
GLOSS_TEXT_MODEL_DIR = MODEL_ROOT / "rut5-small-gloss"
GLOSS_TEXT_PREFIX = "gloss to russian: "
GLOSS_TEXT_MAX_SOURCE_LENGTH = 128
GLOSS_TEXT_MAX_NEW_TOKENS = 24
GLOSS_TEXT_NUM_BEAMS = 1
MIN_GESTURE_CONFIDENCE = 0.0
REPEAT_COOLDOWN = 0.8
PAUSE_THRESHOLD = 1.2
FORCE_FLUSH_TIMEOUT = 3.0
ENABLE_PAUSE_SEGMENTATION = False
MAX_SEGMENT_LEN = 10
MIN_SEGMENT_LEN = 2
NEW_SEGMENT_MARKERS = {
    "сегодня",
    "завтра",
    "вчера",
    "потом",
    "утром",
    "вечером",
    "сейчас",
}
SUBJECT_MARKERS = {
    "я",
    "ты",
    "вы",
    "мы",
    "он",
    "она",
    "они",
    "человек",
    "мама",
    "папа",
    "друг",
    "девушка",
    "мальчик",
    "девочка",
    "ребенок",
}
STANDALONE_FINAL_LABELS = {
    "спасибо",
    "да",
    "нет",
    "помощь",
    "здравствуйте",
    "привет",
    "пока",
    "хорошо",
    "понятно",
    "готово",
}
FINAL_PHRASES = {
    ("до", "свидания"),
    ("не", "знаю"),
}
@dataclass(frozen=True)

class RuntimeModeConfig:
    name: str
    p: int
    t: int
    ckpt_path: Path
    labels_path: Path
    architecture: str
MODE_CONFIGS: dict[str, RuntimeModeConfig] = {
    MODE_HANDS: RuntimeModeConfig(
        name=MODE_HANDS,
        p=42,
        t=T_DEFAULT,
        ckpt_path=MODEL_ROOT / "hands" / "hands_model.pt",
        labels_path=MODEL_ROOT / "hands" / "labels.json",
        architecture="effnet_b0",
    ),
    MODE_HOLISTIC: RuntimeModeConfig(
        name=MODE_HOLISTIC,
        p=75,
        t=T_DEFAULT,
        ckpt_path=MODEL_ROOT / "holistic" / "holistic_model.pt",
        labels_path=MODEL_ROOT / "holistic" / "labels.json",
        architecture="effnetv2_s",
    ),
}
app = FastAPI(title="Gesture Recognition Web App")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
if IMAGES_DIR.exists():
    app.mount("/imgs", StaticFiles(directory=IMAGES_DIR), name="imgs")
if DATASET_VIDEO_DIR.exists():
    app.mount("/gesture-media", StaticFiles(directory=DATASET_VIDEO_DIR), name="gesture-media")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS: dict[str, torch.nn.Module] = {}
MODE_LABELS: dict[str, dict[int, str]] = {}
GLOSS_TEXT_TOKENIZER = None
GLOSS_TEXT_MODEL = None
GLOSS_TEXT_LOAD_ERROR: str | None = None
GLOSS_TEXT_WARMED_UP = False
GLOSS_RAW_LABELS: list[str] = []
GLOSS_DRAFT_LABELS: list[str] = []
GLOSS_FINAL_SEGMENTS: list[str] = []
GLOSS_LAST_TOKEN: dict | None = None
GLOSS_LAST_SUBMITTED_SOURCE = ""
GESTURE_VIDEO_EXACT_INDEX: dict[str, dict] = {}
GESTURE_VIDEO_INDEX: dict[str, dict] = {}
DACTYL_VIDEO_INDEX: dict[str, dict] = {}
GESTURE_VIDEO_EXACT_VARIANTS: dict[str, list[dict]] = {}
GESTURE_VIDEO_VARIANTS: dict[str, list[dict]] = {}
DACTYL_VIDEO_VARIANTS: dict[str, list[dict]] = {}
PREFERRED_VIDEO_RESOLUTION: tuple[int, int] | None = None
MAX_GESTURE_WORDS = 1
MORPH_ANALYZER = None
INFER_LOCK = Lock()
GLOSS_TEXT_LOCK = Lock()
GLOSS_TRANSCRIPT_LOCK = Lock()

class KeypointsBatch(BaseModel):
    frames: list[list[list[float]]]
    mode: str = MODE_HANDS
    duration_sec: float | None = None

class TextToGestureRequest(BaseModel):
    text: str
    variant_offset: int = 0

class TokenVariantRequest(BaseModel):
    token: str
    variant_offset: int = 0

def _normalize_label(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())

def _normalize_gloss_label(text: str) -> str:
    return re.sub(r"[.!?]+$", "", _normalize_label(text))

def _normalize_gloss_exact_label(text: str) -> str:
    return re.sub(r"[.!?]+$", "", _normalize_exact_label(text))

def _is_dactyl_label(text: str) -> bool:
    clean_text = _normalize_gloss_exact_label(text)
    return len(clean_text) == 1 and clean_text.isalpha() and clean_text == clean_text.upper()

def _normalize_exact_label(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())

def _decode_text_maybe(text: str) -> str:
    try:
        return text.encode("latin1").decode("utf-8")
    except Exception:
        return text

def _is_real_gesture_label(label: str) -> bool:
    clean_label = str(label or "").strip()
    return bool(clean_label) and clean_label != "no_event"

def _warmup_gloss_text_model(tokenizer, model) -> None:
    global GLOSS_TEXT_WARMED_UP
    if GLOSS_TEXT_WARMED_UP:
        return
    try:
        inputs = tokenizer(
            GLOSS_TEXT_PREFIX + "привет",
            return_tensors="pt",
            truncation=True,
            max_length=GLOSS_TEXT_MAX_SOURCE_LENGTH,
        )
        inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
        with GLOSS_TEXT_LOCK, torch.inference_mode():
            model.generate(
                **inputs,
                max_new_tokens=8,
                num_beams=1,
                do_sample=False,
                use_cache=True,
            )
        GLOSS_TEXT_WARMED_UP = True
    except Exception:
        GLOSS_TEXT_WARMED_UP = False

def _load_gloss_text_model() -> bool:
    global GLOSS_TEXT_TOKENIZER, GLOSS_TEXT_MODEL, GLOSS_TEXT_LOAD_ERROR
    if GLOSS_TEXT_MODEL is not None and GLOSS_TEXT_TOKENIZER is not None:
        return True
    if not GLOSS_TEXT_MODEL_DIR.exists():
        GLOSS_TEXT_LOAD_ERROR = f"Gloss-to-text model not found: {GLOSS_TEXT_MODEL_DIR}"
        return False
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(GLOSS_TEXT_MODEL_DIR))
        model = AutoModelForSeq2SeqLM.from_pretrained(str(GLOSS_TEXT_MODEL_DIR))
        model.to(DEVICE)
        model.eval()
        GLOSS_TEXT_TOKENIZER = tokenizer
        GLOSS_TEXT_MODEL = model
        GLOSS_TEXT_LOAD_ERROR = None
        _warmup_gloss_text_model(tokenizer, model)
        return True
    except Exception as exc:
        GLOSS_TEXT_TOKENIZER = None
        GLOSS_TEXT_MODEL = None
        GLOSS_TEXT_LOAD_ERROR = str(exc)
        return False

def _generate_gloss_text(source_text: str) -> tuple[str, bool, str | None]:
    source_text = _normalize_label(source_text)
    if not source_text:
        return "", False, None
    if not _load_gloss_text_model():
        return source_text, False, GLOSS_TEXT_LOAD_ERROR
    tokenizer = GLOSS_TEXT_TOKENIZER
    model = GLOSS_TEXT_MODEL
    if tokenizer is None or model is None:
        return source_text, False, "Gloss-to-text model is not initialized"
    try:
        inputs = tokenizer(
            GLOSS_TEXT_PREFIX + source_text,
            return_tensors="pt",
            truncation=True,
            max_length=GLOSS_TEXT_MAX_SOURCE_LENGTH,
        )
        inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
        with GLOSS_TEXT_LOCK, torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=GLOSS_TEXT_MAX_NEW_TOKENS,
                num_beams=GLOSS_TEXT_NUM_BEAMS,
                do_sample=False,
                use_cache=True,
            )
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        return text or source_text, True, None
    except Exception as exc:
        return source_text, False, str(exc)

def _reset_gloss_transcript() -> None:
    global GLOSS_LAST_TOKEN, GLOSS_LAST_SUBMITTED_SOURCE
    with GLOSS_TRANSCRIPT_LOCK:
        GLOSS_RAW_LABELS.clear()
        GLOSS_DRAFT_LABELS.clear()
        GLOSS_FINAL_SEGMENTS.clear()
        GLOSS_LAST_TOKEN = None
        GLOSS_LAST_SUBMITTED_SOURCE = ""

def _join_labels(labels: list[str]) -> str:
    return _normalize_label(" ".join(labels))

def _final_text() -> str:
    return _normalize_exact_label(" ".join(segment for segment in GLOSS_FINAL_SEGMENTS if segment))

def _build_transcript_snapshot(
    *,
    flushed_segments: list[dict] | None = None,
    skipped: dict | None = None,
) -> dict:
    with GLOSS_TRANSCRIPT_LOCK:
        raw_gestures = list(GLOSS_RAW_LABELS)
        draft_gestures = list(GLOSS_DRAFT_LABELS)
        final_text = _final_text()
    return {
        "raw_gestures": raw_gestures,
        "raw_text": _join_labels(raw_gestures),
        "draft_gestures": draft_gestures,
        "draft_text": _join_labels(draft_gestures),
        "final_text": final_text,
        "flushed_segments": flushed_segments or [],
        "skipped": skipped,
    }

def _can_flush_labels(labels: list[str]) -> bool:
    if not labels:
        return False
    is_standalone_final = len(labels) == 1 and labels[0] in STANDALONE_FINAL_LABELS
    return len(labels) >= MIN_SEGMENT_LEN or is_standalone_final

def _should_drop_before_new_token(label: str) -> bool:
    return bool(GLOSS_DRAFT_LABELS) and len(GLOSS_DRAFT_LABELS) < MIN_SEGMENT_LEN and label in NEW_SEGMENT_MARKERS

def _should_flush_before_new_token(label: str, timestamp: float) -> tuple[bool, str]:
    if not GLOSS_DRAFT_LABELS or GLOSS_LAST_TOKEN is None:
        return False, ""
    last_timestamp = float(GLOSS_LAST_TOKEN.get("timestamp") or timestamp)
    if ENABLE_PAUSE_SEGMENTATION:
        if len(GLOSS_DRAFT_LABELS) >= MIN_SEGMENT_LEN and timestamp - last_timestamp > FORCE_FLUSH_TIMEOUT:
            return True, "force_timeout"
        if len(GLOSS_DRAFT_LABELS) >= MIN_SEGMENT_LEN and timestamp - last_timestamp > PAUSE_THRESHOLD:
            return True, "pause"
    if len(GLOSS_DRAFT_LABELS) >= MIN_SEGMENT_LEN and label in NEW_SEGMENT_MARKERS:
        return True, "new_marker"
    if len(GLOSS_DRAFT_LABELS) >= MIN_SEGMENT_LEN and label in SUBJECT_MARKERS:
        return True, "new_subject"
    return False, ""

def _should_flush_after_add() -> tuple[bool, str]:
    if len(GLOSS_DRAFT_LABELS) >= MAX_SEGMENT_LEN:
        return True, "max_len"
    labels = list(GLOSS_DRAFT_LABELS)
    if labels and labels[-1] in STANDALONE_FINAL_LABELS:
        if len(labels) == 1 or len(labels) >= MIN_SEGMENT_LEN:
            return True, "final_marker"
    if len(labels) >= 2 and tuple(labels[-2:]) in FINAL_PHRASES:
        return True, "final_phrase"
    return False, ""

def _flush_gloss_segment(reason: str) -> dict | None:
    global GLOSS_LAST_SUBMITTED_SOURCE
    with GLOSS_TRANSCRIPT_LOCK:
        labels = list(GLOSS_DRAFT_LABELS)
        if not _can_flush_labels(labels):
            return None
        source_text = _join_labels(labels)
        GLOSS_DRAFT_LABELS.clear()
        if source_text == GLOSS_LAST_SUBMITTED_SOURCE:
            return None
        GLOSS_LAST_SUBMITTED_SOURCE = source_text
    target_text, model_used, error = _generate_gloss_text(source_text)
    with GLOSS_TRANSCRIPT_LOCK:
        if target_text:
            GLOSS_FINAL_SEGMENTS.append(target_text)
    return {
        "source": source_text,
        "target": target_text,
        "reason": reason,
        "model_used": model_used,
        "error": error,
    }

def _append_gloss_and_build_transcript(
    label: str | None = None,
    confidence: float | None = None,
    timestamp: float | None = None,
) -> dict:
    global GLOSS_LAST_TOKEN
    if label is None or not _is_real_gesture_label(label):
        if not ENABLE_PAUSE_SEGMENTATION:
            return _build_transcript_snapshot()
        segment = _flush_gloss_segment("pause")
        flushed = [segment] if segment is not None else []
        return _build_transcript_snapshot(flushed_segments=flushed)
    is_dactyl = _is_dactyl_label(label)
    clean_label = (
        _normalize_gloss_exact_label(label).upper()
        if is_dactyl
        else _normalize_gloss_label(label)
    )
    confidence_value = float(confidence if confidence is not None else 1.0)
    timestamp_value = float(timestamp if timestamp is not None else time.time())
    if confidence_value < MIN_GESTURE_CONFIDENCE:
        return _build_transcript_snapshot(
            skipped={
                "label": clean_label,
                "reason": "low_confidence",
                "confidence": confidence_value,
            }
        )
    flushed_segments: list[dict] = []
    if is_dactyl:
        segment = _flush_gloss_segment("before_dactyl")
        if segment is not None:
            flushed_segments.append(segment)
    with GLOSS_TRANSCRIPT_LOCK:
        if not is_dactyl and _should_drop_before_new_token(clean_label):
            GLOSS_DRAFT_LABELS.clear()
        should_flush, flush_reason = (
            (False, "")
            if is_dactyl
            else _should_flush_before_new_token(clean_label, timestamp_value)
        )
    if should_flush:
        segment = _flush_gloss_segment(flush_reason)
        if segment is not None:
            flushed_segments.append(segment)
    with GLOSS_TRANSCRIPT_LOCK:
        previous = GLOSS_LAST_TOKEN
        is_duplicate = (
            previous is not None
            and str(previous.get("label") or "") == clean_label
            and timestamp_value - float(previous.get("timestamp") or timestamp_value) < REPEAT_COOLDOWN
        )
        if is_duplicate:
            skipped = {
                "label": clean_label,
                "reason": "duplicate",
                "confidence": confidence_value,
            }
        else:
            skipped = None
            GLOSS_RAW_LABELS.append(clean_label)
            if is_dactyl:
                GLOSS_FINAL_SEGMENTS.append(clean_label)
            else:
                GLOSS_DRAFT_LABELS.append(clean_label)
            GLOSS_LAST_TOKEN = {
                "label": clean_label,
                "confidence": confidence_value,
                "timestamp": timestamp_value,
                "is_dactyl": is_dactyl,
            }
            should_flush_after, after_reason = (
                (False, "")
                if is_dactyl
                else _should_flush_after_add()
            )
    if skipped is not None:
        return _build_transcript_snapshot(
            flushed_segments=flushed_segments,
            skipped=skipped,
        )
    if should_flush_after:
        segment = _flush_gloss_segment(after_reason)
        if segment is not None:
            flushed_segments.append(segment)
    return _build_transcript_snapshot(flushed_segments=flushed_segments)

def _get_morph_analyzer():
    global MORPH_ANALYZER
    if MORPH_ANALYZER is not None:
        return MORPH_ANALYZER
    try:
        import pymorphy3
        MORPH_ANALYZER = pymorphy3.MorphAnalyzer()
    except Exception:
        MORPH_ANALYZER = False
    return MORPH_ANALYZER

def _lemmatize_word_fallback(word: str) -> str:
    lower = word.lower()
    if not lower or not re.fullmatch(r"[а-яё-]+", lower):
        return lower
    endings = (
        "иями", "ями", "ами", "иями", "его", "ого", "ему", "ому", "ыми", "ими",
        "ейся", "ийся", "ться", "ешь", "ишь", "ете", "ите", "ала", "или", "ыми",
        "ыми", "ого", "ему", "ому", "иях", "иях", "иях", "ях", "ах", "иям",
        "ям", "ием", "ем", "ом", "ой", "ей", "ий", "ый", "ая", "яя", "ое", "ее",
        "ов", "ев", "ие", "ые", "ий", "ый", "ую", "юю", "ам", "ям", "ах", "ях",
        "ы", "и", "а", "я", "е", "у", "ю", "о",
    )
    for ending in endings:
        if lower.endswith(ending) and len(lower) - len(ending) >= 3:
            stem = lower[: -len(ending)]
            if ending in {"ы", "и", "а", "я"}:
                return stem
            return stem
    return lower

def _lemmatize_word(word: str) -> str:
    lower = word.lower()
    morph = _get_morph_analyzer()
    if morph:
        try:
            parsed = morph.parse(lower)
            if parsed:
                return str(parsed[0].normal_form)
        except Exception:
            pass
    return _lemmatize_word_fallback(lower)

def _lemmatize_text(text: str) -> str:
    tokens = re.findall(r"[0-9A-Za-zА-Яа-яЁё-]+", text)
    if not tokens:
        return _normalize_label(text)
    lemmatized = []
    for token in tokens:
        if re.fullmatch(r"[0-9A-Za-z-]+", token):
            lemmatized.append(token.lower())
        else:
            lemmatized.append(_lemmatize_word(token))
    return _normalize_label(" ".join(lemmatized))

def _get_primary_parse(word: str):
    morph = _get_morph_analyzer()
    if not morph:
        return None
    try:
        parsed = morph.parse(word)
        return parsed[0] if parsed else None
    except Exception:
        return None

def _normalize_verb_search_form(lemma: str) -> str:
    stem = lemma
    for suffix in ("ся", "сь"):
        if stem.endswith(suffix) and len(stem) - len(suffix) >= 4:
            stem = stem[: -len(suffix)]
            break
    for suffix in ("ывать", "ивать", "ять", "ать", "еть", "ить", "уть", "ыть", "ти", "чь"):
        if stem.endswith(suffix) and len(stem) - len(suffix) >= 4:
            stem = stem[: -len(suffix)]
            break
    if stem.endswith("л") and len(stem) >= 5:
        stem = stem[:-1]
    return stem

def _normalize_search_token(token: str) -> str:
    lower = token.lower()
    if re.fullmatch(r"[0-9A-Za-z-]+", lower):
        return lower
    lemma = _lemmatize_word(token)
    parse = _get_primary_parse(lower)
    pos = getattr(getattr(parse, "tag", None), "POS", None)
    if pos in {"VERB", "INFN"}:
        return _normalize_verb_search_form(lemma)
    return lemma

def _normalize_search_text(text: str) -> str:
    tokens = re.findall(r"[0-9A-Za-zА-Яа-яЁё-]+", text)
    if not tokens:
        return _normalize_label(text)
    return _normalize_label(" ".join(_normalize_search_token(token) for token in tokens))

def _is_single_letter_label(text: str) -> bool:
    return len(text) == 1 and text.isalpha()

def _choose_better_clip(current: dict | None, candidate: dict) -> dict:
    if current is None:
        return candidate
    preferred = PREFERRED_VIDEO_RESOLUTION
    if preferred is not None:
        current_resolution = (int(current.get("width") or 0), int(current.get("height") or 0))
        candidate_resolution = (int(candidate.get("width") or 0), int(candidate.get("height") or 0))
        current_is_preferred = current_resolution == preferred
        candidate_is_preferred = candidate_resolution == preferred
        if candidate_is_preferred and not current_is_preferred:
            return candidate
        if current_is_preferred and not candidate_is_preferred:
            return current
    current_length = float(current.get("length") or 1e9)
    candidate_length = float(candidate.get("length") or 1e9)
    return candidate if candidate_length < current_length else current

def _clip_sort_key(clip: dict) -> tuple[int, float, str]:
    preferred = PREFERRED_VIDEO_RESOLUTION
    clip_resolution = (int(clip.get("width") or 0), int(clip.get("height") or 0))
    is_preferred = 1 if (preferred is not None and clip_resolution == preferred) else 0
    clip_length = float(clip.get("length") or 1e9)
    attachment_id = str(clip.get("attachment_id") or "")
    return (-is_preferred, clip_length, attachment_id)

def _register_variant(target: dict[str, list[dict]], key: str, clip: dict) -> None:
    target.setdefault(key, []).append(clip)

def _finalize_variant_map(raw_map: dict[str, list[dict]]) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    variants: dict[str, list[dict]] = {}
    best_index: dict[str, dict] = {}
    for key, clips in raw_map.items():
        ordered = sorted(clips, key=_clip_sort_key)
        variants[key] = ordered
        if ordered:
            best_index[key] = ordered[0]
    return variants, best_index

def _pick_variant(variants_map: dict[str, list[dict]], key: str, variant_offset: int) -> dict | None:
    variants = variants_map.get(key)
    if not variants:
        return None
    index = max(0, int(variant_offset)) % len(variants)
    return variants[index]

def _build_video_indexes() -> tuple[
    dict[str, list[dict]],
    dict[str, dict],
    dict[str, list[dict]],
    dict[str, dict],
    dict[str, list[dict]],
    dict[str, dict],
]:
    if not ANNOTATIONS_PATH.exists():
        raise FileNotFoundError(f"annotations.csv not found: {ANNOTATIONS_PATH}")
    if not DATASET_VIDEO_DIR.exists():
        raise FileNotFoundError(f"Dataset video dir not found: {DATASET_VIDEO_DIR}")
    global PREFERRED_VIDEO_RESOLUTION, MAX_GESTURE_WORDS
    by_exact_text_raw: dict[str, list[dict]] = {}
    by_text_raw: dict[str, list[dict]] = {}
    by_letter_raw: dict[str, list[dict]] = {}
    rows: list[dict[str, str]] = []
    resolution_counts: dict[tuple[int, int], int] = {}
    with ANNOTATIONS_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows.append(row)
            try:
                width_value = int(float(row.get("width") or 0))
                height_value = int(float(row.get("height") or 0))
            except Exception:
                continue
            if width_value > 0 and height_value > 0:
                key = (width_value, height_value)
                resolution_counts[key] = resolution_counts.get(key, 0) + 1
    if resolution_counts:
        PREFERRED_VIDEO_RESOLUTION = max(resolution_counts.items(), key=lambda item: item[1])[0]
    else:
        PREFERRED_VIDEO_RESOLUTION = None
    for row in rows:
            raw_text = str(row.get("text") or "").strip()
            if not raw_text:
                continue
            text = _decode_text_maybe(raw_text)
            if text == "no_event":
                continue
            attachment_id = str(row.get("attachment_id") or "").strip()
            if not attachment_id:
                continue
            video_path = DATASET_VIDEO_DIR / f"{attachment_id}.mp4"
            if not video_path.exists():
                continue
            try:
                length_value = float(row.get("length") or 0.0)
            except Exception:
                length_value = 0.0
            try:
                width_value = int(float(row.get("width") or 0))
                height_value = int(float(row.get("height") or 0))
            except Exception:
                width_value = 0
                height_value = 0
            clip = {
                "label": text,
                "attachment_id": attachment_id,
                "length": length_value,
                "width": width_value,
                "height": height_value,
                "video_url": f"/gesture-media/{attachment_id}.mp4",
            }
            exact_key = _normalize_exact_label(text)
            _register_variant(by_exact_text_raw, exact_key, clip)
            normalized = _normalize_label(text)
            upper_text = text.upper()
            is_single_letter = _is_single_letter_label(upper_text)
            is_dactyl_letter = is_single_letter and text == upper_text
            if not is_dactyl_letter:
                _register_variant(by_text_raw, normalized, clip)
                lemmatized = _lemmatize_text(text)
                _register_variant(by_text_raw, lemmatized, clip)
                search_normalized = _normalize_search_text(text)
                _register_variant(by_text_raw, search_normalized, clip)
            if is_dactyl_letter:
                _register_variant(by_letter_raw, upper_text, clip)
    exact_variants, exact_best = _finalize_variant_map(by_exact_text_raw)
    text_variants, text_best = _finalize_variant_map(by_text_raw)
    letter_variants, letter_best = _finalize_variant_map(by_letter_raw)
    max_words = 1
    for key in list(exact_variants.keys()) + list(text_variants.keys()):
        max_words = max(max_words, len(str(key).split()))
    MAX_GESTURE_WORDS = max_words
    return exact_variants, exact_best, text_variants, text_best, letter_variants, letter_best

def _tokenize_input_text(text: str) -> list[str]:
    return re.findall(r"[0-9A-Za-zА-Яа-яЁё-]+", text)

def _resolve_token_to_sequence(token: str, variant_offset: int = 0, allow_dactyl: bool = True) -> dict | None:
    exact_token = _normalize_exact_label(token)
    normalized = _normalize_label(token)
    is_single_letter_token = len(exact_token) == 1 and exact_token.isalpha()
    direct_clip = None
    lookup_form = normalized
    match_kind = "normalized"
    if not is_single_letter_token:
        direct_clip = _pick_variant(GESTURE_VIDEO_EXACT_VARIANTS, exact_token, variant_offset)
        lookup_form = exact_token
        match_kind = "exact"
    if direct_clip is None:
        direct_clip = _pick_variant(GESTURE_VIDEO_VARIANTS, normalized, variant_offset)
        lookup_form = normalized
        match_kind = "normalized"
    if direct_clip is None:
        lemmatized = _lemmatize_text(token)
        direct_clip = _pick_variant(GESTURE_VIDEO_VARIANTS, lemmatized, variant_offset)
        lookup_form = lemmatized
        match_kind = "lemma"
    if direct_clip is None:
        search_normalized = _normalize_search_text(token)
        direct_clip = _pick_variant(GESTURE_VIDEO_VARIANTS, search_normalized, variant_offset)
        lookup_form = search_normalized
        match_kind = "search"
    if direct_clip is not None:
        return {
            "token": token,
            "normalized": normalized,
            "lookup_form": lookup_form,
            "mode": "word",
            "found_direct": True,
            "match_kind": match_kind,
            "sequence": [direct_clip],
            "variant_offset": int(max(0, variant_offset)),
            "missing_letters": [],
        }
    if not allow_dactyl:
        return None
    sequence = []
    missing_letters = []
    for char in token.upper():
        if not char.isalpha():
            continue
        clip = _pick_variant(DACTYL_VIDEO_VARIANTS, char, variant_offset)
        if clip is None:
            missing_letters.append(char)
            continue
        sequence.append(clip)
    return {
        "token": token,
        "normalized": normalized,
        "lookup_form": _lemmatize_text(token),
        "mode": "dactyl",
        "found_direct": False,
        "match_kind": "dactyl",
        "sequence": sequence,
        "variant_offset": int(max(0, variant_offset)),
        "missing_letters": missing_letters,
    }

def _resolve_text_to_items(text: str, variant_offset: int) -> list[dict]:
    tokens = _tokenize_input_text(text)
    items: list[dict] = []
    index = 0
    while index < len(tokens):
        matched_item = None
        max_span = min(MAX_GESTURE_WORDS, len(tokens) - index)
        for span in range(max_span, 1, -1):
            phrase = " ".join(tokens[index : index + span])
            candidate = _resolve_token_to_sequence(
                phrase,
                variant_offset=variant_offset,
                allow_dactyl=False,
            )
            if candidate is not None:
                matched_item = candidate
                index += span
                break
        if matched_item is None:
            single_token = tokens[index]
            matched_item = _resolve_token_to_sequence(
                single_token,
                variant_offset=variant_offset,
                allow_dactyl=True,
            )
            index += 1
        if matched_item is not None:
            items.append(matched_item)
    return items

def _load_labels(labels_path: Path) -> dict[int, str]:
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.json not found: {labels_path}")
    data = json.loads(labels_path.read_text(encoding="utf-8"))
    if isinstance(data["id2label"], list):
        return {i: str(v) for i, v in enumerate(data["id2label"])}
    return {int(k): str(v) for k, v in data["id2label"].items()}

def _make_model(num_classes: int, architecture: str) -> torch.nn.Module:
    if architecture == "effnet_b0":
        return SlovoEfficientNetB0(num_classes=num_classes, drop=0.3)
    if architecture == "effnetv2_s":
        return SlovoEfficientNetV2S(num_classes=num_classes, drop=0.3)
    raise ValueError(f"Unknown model architecture: {architecture}")

def _load_model(num_classes: int, ckpt_path: Path, architecture: str) -> torch.nn.Module:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model = _make_model(num_classes=num_classes, architecture=architecture)
    try:
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()
    return model

def _preprocess_xyz(xyz: np.ndarray) -> np.ndarray:
    xyz = np.nan_to_num(xyz, nan=NAN_TO_NUM).astype(np.float32)
    if PER_SAMPLE_NORMALIZE:
        mean = float(xyz.mean())
        std = max(float(xyz.std()), 1e-6)
        xyz = (xyz - mean) / std
    return np.transpose(xyz, (2, 0, 1)).astype(np.float32)

def _resample_time_nearest(x: np.ndarray, target_t: int) -> np.ndarray:
    t0 = x.shape[0]
    if t0 == target_t:
        return x
    if t0 <= 0:
        raise ValueError("Empty sequence")
    idx = np.linspace(0, t0 - 1, target_t)
    idx = np.rint(idx).astype(np.int64)
    idx = np.clip(idx, 0, t0 - 1)
    return x[idx]

def _infer_from_xyz(xyz: np.ndarray, mode: str) -> tuple[int, str, float]:
    model = MODELS.get(mode)
    id2label = MODE_LABELS.get(mode)
    if model is None or id2label is None:
        raise RuntimeError(f"Model is not initialized for mode='{mode}'")
    x = _preprocess_xyz(xyz)
    x_t = torch.from_numpy(x).unsqueeze(0).to(DEVICE)
    if EFFNET_INPUT_SIZE:
        x_t = F.interpolate(
            x_t,
            size=(EFFNET_INPUT_SIZE, EFFNET_INPUT_SIZE),
            mode="bilinear",
            align_corners=False,
        )
    with torch.no_grad():
        logits = model(x_t)
        probs = torch.softmax(logits, dim=1)[0]
        pred_id = int(torch.argmax(probs).item())
        conf = float(probs[pred_id].item())
    label = id2label.get(pred_id, str(pred_id))
    return pred_id, label, conf
@app.on_event("startup")

def startup_event() -> None:
    global MODELS, MODE_LABELS
    global GESTURE_VIDEO_EXACT_VARIANTS, GESTURE_VIDEO_EXACT_INDEX
    global GESTURE_VIDEO_VARIANTS, GESTURE_VIDEO_INDEX
    global DACTYL_VIDEO_VARIANTS, DACTYL_VIDEO_INDEX
    loaded_models: dict[str, torch.nn.Module] = {}
    loaded_labels: dict[str, dict[int, str]] = {}
    for mode_name, cfg in MODE_CONFIGS.items():
        id2label = _load_labels(cfg.labels_path)
        model = _load_model(
            num_classes=len(id2label),
            ckpt_path=cfg.ckpt_path,
            architecture=cfg.architecture,
        )
        loaded_models[mode_name] = model
        loaded_labels[mode_name] = id2label
    MODELS = loaded_models
    MODE_LABELS = loaded_labels
    (
        GESTURE_VIDEO_EXACT_VARIANTS,
        GESTURE_VIDEO_EXACT_INDEX,
        GESTURE_VIDEO_VARIANTS,
        GESTURE_VIDEO_INDEX,
        DACTYL_VIDEO_VARIANTS,
        DACTYL_VIDEO_INDEX,
    ) = _build_video_indexes()
    _load_gloss_text_model()
@app.on_event("shutdown")

def shutdown_event() -> None:
    return None
@app.get("/", response_class=HTMLResponse)

def root() -> HTMLResponse:
    index_path = FRONTEND_DIR / "index.html"
    return HTMLResponse(index_path.read_text(encoding="utf-8"))
@app.get("/translate", response_class=HTMLResponse)

def translate_page() -> HTMLResponse:
    translate_path = FRONTEND_DIR / "translate.html"
    return HTMLResponse(translate_path.read_text(encoding="utf-8"))
@app.get("/api/modes")

def get_modes() -> dict:
    return {
        "ok": True,
        "default_mode": MODE_HOLISTIC,
        "modes": [
            {"id": MODE_HANDS, "title": "Только руки", "points": MODE_CONFIGS[MODE_HANDS].p},
            {"id": MODE_HOLISTIC, "title": "Руки + поза", "points": MODE_CONFIGS[MODE_HOLISTIC].p},
        ],
    }
@app.get("/api/gloss-to-text/status")

def gloss_to_text_status() -> dict:
    loaded = _load_gloss_text_model()
    return {
        "ok": loaded,
        "model_dir": str(GLOSS_TEXT_MODEL_DIR),
        "device": str(DEVICE),
        "error": GLOSS_TEXT_LOAD_ERROR,
    }
@app.post("/api/text-to-gesture")

def text_to_gesture(payload: TextToGestureRequest) -> dict:
    text = str(payload.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is empty")
    raw_tokens = _tokenize_input_text(text)
    if not raw_tokens:
        raise HTTPException(status_code=400, detail="No valid words found in text")
    variant_offset = max(0, int(payload.variant_offset or 0))
    items = _resolve_text_to_items(text, variant_offset=variant_offset)
    return {
        "ok": True,
        "text": text,
        "variant_offset": variant_offset,
        "tokens": items,
        "stats": {
            "tokens_total": len(items),
            "raw_tokens_total": len(raw_tokens),
            "direct_matches": sum(1 for item in items if item["found_direct"]),
            "dactyl_fallbacks": sum(1 for item in items if not item["found_direct"]),
        },
    }
@app.post("/api/text-to-gesture/token")

def text_to_gesture_token(payload: TokenVariantRequest) -> dict:
    token = str(payload.token or "").strip()
    if not token:
        raise HTTPException(status_code=400, detail="Token is empty")
    variant_offset = max(0, int(payload.variant_offset or 0))
    item = _resolve_token_to_sequence(token, variant_offset=variant_offset)
    return {
        "ok": True,
        "token": token,
        "variant_offset": variant_offset,
        "item": item,
    }
@app.post("/api/session/start")

def session_start() -> dict:
    _reset_gloss_transcript()
    return {"ok": True, "transcript": _append_gloss_and_build_transcript()}
@app.post("/api/session/clear")

def session_clear() -> dict:
    _reset_gloss_transcript()
    return {"ok": True, "transcript": _append_gloss_and_build_transcript()}
@app.post("/api/session/flush")

def session_flush() -> dict:
    segment = _flush_gloss_segment("manual")
    flushed = [segment] if segment is not None else []
    return {"ok": True, "transcript": _build_transcript_snapshot(flushed_segments=flushed)}
@app.post("/api/session/finish")

def session_finish(payload: KeypointsBatch) -> dict:
    frames_collected = len(payload.frames)
    if frames_collected == 0:
        raise HTTPException(status_code=400, detail="No frames collected")
    mode = str(payload.mode).strip().lower()
    cfg = MODE_CONFIGS.get(mode)
    if cfg is None:
        raise HTTPException(status_code=400, detail=f"Unknown mode: {mode}")
    xyz_stack = np.asarray(payload.frames, dtype=np.float32)
    if xyz_stack.ndim != 3 or xyz_stack.shape[1:] != (cfg.p, 3):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid shape, expected (T0,{cfg.p},3), got {xyz_stack.shape}",
        )
    xyz_rs = _resample_time_nearest(xyz_stack, cfg.t)
    with INFER_LOCK:
        pred_id, label, conf = _infer_from_xyz(xyz_rs, mode=mode)
    transcript = _append_gloss_and_build_transcript(
        label=label,
        confidence=conf,
        timestamp=time.time(),
    )
    fps = None
    if payload.duration_sec and payload.duration_sec > 0:
        fps = frames_collected / payload.duration_sec
    return {
        "ok": True,
        "mode": mode,
        "frames_collected": frames_collected,
        "resampled_to": cfg.t,
        "resample_method": "nearest",
        "fps": fps,
        "prediction": {
            "id": pred_id,
            "label": label,
            "confidence": conf,
        },
        "transcript": transcript,
    }
