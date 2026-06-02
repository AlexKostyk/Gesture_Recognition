from dataclasses import dataclass
from pathlib import Path
import os
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = Path(__file__).resolve().parents[1]
@dataclass(frozen=True)

class PreprocessConfig:
    annotations_csv: Path = PROJECT_ROOT / "Dataset" / "slovo" / "annotations.csv"
    videos_dir: Path = PROJECT_ROOT / "Dataset" / "slovo" / "data"
    out_dir: Path = MODELS_ROOT / "Model_hands" / "keypoints_out"
    T: int = 70
    use_pose: bool = False
    use_hands: bool = True
    end_is_exclusive: bool = True
    min_segment_len: int = 4
    train_frac: float = 0.7
    val_frac: float = 0.15
    test_frac: float = 0.15
    split_seed: int = 56
    mp_model_complexity: int = 1
    mp_min_detection_conf: float = 0.5
    mp_min_tracking_conf: float = 0.5
    num_workers: int = max(1, (os.cpu_count() or 4) - 2)
