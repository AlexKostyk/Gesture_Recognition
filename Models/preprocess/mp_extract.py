from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2
@dataclass(frozen=True)

class MPOptions:
    model_complexity: int = 1
    min_detection_conf: float = 0.5
    min_tracking_conf: float = 0.5

class MediaPipeHolisticExtractor:
    def __init__(self, use_hands: bool = True, use_pose: bool = True, opts: MPOptions = MPOptions()):
        self.use_hands = use_hands
        self.use_pose = use_pose
        self.opts = opts
        import mediapipe as mp
        self.mp = mp
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=opts.model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=opts.min_detection_conf,
            min_tracking_confidence=opts.min_tracking_conf,
        )
        self.n_left = 21 if use_hands else 0
        self.n_right = 21 if use_hands else 0
        self.n_pose = 33 if use_pose else 0
        self.P = self.n_left + self.n_right + self.n_pose
    def close(self):
        self.holistic.close()
    def _lm_to_array(self, lm_list, n: int):
        if lm_list is None:
            xyz = np.full((n, 3), np.nan, dtype=np.float32)
            mask = np.zeros((n,), dtype=np.uint8)
            return xyz, mask
        xyz = np.zeros((n, 3), dtype=np.float32)
        mask = np.ones((n,), dtype=np.uint8)
        for i, lm in enumerate(lm_list.landmark[:n]):
            xyz[i, 0] = lm.x
            xyz[i, 1] = lm.y
            xyz[i, 2] = lm.z
        return xyz, mask
    def extract_from_bgr_frames(self, frames_bgr: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        T = len(frames_bgr)
        xyz_seq = np.full((T, self.P, 3), np.nan, dtype=np.float32)
        mask_seq = np.zeros((T, self.P), dtype=np.uint8)
        for t, frame_bgr in enumerate(frames_bgr):
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = self.holistic.process(frame_rgb)
            offset = 0
            if self.use_hands:
                xyz_l, m_l = self._lm_to_array(res.left_hand_landmarks, 21)
                xyz_r, m_r = self._lm_to_array(res.right_hand_landmarks, 21)
                xyz_seq[t, offset:offset + 21] = xyz_l
                mask_seq[t, offset:offset + 21] = m_l
                offset += 21
                xyz_seq[t, offset:offset + 21] = xyz_r
                mask_seq[t, offset:offset + 21] = m_r
                offset += 21
            if self.use_pose:
                xyz_p, m_p = self._lm_to_array(res.pose_landmarks, 33)
                xyz_seq[t, offset:offset + 33] = xyz_p
                mask_seq[t, offset:offset + 33] = m_p
                offset += 33
        return xyz_seq, mask_seq
