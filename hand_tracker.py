"""Hand tracking module using OpenCV and MediaPipe Hands.

Manages camera lifecycle with V4L2/MJPG HD 720p negotiation, mirror flipping,
EMA landmark smoothing, coordinate scaling, palm scale calculation, and drawing.
"""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from config import AppConfig
from utils import LandmarkIndex, Point3D, calculate_palm_scale

logger = logging.getLogger(__name__)


@dataclass
class HandData:
    """Structured container for hand tracking data in a single frame."""
    detected: bool = False
    # 21 landmarks in normalized coordinates [0.0 - 1.0]
    landmarks_norm: List[Point3D] = field(default_factory=list)
    # 21 landmarks in pixel coordinates (x, y)
    landmarks_px: List[Tuple[int, int]] = field(default_factory=list)
    # Estimated palm center (average of wrist and MCPs)
    palm_center_norm: Tuple[float, float] = (0.5, 0.5)
    palm_center_px: Tuple[int, int] = (0, 0)
    # Normalized palm scale (invariant to camera distance)
    palm_scale: float = 0.20
    # Handedness label ("Left" or "Right")
    handedness: str = "Unknown"
    # Tracking confidence score if available
    confidence: float = 0.0


class HandTracker:
    """Encapsulates OpenCV camera capture and MediaPipe Hands processing."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_width: int = 1280
        self.frame_height: int = 720

        # Landmark smoothing history
        self._prev_landmarks_norm: Optional[List[Point3D]] = None

        # MediaPipe initialization
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.config.tracking.max_num_hands,
            min_detection_confidence=self.config.tracking.min_detection_confidence,
            min_tracking_confidence=self.config.tracking.min_tracking_confidence,
            model_complexity=self.config.tracking.model_complexity,
        )

        self._init_camera()

    def _init_camera(self) -> None:
        """Initialize webcam using V4L2 and MJPG for crisp 720p @ 30fps."""
        cam_idx = self.config.camera.camera_index
        logger.info("Initializing webcam at index %d...", cam_idx)

        # On Linux, CAP_V4L2 unlocks hardware MJPG 720p streams
        is_linux = platform.system() == "Linux"
        if is_linux:
            self.cap = cv2.VideoCapture(cam_idx, cv2.CAP_V4L2)
        else:
            self.cap = cv2.VideoCapture(cam_idx)

        if not self.cap or not self.cap.isOpened():
            # Fallback to default backend
            self.cap = cv2.VideoCapture(cam_idx)

        if not self.cap or not self.cap.isOpened():
            logger.error("Webcam device index %d could not be opened.", cam_idx)
            raise RuntimeError(
                f"Unable to access webcam at index {cam_idx}. Please verify camera connection and permissions."
            )

        # Set MJPG FourCC to unlock high resolution 30fps stream
        fourcc_code = self.config.camera.fourcc
        if fourcc_code and len(fourcc_code) == 4:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc_code))

        # Request target resolution (1280x720)
        t_w = self.config.camera.target_width
        t_h = self.config.camera.target_height
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, t_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, t_h)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.camera.fps)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        if actual_w <= 0 or actual_h <= 0:
            fb_w = self.config.camera.fallback_width
            fb_h = self.config.camera.fallback_height
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, fb_w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, fb_h)
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.frame_width = max(320, actual_w)
        self.frame_height = max(240, actual_h)
        logger.info("Camera online: %dx%d @ %d FPS (Format: %s)",
                    self.frame_width, self.frame_height, actual_fps, fourcc_code)

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from webcam and apply horizontal flip."""
        if not self.cap or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None

        # Mirror frame horizontally so controls feel natural
        if self.config.camera.mirror_preview:
            frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        self.frame_width, self.frame_height = w, h
        return True, frame

    def process(self, frame: np.ndarray) -> HandData:
        """Process a BGR video frame through MediaPipe and apply EMA landmark smoothing."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        hand_data = HandData(detected=False)

        if not results.multi_hand_landmarks:
            self._prev_landmarks_norm = None
            return hand_data

        landmarks_proto = results.multi_hand_landmarks[0]
        hand_data.detected = True

        if results.multi_handedness:
            handedness_info = results.multi_handedness[0].classification[0]
            raw_label = handedness_info.label
            hand_data.handedness = "Right" if raw_label == "Left" else "Left"
            hand_data.confidence = float(handedness_info.score)

        h, w = self.frame_height, self.frame_width
        raw_norm: List[Point3D] = [Point3D.from_landmark(lm) for lm in landmarks_proto.landmark]

        # Landmark position temporal smoothing (filters out high-frequency hand jitter)
        alpha = self.config.tracking.smoothing_factor
        if self._prev_landmarks_norm and len(self._prev_landmarks_norm) == len(raw_norm):
            smoothed_norm: List[Point3D] = []
            for prev_pt, curr_pt in zip(self._prev_landmarks_norm, raw_norm):
                sm_x = alpha * prev_pt.x + (1.0 - alpha) * curr_pt.x
                sm_y = alpha * prev_pt.y + (1.0 - alpha) * curr_pt.y
                sm_z = alpha * prev_pt.z + (1.0 - alpha) * curr_pt.z
                smoothed_norm.append(Point3D(sm_x, sm_y, sm_z))
            final_norm = smoothed_norm
        else:
            final_norm = raw_norm

        self._prev_landmarks_norm = final_norm

        landmarks_px: List[Tuple[int, int]] = [
            (int(np.clip(pt.x * w, 0, w - 1)), int(np.clip(pt.y * h, 0, h - 1)))
            for pt in final_norm
        ]

        hand_data.landmarks_norm = final_norm
        hand_data.landmarks_px = landmarks_px
        hand_data.palm_scale = calculate_palm_scale(final_norm)

        # Compute palm center
        center_indices = [
            LandmarkIndex.WRIST,
            LandmarkIndex.INDEX_MCP,
            LandmarkIndex.MIDDLE_MCP,
            LandmarkIndex.RING_MCP,
            LandmarkIndex.PINKY_MCP,
        ]
        avg_x = sum(final_norm[i].x for i in center_indices) / len(center_indices)
        avg_y = sum(final_norm[i].y for i in center_indices) / len(center_indices)
        hand_data.palm_center_norm = (avg_x, avg_y)
        hand_data.palm_center_px = (int(avg_x * w), int(avg_y * h))

        return hand_data

    def draw_landmarks(
        self,
        frame: np.ndarray,
        hand_data: HandData,
        highlight_pinch: bool = False,
        highlight_fist: bool = False,
    ) -> None:
        """Draw sleek, elegant minimalist hand skeleton directly onto the frame."""
        if not hand_data.detected or not hand_data.landmarks_px:
            return

        connections = self.mp_hands.HAND_CONNECTIONS
        lms = hand_data.landmarks_px

        bone_color = (60, 60, 255) if highlight_fist else (255, 200, 0)
        tip_color = (0, 255, 128) if not highlight_pinch else (0, 0, 255)

        # Draw bones
        for start_idx, end_idx in connections:
            if start_idx < len(lms) and end_idx < len(lms):
                cv2.line(frame, lms[start_idx], lms[end_idx], bone_color, 2, cv2.LINE_AA)

        # Draw joints
        tips = {
            LandmarkIndex.THUMB_TIP,
            LandmarkIndex.INDEX_TIP,
            LandmarkIndex.MIDDLE_TIP,
            LandmarkIndex.RING_TIP,
            LandmarkIndex.PINKY_TIP,
        }

        for idx, pt in enumerate(lms):
            r = 5 if idx in tips else 3
            c = tip_color if idx in tips else (220, 220, 220)
            cv2.circle(frame, pt, r, c, -1, cv2.LINE_AA)

        # Draw line between thumb and index tip
        if len(lms) > LandmarkIndex.INDEX_TIP:
            thumb_tip = lms[LandmarkIndex.THUMB_TIP]
            index_tip = lms[LandmarkIndex.INDEX_TIP]
            p_color = (0, 0, 255) if highlight_pinch else (180, 180, 180)
            cv2.line(frame, thumb_tip, index_tip, p_color, 2 if highlight_pinch else 1, cv2.LINE_AA)
            if highlight_pinch:
                mid_pt = ((thumb_tip[0] + index_tip[0]) // 2, (thumb_tip[1] + index_tip[1]) // 2)
                cv2.circle(frame, mid_pt, 6, (0, 0, 255), -1, cv2.LINE_AA)

    def release(self) -> None:
        """Release camera and MediaPipe resources."""
        logger.info("Releasing camera and MediaPipe resources...")
        if self.hands:
            self.hands.close()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        logger.info("HandTracker resources released cleanly.")
