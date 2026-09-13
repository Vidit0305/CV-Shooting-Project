"""Gesture recognition, temporal stabilization, and mouse screen look module.

Converts 21 hand landmarks into high-level game gestures (Pinch/Shoot, Fist/Aim,
WASD Movement, Neutral) and relative mouse delta for camera aiming.
"""

from __future__ import annotations

import collections
import enum
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from config import AppConfig
from hand_tracker import HandData
from utils import (
    LandmarkIndex,
    Point3D,
    calculate_angle_3points,
    euclidean_distance_2d,
    euclidean_distance_3d,
)

logger = logging.getLogger(__name__)


class Gesture(enum.Enum):
    """Enumeration of recognizable hand gestures."""
    NO_HAND = "NO_HAND"
    NEUTRAL = "NEUTRAL"
    PINCH = "PINCH"          # LMB: Shoot
    FIST = "FIST"            # RMB: Aim / ADS
    MOVE_UP = "MOVE_UP"      # W
    MOVE_DOWN = "MOVE_DOWN"  # S
    MOVE_LEFT = "MOVE_LEFT"  # A
    MOVE_RIGHT = "MOVE_RIGHT"# D
    UNKNOWN = "UNKNOWN"


@dataclass
class FingerStates:
    """Extension state of each finger."""
    thumb: bool = False
    index: bool = False
    middle: bool = False
    ring: bool = False
    pinky: bool = False

    def count_extended(self) -> int:
        return sum([self.thumb, self.index, self.middle, self.ring, self.pinky])

    def are_four_folded(self) -> bool:
        """True if index, middle, ring, and pinky are all folded inward."""
        return not (self.index or self.middle or self.ring or self.pinky)


@dataclass
class ControlState:
    """Resolved game control state ready for OS input translation."""
    gesture: Gesture = Gesture.NO_HAND
    raw_gesture: Gesture = Gesture.NO_HAND
    shoot: bool = False                  # Left Mouse Button
    aim: bool = False                    # Right Mouse Button
    move_keys: Set[str] = field(default_factory=set) # Active movement keys (e.g. {"w"})
    # Relative mouse movement delta (in screen pixels) to rotate camera / look around
    mouse_dx: float = 0.0
    mouse_dy: float = 0.0
    pinch_ratio: float = 1.0
    finger_states: FingerStates = field(default_factory=FingerStates)
    hand_offset_x: float = 0.0          # Normalized delta from neutral center
    hand_offset_y: float = 0.0
    confidence: float = 0.0


class GestureRecognizer:
    """Performs geometric gesture analysis on raw hand landmarks."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._is_pinched: bool = False

    def detect_finger_states(self, lms: List[Point3D], palm_scale: float) -> FingerStates:
        """Detect whether each finger is extended using rotation-invariant geometry."""
        if len(lms) < 21:
            return FingerStates()

        wrist = lms[LandmarkIndex.WRIST]
        ext_ratio = self.config.gestures.finger_extension_ratio

        def is_extended(mcp_idx: int, pip_idx: int, dip_idx: int, tip_idx: int) -> bool:
            mcp = lms[mcp_idx]
            pip = lms[pip_idx]
            tip = lms[tip_idx]

            dist_wrist_tip = euclidean_distance_3d(wrist, tip)
            dist_wrist_pip = euclidean_distance_3d(wrist, pip)
            pip_angle = calculate_angle_3points(mcp, pip, tip)
            ratio = dist_wrist_tip / max(0.01, dist_wrist_pip)
            return (ratio > ext_ratio) and (pip_angle > 130.0)

        index_ext = is_extended(
            LandmarkIndex.INDEX_MCP, LandmarkIndex.INDEX_PIP, LandmarkIndex.INDEX_DIP, LandmarkIndex.INDEX_TIP
        )
        middle_ext = is_extended(
            LandmarkIndex.MIDDLE_MCP, LandmarkIndex.MIDDLE_PIP, LandmarkIndex.MIDDLE_DIP, LandmarkIndex.MIDDLE_TIP
        )
        ring_ext = is_extended(
            LandmarkIndex.RING_MCP, LandmarkIndex.RING_PIP, LandmarkIndex.RING_DIP, LandmarkIndex.RING_TIP
        )
        pinky_ext = is_extended(
            LandmarkIndex.PINKY_MCP, LandmarkIndex.PINKY_PIP, LandmarkIndex.PINKY_DIP, LandmarkIndex.PINKY_TIP
        )

        thumb_tip = lms[LandmarkIndex.THUMB_TIP]
        thumb_mcp = lms[LandmarkIndex.THUMB_MCP]
        pinky_mcp = lms[LandmarkIndex.PINKY_MCP]

        dist_thumb_pinky_mcp = euclidean_distance_3d(thumb_tip, pinky_mcp)
        dist_ip_pinky_mcp = euclidean_distance_3d(thumb_tip, pinky_mcp)
        thumb_angle = calculate_angle_3points(wrist, thumb_mcp, thumb_tip)
        thumb_ext = (dist_thumb_pinky_mcp / max(0.01, dist_ip_pinky_mcp) > self.config.gestures.thumb_extension_ratio) and (thumb_angle > 115.0)

        return FingerStates(
            thumb=thumb_ext,
            index=index_ext,
            middle=middle_ext,
            ring=ring_ext,
            pinky=pinky_ext,
        )

    def calculate_pinch_ratio(self, lms: List[Point3D], palm_scale: float) -> float:
        """Calculate normalized pinch distance between thumb tip and index tip."""
        if len(lms) < 21 or palm_scale < 1e-4:
            return 1.0

        thumb_tip = lms[LandmarkIndex.THUMB_TIP]
        index_tip = lms[LandmarkIndex.INDEX_TIP]
        dist_3d = euclidean_distance_3d(thumb_tip, index_tip)
        return dist_3d / palm_scale

    def is_fist(self, lms: List[Point3D], finger_states: FingerStates, palm_scale: float) -> bool:
        """Verify if the hand is in a closed fist pose.
        
        Uses distance from fingertips to wrist and palm center, which is
        robust to hand rotation and perspective tilt toward the camera.
        """
        if len(lms) < 21:
            return False

        wrist = lms[LandmarkIndex.WRIST]
        tips = [
            LandmarkIndex.INDEX_TIP,
            LandmarkIndex.MIDDLE_TIP,
            LandmarkIndex.RING_TIP,
            LandmarkIndex.PINKY_TIP,
        ]
        pips = [
            LandmarkIndex.INDEX_PIP,
            LandmarkIndex.MIDDLE_PIP,
            LandmarkIndex.RING_PIP,
            LandmarkIndex.PINKY_PIP,
        ]

        fold_ratios = [
            euclidean_distance_3d(wrist, lms[t]) / max(0.01, euclidean_distance_3d(wrist, lms[p]))
            for t, p in zip(tips, pips)
        ]
        avg_fold_ratio = sum(fold_ratios) / len(fold_ratios)

        # Also check tip distance to middle MCP (palm center)
        mid_mcp = lms[LandmarkIndex.MIDDLE_MCP]
        avg_dist_to_palm = sum(euclidean_distance_3d(lms[t], mid_mcp) for t in tips) / len(tips)
        normalized_dist_to_palm = avg_dist_to_palm / max(0.01, palm_scale)

        return (avg_fold_ratio < self.config.gestures.fist_fold_ratio) or (normalized_dist_to_palm < 0.70)

    def evaluate_raw(self, hand_data: HandData) -> Tuple[Gesture, float, FingerStates, Set[str], float, float]:
        """Classify single-frame raw gesture and movement keys."""
        if not hand_data.detected or len(hand_data.landmarks_norm) < 21:
            self._is_pinched = False
            return Gesture.NO_HAND, 1.0, FingerStates(), set(), 0.0, 0.0

        lms = hand_data.landmarks_norm
        scale = hand_data.palm_scale
        finger_states = self.detect_finger_states(lms, scale)
        pinch_ratio = self.calculate_pinch_ratio(lms, scale)

        # 1. Hysteresis on pinch detection
        on_thresh = self.config.calibration.calibrated_pinch_on if self.config.calibration.is_calibrated else self.config.gestures.pinch_on_ratio
        off_thresh = self.config.calibration.calibrated_pinch_off if self.config.calibration.is_calibrated else self.config.gestures.pinch_off_ratio

        if not self._is_pinched:
            if pinch_ratio < on_thresh:
                self._is_pinched = True
        else:
            if pinch_ratio > off_thresh:
                self._is_pinched = False

        # 2. Check Fist (Aim)
        is_fist_pose = self.is_fist(lms, finger_states, scale)

        # 3. Movement displacement relative to neutral anchor (only if enabled)
        move_keys: Set[str] = set()
        if self.config.gestures.enable_wasd:
            anchor_x = self.config.calibration.neutral_center_x
            anchor_y = self.config.calibration.neutral_center_y
            hand_x, hand_y = hand_data.palm_center_norm

            dx = hand_x - anchor_x
            dy = hand_y - anchor_y
            deadzone = self.config.gestures.movement_deadzone

            if dy < -deadzone:
                move_keys.add(self.config.inputs.key_forward)  # W
            elif dy > deadzone:
                move_keys.add(self.config.inputs.key_backward) # S

            if dx < -deadzone:
                move_keys.add(self.config.inputs.key_left)     # A
            elif dx > deadzone:
                move_keys.add(self.config.inputs.key_right)    # D
        else:
            dx, dy = 0.0, 0.0

        # 4. Gesture Priority Arbiter: Fist > Pinch > Movement > Neutral
        if is_fist_pose:
            raw_gesture = Gesture.FIST
        elif self._is_pinched:
            raw_gesture = Gesture.PINCH
        elif move_keys:
            abs_x = abs(dx)
            abs_y = abs(dy)
            if abs_y >= abs_x:
                raw_gesture = Gesture.MOVE_UP if dy < 0 else Gesture.MOVE_DOWN
            else:
                raw_gesture = Gesture.MOVE_LEFT if dx < 0 else Gesture.MOVE_RIGHT
        else:
            raw_gesture = Gesture.NEUTRAL

        return raw_gesture, pinch_ratio, finger_states, move_keys, dx, dy


class GestureStabilizer:
    """Temporal filter using sliding-window voting, consecutive activation, and mouse screen look tracker."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.recognizer = GestureRecognizer(config)

        # History buffers
        window_size = self.config.stabilization.voting_window_size
        self._gesture_history: collections.deque[Gesture] = collections.deque(maxlen=window_size)
        
        # State tracking
        self.current_stable_gesture: Gesture = Gesture.NO_HAND
        self._consecutive_candidate: Optional[Gesture] = None
        self._consecutive_count: int = 0
        self._last_detected_time: float = 0.0

        # Mouse look motion state
        self._prev_look_pos: Optional[Tuple[float, float]] = None
        self._smoothed_mouse_dx: float = 0.0
        self._smoothed_mouse_dy: float = 0.0

    def update(self, hand_data: HandData) -> ControlState:
        """Process incoming hand data and return stabilized ControlState with mouse delta."""
        now = time.perf_counter()

        # Handle tracking dropout with grace period
        if not hand_data.detected:
            self._prev_look_pos = None
            self._smoothed_mouse_dx = 0.0
            self._smoothed_mouse_dy = 0.0

            if self._last_detected_time > 0:
                elapsed_ms = (now - self._last_detected_time) * 1000.0
                if elapsed_ms < self.config.stabilization.no_hand_grace_period_ms:
                    return ControlState(
                        gesture=self.current_stable_gesture,
                        raw_gesture=Gesture.NO_HAND,
                        shoot=(self.current_stable_gesture == Gesture.PINCH),
                        aim=(self.current_stable_gesture == Gesture.FIST),
                        move_keys=self._get_active_keys_for_gesture(self.current_stable_gesture),
                        pinch_ratio=1.0,
                        confidence=0.5,
                    )

            self._gesture_history.clear()
            self.current_stable_gesture = Gesture.NO_HAND
            self._consecutive_candidate = None
            self._consecutive_count = 0
            return ControlState(gesture=Gesture.NO_HAND, raw_gesture=Gesture.NO_HAND)

        self._last_detected_time = now

        # Evaluate raw single-frame gesture
        raw_gesture, pinch_ratio, finger_states, move_keys, dx, dy = self.recognizer.evaluate_raw(hand_data)
        self._gesture_history.append(raw_gesture)

        # Consecutive frame threshold check
        if raw_gesture == self._consecutive_candidate:
            self._consecutive_count += 1
        else:
            self._consecutive_candidate = raw_gesture
            self._consecutive_count = 1

        req_consecutive = self.config.stabilization.consecutive_activation_frames
        has_consecutive_support = self._consecutive_count >= req_consecutive

        counts = collections.Counter(self._gesture_history)
        majority_gesture, majority_count = counts.most_common(1)[0]
        majority_ratio = majority_count / len(self._gesture_history)

        # Fast transition for shoot (Pinch) and aim (Fist)
        if raw_gesture in (Gesture.PINCH, Gesture.FIST):
            if self._consecutive_count >= 1:
                self.current_stable_gesture = raw_gesture
        elif has_consecutive_support and majority_gesture == raw_gesture:
            self.current_stable_gesture = raw_gesture

        shoot = (self.current_stable_gesture == Gesture.PINCH)
        aim = (self.current_stable_gesture == Gesture.FIST)

        # Mouse Screen Look (Camera Steering / Aim Look)
        mouse_dx, mouse_dy = self._compute_mouse_look(hand_data, is_aiming=aim)

        return ControlState(
            gesture=self.current_stable_gesture,
            raw_gesture=raw_gesture,
            shoot=shoot,
            aim=aim,
            move_keys=move_keys if not (shoot or aim) else set(),
            mouse_dx=mouse_dx,
            mouse_dy=mouse_dy,
            pinch_ratio=pinch_ratio,
            finger_states=finger_states,
            hand_offset_x=dx,
            hand_offset_y=dy,
            confidence=majority_ratio,
        )

    def _compute_mouse_look(self, hand_data: HandData, is_aiming: bool) -> Tuple[float, float]:
        """Compute smoothed relative mouse delta for in-game camera rotation / screen look."""
        cfg = self.config.inputs.mouse_look
        if not cfg.enabled or not hand_data.detected:
            return 0.0, 0.0

        # Track hand palm center
        curr_pos = hand_data.palm_center_norm

        if self._prev_look_pos is None:
            self._prev_look_pos = curr_pos
            return 0.0, 0.0

        delta_x = (curr_pos[0] - self._prev_look_pos[0])
        delta_y = (curr_pos[1] - self._prev_look_pos[1])
        self._prev_look_pos = curr_pos

        # Deadzone filter
        if abs(delta_x) < cfg.deadzone:
            delta_x = 0.0
        if abs(delta_y) < cfg.deadzone:
            delta_y = 0.0

        # Dynamic velocity acceleration: fast swipes get an acceleration boost
        move_speed = math.hypot(delta_x, delta_y)
        accel = getattr(cfg, "acceleration", 1.5) if move_speed > 0.012 else 1.0

        # Scale to screen pixels
        pixel_dx = delta_x * cfg.sensitivity_x * accel
        pixel_dy = delta_y * cfg.sensitivity_y * accel

        # Apply ADS multiplier when aiming for precision target acquisition
        if is_aiming:
            pixel_dx *= cfg.ads_multiplier
            pixel_dy *= cfg.ads_multiplier

        # Exponential moving average velocity smoothing
        alpha = cfg.smoothing
        self._smoothed_mouse_dx = alpha * self._smoothed_mouse_dx + (1.0 - alpha) * pixel_dx
        self._smoothed_mouse_dy = alpha * self._smoothed_mouse_dy + (1.0 - alpha) * pixel_dy

        # Cut off near-zero residual drift
        final_dx = self._smoothed_mouse_dx if abs(self._smoothed_mouse_dx) > 0.2 else 0.0
        final_dy = self._smoothed_mouse_dy if abs(self._smoothed_mouse_dy) > 0.2 else 0.0

        return final_dx, final_dy

    def _get_active_keys_for_gesture(self, gesture: Gesture) -> Set[str]:
        if gesture == Gesture.MOVE_UP:
            return {self.config.inputs.key_forward}
        if gesture == Gesture.MOVE_DOWN:
            return {self.config.inputs.key_backward}
        if gesture == Gesture.MOVE_LEFT:
            return {self.config.inputs.key_left}
        if gesture == Gesture.MOVE_RIGHT:
            return {self.config.inputs.key_right}
        return set()

    def reset(self) -> None:
        self._gesture_history.clear()
        self.current_stable_gesture = Gesture.NO_HAND
        self._consecutive_candidate = None
        self._consecutive_count = 0
        self._last_detected_time = 0.0
        self._prev_look_pos = None
        self._smoothed_mouse_dx = 0.0
        self._smoothed_mouse_dy = 0.0
        self.recognizer._is_pinched = False
