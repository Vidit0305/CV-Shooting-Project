"""Generates high-resolution 720p demonstration screenshots of active gesture states.

Produces demo images for Shoot (Pinch), Aim (Fist), and Movement (WASD)
with the minimalist cyber HUD styling.
"""

from __future__ import annotations

import os
import sys

# Ensure root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np

from config import AppConfig
from gesture_controller import ControlState, Gesture
from hand_tracker import HandData, HandTracker
from tests.test_components import create_synthetic_hand
from ui import UIRenderer


def generate_demos() -> None:
    os.makedirs("screenshots", exist_ok=True)
    config = AppConfig.load()
    ui = UIRenderer(config)

    w, h = 1280, 720

    tracker_dummy = HandTracker.__new__(HandTracker)
    tracker_dummy.mp_hands = type("Dummy", (), {"HAND_CONNECTIONS": [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
    ]})()

    # 1. Demo Shoot (Pinch)
    img_shoot = np.full((h, w, 3), (18, 18, 24), dtype=np.uint8)
    lms_shoot = create_synthetic_hand(wrist_xy=(0.50, 0.60), finger_extension=1.0, pinch_distance=0.03)
    px_shoot = [(int(pt.x * w), int(pt.y * h)) for pt in lms_shoot]
    hd_shoot = HandData(
        detected=True,
        landmarks_norm=lms_shoot,
        landmarks_px=px_shoot,
        palm_center_norm=(0.50, 0.60),
        palm_center_px=(int(0.50 * w), int(0.60 * h)),
        palm_scale=0.18,
        handedness="Right",
        confidence=0.96,
    )
    cs_shoot = ControlState(
        gesture=Gesture.PINCH,
        raw_gesture=Gesture.PINCH,
        shoot=True,
        aim=False,
        move_keys=set(),
        mouse_dx=0.0,
        mouse_dy=0.0,
        pinch_ratio=0.16,
        confidence=0.96,
    )

    HandTracker.draw_landmarks(tracker_dummy, img_shoot, hd_shoot, highlight_pinch=True, highlight_fist=False)
    frame_shoot = ui.render(frame=img_shoot, hand_data=hd_shoot, control_state=cs_shoot, controls_enabled=True, fps=30.0, debug_mode=True)
    cv2.imwrite("screenshots/demo_shoot_pinch.png", frame_shoot)
    print("Generated screenshots/demo_shoot_pinch.png")

    # 2. Demo Aim (Fist ADS)
    img_aim = np.full((h, w, 3), (18, 18, 24), dtype=np.uint8)
    lms_aim = create_synthetic_hand(wrist_xy=(0.50, 0.60), finger_extension=0.0)
    px_aim = [(int(pt.x * w), int(pt.y * h)) for pt in lms_aim]
    hd_aim = HandData(
        detected=True,
        landmarks_norm=lms_aim,
        landmarks_px=px_aim,
        palm_center_norm=(0.50, 0.60),
        palm_center_px=(int(0.50 * w), int(0.60 * h)),
        palm_scale=0.16,
        handedness="Right",
        confidence=0.95,
    )
    cs_aim = ControlState(
        gesture=Gesture.FIST,
        raw_gesture=Gesture.FIST,
        shoot=False,
        aim=True,
        move_keys=set(),
        mouse_dx=0.0,
        mouse_dy=0.0,
        pinch_ratio=0.85,
        confidence=0.95,
    )

    HandTracker.draw_landmarks(tracker_dummy, img_aim, hd_aim, highlight_pinch=False, highlight_fist=True)
    frame_aim = ui.render(frame=img_aim, hand_data=hd_aim, control_state=cs_aim, controls_enabled=True, fps=30.0, debug_mode=True)
    cv2.imwrite("screenshots/demo_aim_fist.png", frame_aim)
    print("Generated screenshots/demo_aim_fist.png")

    # 3. Demo Move WASD (W + D with Screen Look Mouse Delta)
    img_move = np.full((h, w, 3), (18, 18, 24), dtype=np.uint8)
    lms_move = create_synthetic_hand(wrist_xy=(0.65, 0.40), finger_extension=1.0, pinch_distance=0.25)
    px_move = [(int(pt.x * w), int(pt.y * h)) for pt in lms_move]
    hd_move = HandData(
        detected=True,
        landmarks_norm=lms_move,
        landmarks_px=px_move,
        palm_center_norm=(0.65, 0.40),
        palm_center_px=(int(0.65 * w), int(0.40 * h)),
        palm_scale=0.18,
        handedness="Right",
        confidence=0.98,
    )
    cs_move = ControlState(
        gesture=Gesture.MOVE_UP,
        raw_gesture=Gesture.MOVE_UP,
        shoot=False,
        aim=False,
        move_keys={"w", "d"},
        mouse_dx=14.0,
        mouse_dy=-8.0,
        pinch_ratio=0.80,
        confidence=0.98,
    )

    HandTracker.draw_landmarks(tracker_dummy, img_move, hd_move, highlight_pinch=False, highlight_fist=False)
    frame_move = ui.render(frame=img_move, hand_data=hd_move, control_state=cs_move, controls_enabled=True, fps=30.0, debug_mode=True)
    cv2.imwrite("screenshots/demo_move_wasd.png", frame_move)
    print("Generated screenshots/demo_move_wasd.png")


if __name__ == "__main__":
    generate_demos()
