"""Automated component and integration tests for CV Gesture Shooting Controller.

Validates geometric calculations, finger state detection, gesture classification,
temporal stabilization, input controller state guarantees, and safety cleanup.
"""

from __future__ import annotations

import time
import unittest
from typing import List

from calibration import CalibrationStep, Calibrator
from config import AppConfig
from gesture_controller import (
    ControlState,
    FingerStates,
    Gesture,
    GestureRecognizer,
    GestureStabilizer,
)
from hand_tracker import HandData
from input_controller import InputController
from utils import (
    FPSCounter,
    LandmarkIndex,
    Point3D,
    calculate_angle_3points,
    calculate_palm_scale,
    euclidean_distance_2d,
    euclidean_distance_3d,
)


def create_synthetic_hand(
    wrist_xy: tuple[float, float] = (0.5, 0.55),
    finger_extension: float = 1.0,
    pinch_distance: float = 0.30,
) -> List[Point3D]:
    """Generate synthetic 21 hand landmarks for reproducible testing.
    
    finger_extension: 1.0 = fully extended upright; 0.0 = fully curled into fist.
    pinch_distance: distance between thumb tip and index tip.
    """
    wx, wy = wrist_xy
    lms: List[Point3D] = [Point3D(wx, wy, 0.0)] * 21

    # Base palm landmarks (invariant structure)
    # Index MCP (5), Middle MCP (9), Ring MCP (13), Pinky MCP (17)
    lms[LandmarkIndex.INDEX_MCP] = Point3D(wx - 0.06, wy - 0.12, 0.0)
    lms[LandmarkIndex.MIDDLE_MCP] = Point3D(wx - 0.02, wy - 0.14, 0.0)
    lms[LandmarkIndex.RING_MCP] = Point3D(wx + 0.02, wy - 0.13, 0.0)
    lms[LandmarkIndex.PINKY_MCP] = Point3D(wx + 0.06, wy - 0.11, 0.0)

    # Helper to generate finger joints (MCP -> PIP -> DIP -> TIP)
    def set_finger(mcp_idx: int, pip_idx: int, dip_idx: int, tip_idx: int, base_x: float, base_y: float):
        mcp = Point3D(base_x, base_y, 0.0)
        lms[mcp_idx] = mcp
        if finger_extension > 0.5:
            # Extended straight upward
            lms[pip_idx] = Point3D(base_x, base_y - 0.06, 0.0)
            lms[dip_idx] = Point3D(base_x, base_y - 0.11, 0.0)
            lms[tip_idx] = Point3D(base_x, base_y - 0.16, 0.0)
        else:
            # Curled inward toward palm
            lms[pip_idx] = Point3D(base_x, base_y - 0.04, 0.02)
            lms[dip_idx] = Point3D(base_x, base_y - 0.02, 0.05)
            lms[tip_idx] = Point3D(base_x, base_y + 0.01, 0.06)

    # Setup index, middle, ring, pinky
    set_finger(LandmarkIndex.INDEX_MCP, LandmarkIndex.INDEX_PIP, LandmarkIndex.INDEX_DIP, LandmarkIndex.INDEX_TIP, wx - 0.06, wy - 0.12)
    set_finger(LandmarkIndex.MIDDLE_MCP, LandmarkIndex.MIDDLE_PIP, LandmarkIndex.MIDDLE_DIP, LandmarkIndex.MIDDLE_TIP, wx - 0.02, wy - 0.14)
    set_finger(LandmarkIndex.RING_MCP, LandmarkIndex.RING_PIP, LandmarkIndex.RING_DIP, LandmarkIndex.RING_TIP, wx + 0.02, wy - 0.13)
    set_finger(LandmarkIndex.PINKY_MCP, LandmarkIndex.PINKY_PIP, LandmarkIndex.PINKY_DIP, LandmarkIndex.PINKY_TIP, wx + 0.06, wy - 0.11)

    # Thumb setup
    lms[LandmarkIndex.THUMB_CMC] = Point3D(wx - 0.04, wy - 0.03, 0.0)
    lms[LandmarkIndex.THUMB_MCP] = Point3D(wx - 0.08, wy - 0.06, 0.0)
    lms[LandmarkIndex.THUMB_IP] = Point3D(wx - 0.10, wy - 0.09, 0.0)

    if finger_extension <= 0.5:
        # Thumb folded across fist
        lms[LandmarkIndex.THUMB_TIP] = Point3D(wx - 0.02, wy - 0.07, 0.05)
    else:
        # Thumb position determined by pinch_distance from index tip
        idx_tip = lms[LandmarkIndex.INDEX_TIP]
        lms[LandmarkIndex.THUMB_TIP] = Point3D(idx_tip.x - pinch_distance, idx_tip.y, 0.0)

    return lms


class TestUtils(unittest.TestCase):
    """Test geometric calculations and FPS tracker."""

    def test_euclidean_distances(self):
        p1 = Point3D(0.0, 0.0, 0.0)
        p2 = Point3D(3.0, 4.0, 0.0)
        self.assertAlmostEqual(euclidean_distance_2d(p1, p2), 5.0)
        self.assertAlmostEqual(euclidean_distance_3d(p1, p2), 5.0)

        p3 = Point3D(0.0, 0.0, 12.0)
        self.assertAlmostEqual(euclidean_distance_3d(p1, p3), 12.0)

    def test_calculate_angle_3points(self):
        # 90-degree right angle at (0, 0)
        a = Point3D(0.0, 1.0, 0.0)
        b = Point3D(0.0, 0.0, 0.0)
        c = Point3D(1.0, 0.0, 0.0)
        angle = calculate_angle_3points(a, b, c)
        self.assertAlmostEqual(angle, 90.0, places=1)

        # 180-degree straight line
        d = Point3D(0.0, -1.0, 0.0)
        angle_straight = calculate_angle_3points(a, b, d)
        self.assertAlmostEqual(angle_straight, 180.0, places=1)

    def test_palm_scale(self):
        lms = create_synthetic_hand()
        scale = calculate_palm_scale(lms)
        self.assertGreater(scale, 0.05)
        self.assertLess(scale, 0.50)

    def test_fps_counter(self):
        fps = FPSCounter(smoothing_factor=0.5)
        # Call update multiple times
        val = fps.update()
        time.sleep(0.02)
        val = fps.update()
        self.assertGreater(val, 0.0)


class TestConfig(unittest.TestCase):
    """Test configuration defaults and serialization."""

    def test_config_defaults(self):
        config = AppConfig()
        self.assertEqual(config.inputs.key_forward, "w")
        self.assertEqual(config.inputs.key_backward, "s")
        self.assertEqual(config.inputs.key_left, "a")
        self.assertEqual(config.inputs.key_right, "d")
        self.assertFalse(config.inputs.controls_enabled_by_default)
        self.assertAlmostEqual(config.gestures.pinch_on_ratio, 0.35)


class TestGestureRecognition(unittest.TestCase):
    """Test gesture detection and stabilization logic."""

    def setUp(self):
        self.config = AppConfig()
        self.recognizer = GestureRecognizer(self.config)
        self.stabilizer = GestureStabilizer(self.config)

    def test_open_hand_neutral(self):
        lms = create_synthetic_hand(finger_extension=1.0, pinch_distance=0.25)
        hand_data = HandData(
            detected=True,
            landmarks_norm=lms,
            palm_center_norm=(0.5, 0.55),
            palm_scale=calculate_palm_scale(lms),
        )
        fs = self.recognizer.detect_finger_states(lms, hand_data.palm_scale)
        self.assertTrue(fs.index)
        self.assertTrue(fs.middle)
        self.assertTrue(fs.ring)
        self.assertTrue(fs.pinky)

        raw_gesture, pinch_ratio, _, move_keys, _, _ = self.recognizer.evaluate_raw(hand_data)
        self.assertEqual(raw_gesture, Gesture.NEUTRAL)
        self.assertEqual(len(move_keys), 0)

    def test_pinch_to_shoot(self):
        # Distance between thumb tip and index tip is very small (0.02)
        lms = create_synthetic_hand(finger_extension=1.0, pinch_distance=0.02)
        hand_data = HandData(
            detected=True,
            landmarks_norm=lms,
            palm_center_norm=(0.5, 0.55),
            palm_scale=calculate_palm_scale(lms),
        )
        raw_gesture, pinch_ratio, _, _, _, _ = self.recognizer.evaluate_raw(hand_data)
        self.assertEqual(raw_gesture, Gesture.PINCH)
        self.assertLess(pinch_ratio, self.config.gestures.pinch_on_ratio)

    def test_fist_to_aim(self):
        # All 4 fingers folded
        lms = create_synthetic_hand(finger_extension=0.0)
        hand_data = HandData(
            detected=True,
            landmarks_norm=lms,
            palm_center_norm=(0.5, 0.55),
            palm_scale=calculate_palm_scale(lms),
        )
        raw_gesture, _, finger_states, _, _, _ = self.recognizer.evaluate_raw(hand_data)
        self.assertEqual(raw_gesture, Gesture.FIST)
        self.assertTrue(finger_states.are_four_folded())

    def test_wasd_movement_directions(self):
        # Hand displaced upward: y = 0.30 (anchor is 0.55, dy = -0.25 < -0.14)
        lms_up = create_synthetic_hand(wrist_xy=(0.50, 0.30), finger_extension=1.0, pinch_distance=0.25)
        hand_up = HandData(detected=True, landmarks_norm=lms_up, palm_center_norm=(0.50, 0.30), palm_scale=0.15)
        raw_up, _, _, move_up, _, _ = self.recognizer.evaluate_raw(hand_up)
        self.assertEqual(raw_up, Gesture.MOVE_UP)
        self.assertIn("w", move_up)

        # Hand displaced downward: y = 0.80 (dy = +0.25 > 0.14)
        lms_down = create_synthetic_hand(wrist_xy=(0.50, 0.80), finger_extension=1.0, pinch_distance=0.25)
        hand_down = HandData(detected=True, landmarks_norm=lms_down, palm_center_norm=(0.50, 0.80), palm_scale=0.15)
        raw_down, _, _, move_down, _, _ = self.recognizer.evaluate_raw(hand_down)
        self.assertEqual(raw_down, Gesture.MOVE_DOWN)
        self.assertIn("s", move_down)

        # Hand displaced left: x = 0.25 (anchor is 0.50, dx = -0.25 < -0.14)
        lms_left = create_synthetic_hand(wrist_xy=(0.25, 0.55), finger_extension=1.0, pinch_distance=0.25)
        hand_left = HandData(detected=True, landmarks_norm=lms_left, palm_center_norm=(0.25, 0.55), palm_scale=0.15)
        raw_left, _, _, move_left, _, _ = self.recognizer.evaluate_raw(hand_left)
        self.assertEqual(raw_left, Gesture.MOVE_LEFT)
        self.assertIn("a", move_left)

        # Hand displaced right: x = 0.75 (dx = +0.25 > 0.14)
        lms_right = create_synthetic_hand(wrist_xy=(0.75, 0.55), finger_extension=1.0, pinch_distance=0.25)
        hand_right = HandData(detected=True, landmarks_norm=lms_right, palm_center_norm=(0.75, 0.55), palm_scale=0.15)
        raw_right, _, _, move_right, _, _ = self.recognizer.evaluate_raw(hand_right)
        self.assertEqual(raw_right, Gesture.MOVE_RIGHT)
        self.assertIn("d", move_right)

    def test_stabilization_sliding_window(self):
        # Initial state is NO_HAND
        lms_pinch = create_synthetic_hand(finger_extension=1.0, pinch_distance=0.02)
        hand_pinch = HandData(detected=True, landmarks_norm=lms_pinch, palm_center_norm=(0.5, 0.55), palm_scale=0.15)

        # Feed 1 frame: should not immediately transition if consecutive requirement is 3
        state = self.stabilizer.update(hand_pinch)
        self.assertEqual(state.raw_gesture, Gesture.PINCH)

        # Feed remaining frames to satisfy consecutive threshold (3 frames)
        self.stabilizer.update(hand_pinch)
        state = self.stabilizer.update(hand_pinch)
        self.assertEqual(state.gesture, Gesture.PINCH)
        self.assertTrue(state.shoot)


    def test_mouse_look_delta(self):
        # Frame 1: Hand at (0.50, 0.50)
        lms1 = create_synthetic_hand(wrist_xy=(0.50, 0.50), finger_extension=1.0, pinch_distance=0.25)
        h1 = HandData(detected=True, landmarks_norm=lms1, palm_center_norm=(0.50, 0.50), palm_scale=0.18)
        self.stabilizer.update(h1)

        # Frame 2: Hand moved to (0.52, 0.51) -> dx = +0.02, dy = +0.01
        lms2 = create_synthetic_hand(wrist_xy=(0.52, 0.51), finger_extension=1.0, pinch_distance=0.25)
        h2 = HandData(detected=True, landmarks_norm=lms2, palm_center_norm=(0.52, 0.51), palm_scale=0.18)
        state2 = self.stabilizer.update(h2)

        # mouse_dx should be positive and non-zero
        self.assertGreater(state2.mouse_dx, 0.0)
        self.assertGreater(state2.mouse_dy, 0.0)


class TestInputController(unittest.TestCase):
    """Test OS input controller state management, conflict prevention, and safety release."""

    def setUp(self):
        self.config = AppConfig()
        self.controller = InputController(self.config)

    def test_move_mouse(self):
        # Should not throw any exception when moving mouse
        self.controller.set_enabled(True)
        self.controller.move_mouse(5.0, -5.0)
        self.controller.set_enabled(False)

    def test_disabled_by_default(self):
        self.assertFalse(self.controller.enabled)
        # When disabled, pressing key updates internal state if called directly, but no OS crash
        self.controller.set_enabled(False)
        self.controller.press_key("w")
        self.assertIn("w", self.controller.pressed_keys)
        self.controller.release_key("w")
        self.assertNotIn("w", self.controller.pressed_keys)

    def test_directional_conflict_resolution(self):
        # Desired: both 'w' and 's' simultaneously
        self.controller.sync_movement_keys(desired_keys={"w", "s"})
        # Should resolve conflict and retain only one direction
        has_both = ("w" in self.controller.pressed_keys) and ("s" in self.controller.pressed_keys)
        self.assertFalse(has_both)

        # Desired: both 'a' and 'd' simultaneously
        self.controller.sync_movement_keys(desired_keys={"a", "d"})
        has_both_horizontal = ("a" in self.controller.pressed_keys) and ("d" in self.controller.pressed_keys)
        self.assertFalse(has_both_horizontal)

    def test_safety_release_all_inputs(self):
        self.controller.pressed_keys.update(["w", "a", "s", "d"])
        self.controller.pressed_mouse_buttons.update(["left", "right"])

        self.controller.release_all_inputs()
        self.assertEqual(len(self.controller.pressed_keys), 0)
        self.assertEqual(len(self.controller.pressed_mouse_buttons), 0)

    def test_toggle_enabled_releases_inputs(self):
        self.controller.set_enabled(True)
        self.controller.pressed_keys.add("w")
        self.controller.pressed_mouse_buttons.add("left")

        # Disabling must trigger emergency release
        self.controller.set_enabled(False)
        self.assertEqual(len(self.controller.pressed_keys), 0)
        self.assertEqual(len(self.controller.pressed_mouse_buttons), 0)


class TestCalibrator(unittest.TestCase):
    """Test calibration step workflow and threshold computation."""

    def test_calibration_lifecycle(self):
        config = AppConfig()
        calibrator = Calibrator(config, samples_per_step=5)
        self.assertEqual(calibrator.step, CalibrationStep.IDLE)

        calibrator.start()
        self.assertEqual(calibrator.step, CalibrationStep.INSTRUCT_OPEN)
        self.assertTrue(calibrator.is_active())

        calibrator.cancel()
        self.assertEqual(calibrator.step, CalibrationStep.IDLE)
        self.assertFalse(calibrator.is_active())


if __name__ == "__main__":
    unittest.main()
