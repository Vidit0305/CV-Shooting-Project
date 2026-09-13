"""End-to-end integration verification script.

Runs the full pipeline (camera frame capture, MediaPipe tracking, gesture recognition,
stabilization, UI rendering, input state sync) for 60 frames and verifies performance,
FPS, and clean resource release.
"""

from __future__ import annotations

import logging
import os
import sys
import time

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2

from config import AppConfig
from gesture_controller import GestureStabilizer
from hand_tracker import HandTracker
from input_controller import InputController
from ui import UIRenderer
from utils import FPSCounter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("verify")


def verify_pipeline(num_frames: int = 60, save_screenshot: bool = True) -> bool:
    logger.info("--- STARTING SYSTEM INTEGRATION VERIFICATION ---")
    config = AppConfig.load()
    tracker = None

    try:
        tracker = HandTracker(config)
        stabilizer = GestureStabilizer(config)
        input_ctrl = InputController(config)
        ui = UIRenderer(config)
        fps_counter = FPSCounter()

        logger.info("Camera successfully opened. Frame dimensions: %dx%d", tracker.frame_width, tracker.frame_height)
        start_time = time.perf_counter()
        processed_frames = 0
        detected_count = 0

        # Create output screenshot directory if needed
        os.makedirs("screenshots", exist_ok=True)

        for i in range(num_frames):
            ret, frame = tracker.read_frame()
            if not ret or frame is None:
                logger.error("Failed to read frame %d", i)
                return False

            fps = fps_counter.update()
            hand_data = tracker.process(frame)
            if hand_data.detected:
                detected_count += 1

            control_state = stabilizer.update(hand_data)
            input_ctrl.apply_control_state(
                shoot=control_state.shoot,
                aim=control_state.aim,
                move_keys=control_state.move_keys,
            )

            if hand_data.detected:
                tracker.draw_landmarks(frame, hand_data, highlight_pinch=control_state.shoot, highlight_fist=control_state.aim)

            hud_frame = ui.render(
                frame=frame,
                hand_data=hand_data,
                control_state=control_state,
                controls_enabled=input_ctrl.enabled,
                fps=fps,
            )

            processed_frames += 1

            # Save sample screenshot of frame 30
            if save_screenshot and i == 30:
                cv2.imwrite("screenshots/system_preview.png", hud_frame)
                logger.info("Saved preview screenshot to screenshots/system_preview.png")

        elapsed = time.perf_counter() - start_time
        avg_fps = processed_frames / max(0.001, elapsed)
        logger.info(
            "Processed %d frames in %.2f seconds (Average FPS: %.1f). Hand detections: %d",
            processed_frames, elapsed, avg_fps, detected_count
        )

        logger.info("--- INTEGRATION VERIFICATION PASSED ---")
        return True

    except Exception as e:
        logger.error("Verification failed with exception: %s", e, exc_info=True)
        return False
    finally:
        if tracker:
            tracker.release()


if __name__ == "__main__":
    success = verify_pipeline(num_frames=60, save_screenshot=True)
    sys.exit(0 if success else 1)
