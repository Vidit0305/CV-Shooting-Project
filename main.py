"""Main application entry point for the Computer Vision Gesture Shooting Controller.

Orchestrates HD camera capture, MediaPipe hand tracking, geometric gesture recognition,
mouse screen look / camera steering, OS input dispatching, and minimalist HUD rendering.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from typing import Optional

import cv2
import numpy as np

from calibration import Calibrator
from config import AppConfig
from gesture_controller import GestureStabilizer
from hand_tracker import HandTracker
from input_controller import InputController
from ui import UIRenderer
from utils import FPSCounter

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")


class GestureShootingApp:
    """Main application lifecycle controller."""

    def __init__(self, config_path: str = "config.json", debug: bool = False):
        self.config_path = config_path
        self.config = AppConfig.load(config_path)
        if debug:
            self.config.ui.show_debug = True

        self.running: bool = False
        self.is_fullscreen: bool = self.config.ui.fullscreen
        self.tracker: Optional[HandTracker] = None
        self.stabilizer = GestureStabilizer(self.config)
        self.input_ctrl = InputController(self.config)
        self.calibrator = Calibrator(self.config)
        self.ui = UIRenderer(self.config)
        self.fps_counter = FPSCounter(self.config.ui.fps_smoothing_factor)

    def initialize(self) -> None:
        """Initialize camera and hardware resources."""
        logger.info("Initializing Gesture Shooting Controller...")
        self.tracker = HandTracker(self.config)
        self.running = True

        # Signal handlers for clean termination on SIGINT / SIGTERM
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum: int, frame) -> None:
        """Handle interrupt signal safely."""
        logger.info("Received termination signal (%d). Shutting down safely...", signum)
        self.running = False

    def run(self) -> None:
        """Run the main processing and rendering loop."""
        if not self.tracker:
            self.initialize()

        win_name = self.config.ui.window_name
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, self.tracker.frame_width, self.tracker.frame_height)

        if self.is_fullscreen:
            cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        logger.info("Application loop started. Press '%s' to toggle controls, '%s' to quit, '%s' for fullscreen.",
                    self.config.keybinds.key_toggle_controls.upper(),
                    self.config.keybinds.key_emergency_stop.upper(),
                    self.config.keybinds.key_fullscreen.upper())

        try:
            while self.running:
                ret, frame = self.tracker.read_frame()
                if not ret or frame is None:
                    logger.warning("Failed to retrieve webcam frame. Re-attempting...")
                    cv2.waitKey(10)
                    continue

                fps = self.fps_counter.update()

                # 1. MediaPipe Hand Tracking (with landmark smoothing)
                hand_data = self.tracker.process(frame)

                # 2. Calibration Processing (if active)
                if self.calibrator.is_active():
                    self.calibrator.update(hand_data)

                # 3. Gesture Recognition, Stabilization & Mouse Look Delta
                control_state = self.stabilizer.update(hand_data)

                # 4. OS Input Dispatch (WASD + Mouse Shoot/Aim + Mouse Screen Look)
                if not self.calibrator.is_active():
                    self.input_ctrl.apply_control_state(
                        shoot=control_state.shoot,
                        aim=control_state.aim,
                        move_keys=control_state.move_keys,
                        mouse_dx=control_state.mouse_dx,
                        mouse_dy=control_state.mouse_dy,
                    )
                else:
                    self.input_ctrl.release_all_inputs()

                # 5. Draw Minimalist Skeleton & Landmarks
                if self.config.ui.show_landmarks and hand_data.detected and self.config.ui.show_hud:
                    self.tracker.draw_landmarks(
                        frame,
                        hand_data,
                        highlight_pinch=control_state.shoot,
                        highlight_fist=control_state.aim,
                    )

                # 6. Render Floating Minimalist HUD
                if self.config.ui.show_hud:
                    rendered_frame = self.ui.render(
                        frame=frame,
                        hand_data=hand_data,
                        control_state=control_state,
                        controls_enabled=self.input_ctrl.enabled,
                        fps=fps,
                        calibrator=self.calibrator,
                        debug_mode=self.config.ui.show_debug,
                    )
                else:
                    rendered_frame = frame

                # 7. Aspect Ratio / Letterbox handling (Guarantees zero white borders)
                final_display = self._fit_to_window(rendered_frame, win_name)

                # 8. Display Frame
                cv2.imshow(win_name, final_display)

                # 9. Check window close button (X)
                if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                    logger.info("Window close event detected.")
                    break

                # 10. Handle Keypresses
                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    self._handle_keypress(key, win_name)

        except Exception as e:
            logger.error("Unhandled exception in main loop: %s", e, exc_info=True)
        finally:
            self.shutdown()

    def _fit_to_window(self, frame: np.ndarray, win_name: str) -> np.ndarray:
        """Pad and scale frame with deep black background to prevent OpenCV white pillarboxing."""
        try:
            rect = cv2.getWindowImageRect(win_name)
            if rect and len(rect) >= 4:
                _, _, win_w, win_h = rect
                if win_w > 100 and win_h > 100:
                    fh, fw = frame.shape[:2]
                    scale = min(win_w / fw, win_h / fh)
                    new_w = int(fw * scale)
                    new_h = int(fh * scale)

                    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    canvas = np.full((win_h, win_w, 3), (10, 10, 14), dtype=np.uint8)

                    ox = (win_w - new_w) // 2
                    oy = (win_h - new_h) // 2
                    canvas[oy : oy + new_h, ox : ox + new_w] = resized
                    return canvas
        except Exception:
            pass
        return frame

    def _handle_keypress(self, key: int, win_name: str) -> None:
        """Process keyboard commands."""
        # Emergency Stop / Exit
        if key == ord(self.config.keybinds.key_emergency_stop.lower()) or key == self.config.keybinds.key_escape_code:
            logger.info("Emergency stop triggered via keypress.")
            self.running = False
            return

        # Toggle Game Controls Enabled / Disabled
        if key == ord(self.config.keybinds.key_toggle_controls.lower()):
            new_state = self.input_ctrl.toggle_enabled()
            logger.info("Toggled Game Controls -> %s", "ACTIVE" if new_state else "PAUSED")

        # Toggle Fullscreen Mode (F)
        elif key == ord(self.config.keybinds.key_fullscreen.lower()):
            self.is_fullscreen = not self.is_fullscreen
            prop = cv2.WINDOW_FULLSCREEN if self.is_fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, prop)
            logger.info("Fullscreen Mode: %s", "ENABLED" if self.is_fullscreen else "WINDOWED")

        # Toggle HUD Visibility (H)
        elif key == ord(self.config.keybinds.key_toggle_hud.lower()):
            self.config.ui.show_hud = not self.config.ui.show_hud
            logger.info("HUD Visibility: %s", self.config.ui.show_hud)

        # Recalibrate Hand Geometry (R)
        elif key == ord(self.config.keybinds.key_calibrate.lower()):
            if not self.calibrator.is_active():
                self.calibrator.start()
            else:
                self.calibrator.cancel()

        # Toggle Debug Metrics Overlay (D)
        elif key == ord(self.config.keybinds.key_toggle_debug.lower()):
            self.config.ui.show_debug = not self.config.ui.show_debug
            logger.info("Debug overlay: %s", self.config.ui.show_debug)

    def shutdown(self) -> None:
        """Safely release all OS inputs and hardware resources."""
        logger.info("Shutting down Gesture Shooting Controller...")
        self.running = False

        # Guaranteed release of all pressed keys and mouse buttons
        try:
            self.input_ctrl.release_all_inputs()
        except Exception as e:
            logger.error("Error releasing inputs during shutdown: %s", e)

        # Release Camera and MediaPipe
        try:
            if self.tracker:
                self.tracker.release()
        except Exception as e:
            logger.error("Error releasing tracker during shutdown: %s", e)

        # Close all OpenCV Windows
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            logger.error("Error destroying OpenCV windows: %s", e)

        logger.info("Shutdown completed cleanly. All inputs released.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-Time CV Gesture Shooting Game Controller")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json file")
    parser.add_argument("--camera", type=int, default=None, help="Camera device index")
    parser.add_argument("--fullscreen", action="store_true", help="Start directly in fullscreen mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug telemetry overlay")
    args = parser.parse_args()

    app = GestureShootingApp(config_path=args.config, debug=args.debug)
    if args.camera is not None:
        app.config.camera.camera_index = args.camera
    if args.fullscreen:
        app.is_fullscreen = True

    app.initialize()
    app.run()


if __name__ == "__main__":
    main()
