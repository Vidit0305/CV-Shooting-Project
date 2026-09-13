"""Main application entry point for the Computer Vision Gesture Shooting Controller.

Orchestrates HD camera capture, MediaPipe hand tracking, geometric gesture recognition,
mouse screen look / camera steering, OS input dispatching, global hotkeys, and borderless
full-window edge-to-edge rendering with zero letterbox bars.
"""

from __future__ import annotations

import argparse
import logging
import platform
import signal
import subprocess
import sys
import threading
from typing import Optional, Tuple

import cv2
import numpy as np

# Global hotkey listener via pynput
try:
    from pynput import keyboard as pynput_kbd
    PYNPUT_KBD_AVAILABLE = True
except ImportError:
    PYNPUT_KBD_AVAILABLE = False

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


def get_x11_window_size(win_name: str) -> Tuple[Optional[int], Optional[int]]:
    """Query X11 window geometry using xwininfo to get true widget dimensions."""
    if platform.system() != "Linux":
        return None, None
    try:
        out = subprocess.check_output(["xwininfo", "-name", win_name], stderr=subprocess.DEVNULL, text=True)
        w, h = None, None
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("Width:"):
                w = int(line.split()[1])
            elif line.startswith("Height:"):
                h = int(line.split()[1])
        return w, h
    except Exception:
        return None, None


def fit_to_cover(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Scale and center-crop frame to fill target_w x target_h edge-to-edge with zero bars."""
    fh, fw = frame.shape[:2]
    if fw == target_w and fh == target_h:
        return frame

    scale = max(target_w / fw, target_h / fh)
    scaled_w = int(round(fw * scale))
    scaled_h = int(round(fh * scale))
    resized = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

    crop_x = max(0, (scaled_w - target_w) // 2)
    crop_y = max(0, (scaled_h - target_h) // 2)
    return resized[crop_y : crop_y + target_h, crop_x : crop_x + target_w]


class GestureShootingApp:
    """Main application lifecycle controller."""

    def __init__(self, config_path: str = "config.json", debug: bool = False, fullscreen: bool = False):
        self.config_path = config_path
        self.config = AppConfig.load(config_path)
        if debug:
            self.config.ui.show_debug = True
        if fullscreen:
            self.config.ui.fullscreen = True

        self.running: bool = False
        self.is_fullscreen: bool = self.config.ui.fullscreen
        self.tracker: Optional[HandTracker] = None
        self.stabilizer = GestureStabilizer(self.config)
        self.input_ctrl = InputController(self.config)
        self.calibrator = Calibrator(self.config)
        self.ui = UIRenderer(self.config)
        self.fps_counter = FPSCounter(self.config.ui.fps_smoothing_factor)

        # Global hotkey listener
        self._global_listener: Optional[pynput_kbd.Listener] = None

        # Cached window dimensions
        self._cached_win_w: int = self.config.camera.target_width
        self._cached_win_h: int = self.config.camera.target_height
        self._frame_count: int = 0

    def initialize(self) -> None:
        """Initialize camera, global listener, and hardware resources."""
        logger.info("Initializing Gesture Shooting Controller...")
        self.tracker = HandTracker(self.config)
        self.running = True

        # Ensure controls start active
        self.input_ctrl.set_enabled(True)

        # Start global keyboard listener so controls can be toggled even when in-game
        if PYNPUT_KBD_AVAILABLE:
            try:
                self._global_listener = pynput_kbd.Listener(on_press=self._on_global_key_press)
                self._global_listener.daemon = True
                self._global_listener.start()
                logger.info("Global background hotkey listener started successfully.")
            except Exception as e:
                logger.warning("Could not start global hotkey listener: %s", e)

        # Signal handlers for clean termination on SIGINT / SIGTERM
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum: int, frame) -> None:
        """Handle interrupt signal safely."""
        logger.info("Received termination signal (%d). Shutting down safely...", signum)
        self.running = False

    def _on_global_key_press(self, key) -> None:
        """Handle global keyboard events from any focused application/game."""
        try:
            char = getattr(key, "char", None)
            if char:
                c = char.lower()
                # Toggle controls with 'c'
                if c == self.config.keybinds.key_toggle_controls.lower():
                    new_state = self.input_ctrl.toggle_enabled()
                    logger.info(">>> GLOBAL KEY [C] PRESSED -> Controls are now %s <<<",
                                "ACTIVE (Live Game Input)" if new_state else "PAUSED")
                # Emergency Stop with 'q'
                elif c == self.config.keybinds.key_emergency_stop.lower():
                    logger.info(">>> GLOBAL EMERGENCY STOP KEY [Q] PRESSED <<<")
                    self.running = False
                # Fullscreen with 'f'
                elif c == self.config.keybinds.key_fullscreen.lower():
                    self.toggle_fullscreen()
                # Hide/Show HUD with 'h'
                elif c == self.config.keybinds.key_toggle_hud.lower():
                    self.config.ui.show_hud = not self.config.ui.show_hud
                    logger.info("HUD Visibility: %s", self.config.ui.show_hud)
            elif key == pynput_kbd.Key.esc:
                logger.info(">>> GLOBAL EMERGENCY STOP KEY [ESC] PRESSED <<<")
                self.running = False
        except Exception:
            pass

    def toggle_fullscreen(self) -> None:
        """Toggle between borderless fullscreen and windowed mode."""
        self.is_fullscreen = not self.is_fullscreen
        win_name = self.config.ui.window_name
        prop = cv2.WINDOW_FULLSCREEN if self.is_fullscreen else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, prop)
        logger.info("Fullscreen Mode: %s", "ENABLED" if self.is_fullscreen else "WINDOWED")

    def run(self) -> None:
        """Run the main processing and rendering loop."""
        if not self.tracker:
            self.initialize()

        win_name = self.config.ui.window_name
        # Use GUI_NORMAL and FREERATIO to eliminate white toolbars and border padding
        flags = cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_FREERATIO
        cv2.namedWindow(win_name, flags)
        cv2.resizeWindow(win_name, self.tracker.frame_width, self.tracker.frame_height)

        if self.is_fullscreen:
            cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        logger.info("Application loop online. CONTROLS: ACTIVE by default.")
        logger.info("Hotkeys (work globally): [C] Toggle Controls | [F] Fullscreen | [H] Toggle HUD | [Q / ESC] Quit")

        try:
            while self.running:
                ret, frame = self.tracker.read_frame()
                if not ret or frame is None:
                    logger.warning("Failed to retrieve webcam frame. Re-attempting...")
                    cv2.waitKey(10)
                    continue

                self._frame_count += 1
                fps = self.fps_counter.update()

                # Periodically query true window dimensions to eliminate all white/black letterboxing
                if self._frame_count % 20 == 1:
                    w_real, h_real = get_x11_window_size(win_name)
                    if w_real and h_real and w_real > 100 and h_real > 100:
                        self._cached_win_w = w_real
                        self._cached_win_h = h_real

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

                # 5. Fit camera frame to window edge-to-edge (zero black/white borders)
                cover_frame = fit_to_cover(frame, self._cached_win_w, self._cached_win_h)

                # 6. Render Floating Minimalist HUD
                if self.config.ui.show_hud:
                    display_frame = self.ui.render(
                        frame=cover_frame,
                        hand_data=hand_data,
                        control_state=control_state,
                        controls_enabled=self.input_ctrl.enabled,
                        fps=fps,
                        calibrator=self.calibrator,
                        debug_mode=self.config.ui.show_debug,
                    )
                else:
                    display_frame = cover_frame

                # 7. Display Frame Edge-to-Edge
                cv2.imshow(win_name, display_frame)

                # 8. Check window close button (X)
                if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                    logger.info("Window close event detected.")
                    break

                # 9. Handle local window keypresses
                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    self._handle_local_keypress(key)

        except Exception as e:
            logger.error("Unhandled exception in main loop: %s", e, exc_info=True)
        finally:
            self.shutdown()

    def _handle_local_keypress(self, key: int) -> None:
        """Process keyboard commands when OpenCV window has focus."""
        if key == ord(self.config.keybinds.key_emergency_stop.lower()) or key == self.config.keybinds.key_escape_code:
            self.running = False
        elif key == ord(self.config.keybinds.key_toggle_controls.lower()):
            self.input_ctrl.toggle_enabled()
        elif key == ord(self.config.keybinds.key_fullscreen.lower()):
            self.toggle_fullscreen()
        elif key == ord(self.config.keybinds.key_toggle_hud.lower()):
            self.config.ui.show_hud = not self.config.ui.show_hud
        elif key == ord(self.config.keybinds.key_calibrate.lower()):
            if not self.calibrator.is_active():
                self.calibrator.start()
            else:
                self.calibrator.cancel()
        elif key == ord(self.config.keybinds.key_toggle_debug.lower()):
            self.config.ui.show_debug = not self.config.ui.show_debug

    def shutdown(self) -> None:
        """Safely release all OS inputs and hardware resources."""
        logger.info("Shutting down Gesture Shooting Controller...")
        self.running = False

        # Stop global keyboard listener
        if self._global_listener and self._global_listener.is_alive():
            try:
                self._global_listener.stop()
            except Exception:
                pass

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

    app = GestureShootingApp(config_path=args.config, debug=args.debug, fullscreen=args.fullscreen)
    if args.camera is not None:
        app.config.camera.camera_index = args.camera

    app.initialize()
    app.run()


if __name__ == "__main__":
    main()
