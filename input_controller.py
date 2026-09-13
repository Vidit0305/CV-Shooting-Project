"""OS-level keyboard and mouse automation controller.

Manages stateful key/mouse tracking, avoids redundant input spam,
provides smooth relative mouse screen look / aiming, prevents conflicting directional
keys (W+S, A+D), provides enable/disable safety gates, and guarantees clean emergency input release.
"""

from __future__ import annotations

import logging
from typing import Optional, Set

from config import AppConfig

logger = logging.getLogger(__name__)

# Attempt importing pynput and pyautogui
try:
    from pynput.keyboard import Controller as PynputKeyboardController, Key as PynputKey
    from pynput.mouse import Button as PynputButton, Controller as PynputMouseController
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    logger.warning("pynput is not installed. Will fallback to pyautogui.")

try:
    import pyautogui
    pyautogui.PAUSE = 0.0
    pyautogui.FAILSAFE = False
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False
    logger.warning("pyautogui is not installed.")


class InputController:
    """Manages real OS-level inputs with state tracking and safety controls."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.enabled: bool = config.inputs.controls_enabled_by_default

        # Internal state tracking to prevent repeated calls
        self.pressed_keys: Set[str] = set()
        self.pressed_mouse_buttons: Set[str] = set()

        # Hardware backend initialization
        self._pynput_kbd: Optional[PynputKeyboardController] = None
        self._pynput_mouse: Optional[PynputMouseController] = None

        if PYNPUT_AVAILABLE:
            try:
                self._pynput_kbd = PynputKeyboardController()
                self._pynput_mouse = PynputMouseController()
                logger.info("Initialized pynput input controller backend.")
            except Exception as e:
                logger.error("Failed to initialize pynput: %s", e)
                self._pynput_kbd = None
                self._pynput_mouse = None

        if not self._pynput_kbd and not PYAUTOGUI_AVAILABLE:
            logger.error("No working OS input library available! Inputs will be simulated.")

    def set_enabled(self, enabled: bool) -> None:
        """Toggle input sending on or off. Releases all inputs when disabled."""
        if self.enabled == enabled:
            return

        self.enabled = enabled
        if not self.enabled:
            logger.info("Controls DISABLED - Releasing all active inputs.")
            self.release_all_inputs()
        else:
            logger.info("Controls ENABLED - Active game inputs will now be sent to OS.")

    def toggle_enabled(self) -> bool:
        """Toggle control active state and return new state."""
        self.set_enabled(not self.enabled)
        return self.enabled

    # --- Mouse Movement / Screen Look (Camera Steering) ---

    def move_mouse(self, delta_x: float, delta_y: float) -> None:
        """Move the mouse cursor / in-game camera look relatively by (delta_x, delta_y) pixels."""
        if not self.enabled:
            return

        idx = int(round(delta_x))
        idy = int(round(delta_y))
        if idx == 0 and idy == 0:
            return

        try:
            if self._pynput_mouse:
                self._pynput_mouse.move(idx, idy)
            elif PYAUTOGUI_AVAILABLE:
                pyautogui.moveRel(idx, idy)
        except Exception as e:
            logger.error("Error moving mouse: %s", e)

    # --- OS Key Execution ---

    def _os_key_down(self, key: str) -> None:
        """Send physical key down event via backend."""
        if not self.enabled:
            return
        try:
            if self._pynput_kbd:
                self._pynput_kbd.press(key)
            elif PYAUTOGUI_AVAILABLE:
                pyautogui.keyDown(key)
        except Exception as e:
            logger.error("Error pressing key '%s': %s", key, e)

    def _os_key_up(self, key: str) -> None:
        """Send physical key up event via backend."""
        try:
            if self._pynput_kbd:
                self._pynput_kbd.release(key)
            elif PYAUTOGUI_AVAILABLE:
                pyautogui.keyUp(key)
        except Exception as e:
            logger.error("Error releasing key '%s': %s", key, e)

    # --- OS Mouse Execution ---

    def _os_mouse_down(self, button: str) -> None:
        """Send physical mouse down event via backend."""
        if not self.enabled:
            return
        try:
            if self._pynput_mouse:
                btn = PynputButton.left if button == "left" else PynputButton.right
                self._pynput_mouse.press(btn)
            elif PYAUTOGUI_AVAILABLE:
                pyautogui.mouseDown(button=button)
        except Exception as e:
            logger.error("Error pressing mouse '%s': %s", button, e)

    def _os_mouse_up(self, button: str) -> None:
        """Send physical mouse up event via backend."""
        try:
            if self._pynput_mouse:
                btn = PynputButton.left if button == "left" else PynputButton.right
                self._pynput_mouse.release(btn)
            elif PYAUTOGUI_AVAILABLE:
                pyautogui.mouseUp(button=button)
        except Exception as e:
            logger.error("Error releasing mouse '%s': %s", button, e)

    # --- High-Level State Management ---

    def press_key(self, key: str) -> None:
        """Press key only if not already pressed."""
        if key not in self.pressed_keys:
            self._os_key_down(key)
            self.pressed_keys.add(key)
            logger.debug("Key DOWN: %s", key)

    def release_key(self, key: str) -> None:
        """Release key only if currently pressed."""
        if key in self.pressed_keys:
            self._os_key_up(key)
            self.pressed_keys.remove(key)
            logger.debug("Key UP: %s", key)

    def press_mouse(self, button: str) -> None:
        """Press mouse button only if not already pressed."""
        if button not in self.pressed_mouse_buttons:
            self._os_mouse_down(button)
            self.pressed_mouse_buttons.add(button)
            logger.debug("Mouse DOWN: %s", button)

    def release_mouse(self, button: str) -> None:
        """Release mouse button only if currently pressed."""
        if button in self.pressed_mouse_buttons:
            self._os_mouse_up(button)
            self.pressed_mouse_buttons.remove(button)
            logger.debug("Mouse UP: %s", button)

    def sync_movement_keys(self, desired_keys: Set[str]) -> None:
        """Synchronize active movement keys, resolving directional conflicts (W+S, A+D)."""
        forward = self.config.inputs.key_forward
        backward = self.config.inputs.key_backward
        left = self.config.inputs.key_left
        right = self.config.inputs.key_right

        sanitized_desired = set(desired_keys)

        # Conflict check: Forward vs Backward
        if forward in sanitized_desired and backward in sanitized_desired:
            sanitized_desired.remove(backward)

        # Conflict check: Left vs Right
        if left in sanitized_desired and right in sanitized_desired:
            sanitized_desired.remove(right)

        movement_keys = {forward, backward, left, right}

        # Release keys that are pressed but no longer desired
        for k in list(self.pressed_keys):
            if k in movement_keys and k not in sanitized_desired:
                self.release_key(k)

        # Press keys that are desired but not yet pressed
        for k in sanitized_desired:
            if k in movement_keys and k not in self.pressed_keys:
                self.press_key(k)

    def sync_mouse_state(self, shoot: bool, aim: bool) -> None:
        """Synchronize mouse button states (LMB for shoot, RMB for aim)."""
        shoot_btn = self.config.inputs.mouse_shoot
        aim_btn = self.config.inputs.mouse_aim

        # Shoot (Left Mouse)
        if shoot and shoot_btn not in self.pressed_mouse_buttons:
            self.press_mouse(shoot_btn)
        elif not shoot and shoot_btn in self.pressed_mouse_buttons:
            self.release_mouse(shoot_btn)

        # Aim (Right Mouse)
        if aim and aim_btn not in self.pressed_mouse_buttons:
            self.press_mouse(aim_btn)
        elif not aim and aim_btn in self.pressed_mouse_buttons:
            self.release_mouse(aim_btn)

    def apply_control_state(self, shoot: bool, aim: bool, move_keys: Set[str], mouse_dx: float = 0.0, mouse_dy: float = 0.0) -> None:
        """Update both keyboard and mouse states from a single resolved control state."""
        self.sync_mouse_state(shoot=shoot, aim=aim)
        self.sync_movement_keys(desired_keys=move_keys)
        if mouse_dx != 0.0 or mouse_dy != 0.0:
            self.move_mouse(mouse_dx, mouse_dy)

    def release_all_keys(self) -> None:
        """Release every currently pressed key."""
        for key in list(self.pressed_keys):
            self.release_key(key)
        self.pressed_keys.clear()

    def release_all_mouse_buttons(self) -> None:
        """Release every currently pressed mouse button."""
        for btn in list(self.pressed_mouse_buttons):
            self.release_mouse(btn)
        self.pressed_mouse_buttons.clear()

    def release_all_inputs(self) -> None:
        """Emergency safe release of all keys and mouse buttons."""
        logger.info("Executing full release of all inputs.")
        self.release_all_keys()
        self.release_all_mouse_buttons()
