"""Configuration system for the Computer Vision Gesture Controller.

Provides strongly typed AppConfig dataclass with defaults, JSON persistence,
and runtime updates.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    camera_index: int = 0
    target_width: int = 1280
    target_height: int = 720
    fallback_width: int = 640
    fallback_height: int = 480
    fourcc: str = "MJPG"
    mirror_preview: bool = True
    fps: int = 30


@dataclass
class TrackingConfig:
    min_detection_confidence: float = 0.70
    min_tracking_confidence: float = 0.60
    max_num_hands: int = 1
    model_complexity: int = 1
    # Landmark position smoothing (0.0 = no smoothing, 0.9 = heavy smoothing)
    smoothing_factor: float = 0.50


@dataclass
class MouseLookConfig:
    """Settings for moving the screen / in-game camera look via hand motion."""
    enabled: bool = True
    # Sensitivity multipliers: scale normalized hand delta to screen mouse pixels
    sensitivity_x: float = 1400.0
    sensitivity_y: float = 1100.0
    # Minimum normalized displacement delta to ignore hand tremor / micro-jitter
    deadzone: float = 0.002
    # Exponential smoothing factor for mouse velocity (0.0 = raw, 0.8 = ultra smooth)
    smoothing: float = 0.60
    # Sensitivity factor when Aiming Down Sights (FIST pose) for precision shooting
    ads_multiplier: float = 0.50


@dataclass
class GestureConfig:
    # Pinch thresholds (normalized distance: thumb_tip to index_tip / palm_scale)
    # Hysteresis prevents flickering
    pinch_on_ratio: float = 0.35
    pinch_off_ratio: float = 0.48

    # Fist detection: all 4 fingers folded (tip distance from wrist / pip distance from wrist)
    fist_fold_ratio: float = 0.85

    # WASD Movement: displacement from neutral anchor center [0.0 - 1.0 normalized]
    movement_deadzone: float = 0.14
    tilt_threshold_deg: float = 25.0

    # Minimum finger extension ratio (tip_dist / pip_dist from wrist)
    finger_extension_ratio: float = 1.15
    thumb_extension_ratio: float = 1.10


@dataclass
class StabilizationConfig:
    # Majority vote sliding window frame count
    voting_window_size: int = 7
    # Required consecutive matching frames for gesture state change
    consecutive_activation_frames: int = 3
    # Milliseconds grace period before releasing held inputs on tracking loss
    no_hand_grace_period_ms: int = 250


@dataclass
class InputConfig:
    # Key mapping
    key_forward: str = "w"
    key_backward: str = "s"
    key_left: str = "a"
    key_right: str = "d"
    mouse_shoot: str = "left"
    mouse_aim: str = "right"
    # Backend: "auto", "pynput", or "pyautogui"
    input_backend: str = "auto"
    # Safety: start with controls disabled
    controls_enabled_by_default: bool = False
    mouse_look: MouseLookConfig = field(default_factory=MouseLookConfig)


@dataclass
class UIConfig:
    window_name: str = "Gesture Shooting Controller"
    fullscreen: bool = False
    show_hud: bool = True
    show_crosshair: bool = True
    show_landmarks: bool = True
    show_debug: bool = False
    fps_smoothing_factor: float = 0.90
    theme_color_primary: tuple[int, int, int] = (0, 255, 128)     # Neon Mint Green
    theme_color_cyan: tuple[int, int, int] = (255, 220, 0)        # Electric Cyan
    theme_color_danger: tuple[int, int, int] = (60, 60, 255)      # Laser Red (Shoot)
    theme_color_aim: tuple[int, int, int] = (255, 140, 0)         # Blue (Aim/ADS)
    theme_color_warning: tuple[int, int, int] = (0, 165, 255)     # Amber Orange


@dataclass
class KeybindConfig:
    key_toggle_controls: str = "c"
    key_fullscreen: str = "f"
    key_toggle_hud: str = "h"
    key_calibrate: str = "r"
    key_toggle_debug: str = "d"
    key_emergency_stop: str = "q"
    key_escape_code: int = 27  # ASCII ESC


@dataclass
class CalibrationData:
    neutral_center_x: float = 0.50
    neutral_center_y: float = 0.55
    calibrated_palm_scale: float = 1.0
    calibrated_pinch_on: float = 0.35
    calibrated_pinch_off: float = 0.48
    is_calibrated: bool = False


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    gestures: GestureConfig = field(default_factory=GestureConfig)
    stabilization: StabilizationConfig = field(default_factory=StabilizationConfig)
    inputs: InputConfig = field(default_factory=InputConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    keybinds: KeybindConfig = field(default_factory=KeybindConfig)
    calibration: CalibrationData = field(default_factory=CalibrationData)

    @classmethod
    def load(cls, config_path: str = "config.json") -> AppConfig:
        """Load configuration from JSON file or return defaults if not found."""
        if not os.path.exists(config_path):
            logger.info("Config file '%s' not found. Using defaults.", config_path)
            config = cls()
            config.save(config_path)
            return config

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data: Dict[str, Any] = json.load(f)

            # Reconstruct nested dataclasses
            inputs_data = data.get("inputs", {})
            mouse_look_data = inputs_data.pop("mouse_look", {})
            input_cfg = InputConfig(**inputs_data, mouse_look=MouseLookConfig(**mouse_look_data))

            config = cls(
                camera=CameraConfig(**data.get("camera", {})),
                tracking=TrackingConfig(**data.get("tracking", {})),
                gestures=GestureConfig(**data.get("gestures", {})),
                stabilization=StabilizationConfig(**data.get("stabilization", {})),
                inputs=input_cfg,
                ui=UIConfig(**data.get("ui", {})),
                keybinds=KeybindConfig(**data.get("keybinds", {})),
                calibration=CalibrationData(**data.get("calibration", {})),
            )
            logger.info("Loaded configuration from '%s'", config_path)
            return config
        except Exception as e:
            logger.error("Error reading config file '%s': %s. Using default settings.", config_path, e)
            return cls()

    def save(self, config_path: str = "config.json") -> None:
        """Serialize configuration to a JSON file."""
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(asdict(self), f, indent=4)
            logger.info("Configuration successfully saved to '%s'", config_path)
        except Exception as e:
            logger.error("Failed to save config to '%s': %s", config_path, e)
