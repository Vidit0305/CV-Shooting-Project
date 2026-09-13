"""Minimalist Cyberpunk OpenCV HUD and Visual Feedback Renderer.

Renders an ultra-clean, floating, non-intrusive gaming overlay with:
- Zero bulky opaque side panels (95%+ unobstructed camera feed)
- Top floating status pill ([ACTIVE] / [PAUSED]) + FPS
- Dynamic reactive crosshair (expands on Shoot, brackets on ADS Aim)
- Sleek floating key chips (W/A/S/D, LMB, RMB) that glow only when pressed
- Full-screen support with deep-black letterboxing (zero white bars)
"""

from __future__ import annotations

from typing import Optional, Set, Tuple

import cv2
import numpy as np

from calibration import Calibrator
from config import AppConfig
from gesture_controller import ControlState, Gesture
from hand_tracker import HandData


class UIRenderer:
    """Renders modern, sleek, minimalist gaming graphics directly on frames."""

    def __init__(self, config: AppConfig):
        self.config = config

        # Sleek Neon Cyber Palette (BGR)
        self.COLOR_BLACK = (10, 10, 14)
        self.COLOR_PANEL = (22, 22, 30)
        self.COLOR_GREEN = (0, 255, 128)        # Neon Mint Green
        self.COLOR_CYAN = (255, 220, 0)         # Bright Cyan
        self.COLOR_RED = (60, 60, 255)          # Laser Red (Shoot)
        self.COLOR_BLUE = (255, 140, 0)         # Electric Blue (Aim/ADS)
        self.COLOR_AMBER = (0, 165, 255)        # Amber Warning
        self.COLOR_WHITE = (245, 245, 245)
        self.COLOR_MUTED = (120, 120, 130)

        # Reticle animation state
        self._reticle_spread: float = 14.0

    def render(
        self,
        frame: np.ndarray,
        hand_data: HandData,
        control_state: ControlState,
        controls_enabled: bool,
        fps: float,
        calibrator: Optional[Calibrator] = None,
        debug_mode: bool = False,
    ) -> np.ndarray:
        """Render floating minimalist HUD overlay."""
        h, w = frame.shape[:2]

        # 1. Top Floating Status Pill (Minimalist capsule)
        self._draw_top_pill(frame, controls_enabled, fps)

        # 2. Dynamic Center Crosshair / Aim Reticle
        if self.config.ui.show_crosshair:
            self._draw_dynamic_crosshair(frame, control_state, hand_data)

        # 3. Compact Key Chips (Bottom Right)
        self._draw_compact_keys(frame, control_state, controls_enabled)

        # 4. Calibration Overlay (if active)
        if calibrator and calibrator.is_active():
            self._draw_calibration_card(frame, calibrator)

        # 5. Optional Minimal Debug Bar (Bottom Left)
        if debug_mode or self.config.ui.show_debug:
            self._draw_debug_metrics(frame, hand_data, control_state)

        return frame

    def _draw_capsule(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        bg_color: Tuple[int, int, int] = (20, 20, 28),
        border_color: Tuple[int, int, int] = (60, 60, 80),
        alpha: float = 0.70,
    ) -> None:
        """Draw a sleek, semi-transparent rounded pill/capsule."""
        sub = frame[y : y + h, x : x + w]
        if sub.shape[0] != h or sub.shape[1] != w:
            return
        rect = np.full(sub.shape, bg_color, dtype=np.uint8)
        cv2.addWeighted(rect, alpha, sub, 1.0 - alpha, 0, sub)
        frame[y : y + h, x : x + w] = sub
        cv2.rectangle(frame, (x, y), (x + w, y + h), border_color, 1, cv2.LINE_AA)

    def _draw_top_pill(self, frame: np.ndarray, controls_enabled: bool, fps: float) -> None:
        """Draw a floating status pill at top center."""
        h, w = frame.shape[:2]
        pill_w, pill_h = 320, 34
        x = (w - pill_w) // 2
        y = 14

        border_col = self.COLOR_GREEN if controls_enabled else (70, 70, 90)
        self._draw_capsule(frame, x, y, pill_w, pill_h, alpha=0.75, border_color=border_col)

        # Status Dot
        dot_color = self.COLOR_GREEN if controls_enabled else self.COLOR_AMBER
        cv2.circle(frame, (x + 20, y + 17), 5, dot_color, -1, cv2.LINE_AA)

        # Controls Status Text
        status_text = "CONTROLS: ACTIVE" if controls_enabled else "CONTROLS: PAUSED (Press C)"
        cv2.putText(
            frame,
            status_text,
            (x + 34, y + 22),
            cv2.FONT_HERSHEY_DUPLEX,
            0.42,
            self.COLOR_WHITE if controls_enabled else self.COLOR_AMBER,
            1,
            cv2.LINE_AA,
        )

        # FPS indicator
        fps_text = f"{int(round(fps))} FPS"
        cv2.putText(
            frame,
            fps_text,
            (x + pill_w - 65, y + 22),
            cv2.FONT_HERSHEY_DUPLEX,
            0.40,
            self.COLOR_MUTED,
            1,
            cv2.LINE_AA,
        )

    def _draw_dynamic_crosshair(
        self,
        frame: np.ndarray,
        control_state: ControlState,
        hand_data: HandData,
    ) -> None:
        """Draw dynamic in-game crosshair that reacts to Shoot, Aim, and Mouse Look."""
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2

        # Reticle color & spread animation
        if control_state.shoot:
            reticle_color = self.COLOR_RED
            target_spread = 26.0
        elif control_state.aim:
            reticle_color = self.COLOR_BLUE
            target_spread = 8.0
        else:
            reticle_color = self.COLOR_CYAN
            target_spread = 14.0

        # Smooth spread expansion
        self._reticle_spread += 0.40 * (target_spread - self._reticle_spread)
        spread = int(round(self._reticle_spread))
        tick_len = 8

        # Center dot
        cv2.circle(frame, (cx, cy), 2, reticle_color, -1, cv2.LINE_AA)

        # 4 Reticle Ticks (Top, Bottom, Left, Right)
        cv2.line(frame, (cx, cy - spread - tick_len), (cx, cy - spread), reticle_color, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy + spread), (cx, cy + spread + tick_len), reticle_color, 1, cv2.LINE_AA)
        cv2.line(frame, (cx - spread - tick_len, cy), (cx - spread, cy), reticle_color, 1, cv2.LINE_AA)
        cv2.line(frame, (cx + spread, cy), (cx + spread + tick_len, cy), reticle_color, 1, cv2.LINE_AA)

        # Aim (ADS) brackets
        if control_state.aim:
            b_size = 18
            b_offset = spread + 8
            # Corner brackets
            cv2.line(frame, (cx - b_offset, cy - b_size), (cx - b_offset, cy + b_size), self.COLOR_BLUE, 2, cv2.LINE_AA)
            cv2.line(frame, (cx + b_offset, cy - b_size), (cx + b_offset, cy + b_size), self.COLOR_BLUE, 2, cv2.LINE_AA)
            cv2.putText(frame, "ADS", (cx - 16, cy + b_offset + 18), cv2.FONT_HERSHEY_DUPLEX, 0.42, self.COLOR_BLUE, 1, cv2.LINE_AA)

        # Shoot banner text
        elif control_state.shoot:
            cv2.putText(frame, "FIRE", (cx - 20, cy + spread + 22), cv2.FONT_HERSHEY_DUPLEX, 0.48, self.COLOR_RED, 1, cv2.LINE_AA)

        # Directional WASD indicator arrow from center
        if control_state.move_keys:
            joined = "+".join(sorted(list(control_state.move_keys))).upper()
            cv2.putText(frame, joined, (cx - 15, cy - spread - 14), cv2.FONT_HERSHEY_DUPLEX, 0.45, self.COLOR_GREEN, 1, cv2.LINE_AA)

    def _draw_compact_keys(
        self,
        frame: np.ndarray,
        control_state: ControlState,
        controls_enabled: bool,
    ) -> None:
        """Render floating minimalist key chips in the bottom right corner."""
        h, w = frame.shape[:2]
        chip_size = 28
        spacing = 6
        base_x = w - 190
        base_y = h - 75

        # WASD Layout
        w_rect = (base_x + chip_size + spacing, base_y - chip_size - spacing, chip_size, chip_size)
        a_rect = (base_x, base_y, chip_size, chip_size)
        s_rect = (base_x + chip_size + spacing, base_y, chip_size, chip_size)
        d_rect = (base_x + (chip_size + spacing) * 2, base_y, chip_size, chip_size)

        keys_map = [
            ("W", w_rect, "w" in control_state.move_keys),
            ("A", a_rect, "a" in control_state.move_keys),
            ("S", s_rect, "s" in control_state.move_keys),
            ("D", d_rect, "d" in control_state.move_keys),
        ]

        for label, (rx, ry, rw, rh), is_down in keys_map:
            bg_col = self.COLOR_GREEN if is_down else (25, 25, 35)
            txt_col = (10, 10, 14) if is_down else self.COLOR_MUTED
            self._draw_capsule(frame, rx, ry, rw, rh, bg_color=bg_col, alpha=0.85 if is_down else 0.50)
            cv2.putText(frame, label, (rx + 8, ry + 19), cv2.FONT_HERSHEY_DUPLEX, 0.40, txt_col, 1, cv2.LINE_AA)

        # Mouse Chips (LMB / RMB)
        mouse_x = base_x + (chip_size + spacing) * 3 + 12
        lmb_rect = (mouse_x, base_y - chip_size - spacing, 46, chip_size)
        rmb_rect = (mouse_x, base_y, 46, chip_size)

        # LMB (Shoot)
        lmb_down = control_state.shoot
        lmb_bg = self.COLOR_RED if lmb_down else (25, 25, 35)
        lmb_txt = self.COLOR_WHITE if lmb_down else self.COLOR_MUTED
        self._draw_capsule(frame, lmb_rect[0], lmb_rect[1], lmb_rect[2], lmb_rect[3], bg_color=lmb_bg, alpha=0.9 if lmb_down else 0.5)
        cv2.putText(frame, "LMB", (mouse_x + 8, lmb_rect[1] + 19), cv2.FONT_HERSHEY_DUPLEX, 0.38, lmb_txt, 1, cv2.LINE_AA)

        # RMB (Aim)
        rmb_down = control_state.aim
        rmb_bg = self.COLOR_BLUE if rmb_down else (25, 25, 35)
        rmb_txt = self.COLOR_WHITE if rmb_down else self.COLOR_MUTED
        self._draw_capsule(frame, rmb_rect[0], rmb_rect[1], rmb_rect[2], rmb_rect[3], bg_color=rmb_bg, alpha=0.9 if rmb_down else 0.5)
        cv2.putText(frame, "RMB", (mouse_x + 8, rmb_rect[1] + 19), cv2.FONT_HERSHEY_DUPLEX, 0.38, rmb_txt, 1, cv2.LINE_AA)

    def _draw_debug_metrics(
        self,
        frame: np.ndarray,
        hand_data: HandData,
        control_state: ControlState,
    ) -> None:
        """Render clean, compact telemetry bar at bottom left."""
        h = frame.shape[0]
        y = h - 22
        x = 18

        hand_str = f"Hand: {hand_data.handedness}" if hand_data.detected else "Hand: None"
        pinch_str = f"Pinch: {control_state.pinch_ratio:.2f}"
        mouse_str = f"MouseDelta: ({int(control_state.mouse_dx)}, {int(control_state.mouse_dy)})"

        debug_line = f"[D] {hand_str}  |  {pinch_str}  |  {mouse_str}  |  [F] Fullscreen  |  [Q] Exit"
        cv2.putText(frame, debug_line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.COLOR_MUTED, 1, cv2.LINE_AA)

    def _draw_calibration_card(self, frame: np.ndarray, calibrator: Calibrator) -> None:
        """Render calibration card centered without clunky background."""
        h, w = frame.shape[:2]
        cw, ch = 480, 100
        cx = (w - cw) // 2
        cy = (h - ch) // 2

        self._draw_capsule(frame, cx, cy, cw, ch, bg_color=(20, 20, 30), border_color=self.COLOR_AMBER, alpha=0.90)

        cv2.putText(frame, "HAND CALIBRATION", (cx + 20, cy + 28), cv2.FONT_HERSHEY_DUPLEX, 0.55, self.COLOR_AMBER, 1, cv2.LINE_AA)
        cv2.putText(frame, calibrator.status_message, (cx + 20, cy + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.44, self.COLOR_WHITE, 1, cv2.LINE_AA)

        # Progress bar
        bx, by, bw, bh = cx + 20, cy + 70, cw - 40, 10
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (40, 40, 50), -1)
        fw = int(bw * np.clip(calibrator.progress_pct, 0.0, 1.0))
        cv2.rectangle(frame, (bx, by), (bx + fw, by + bh), self.COLOR_GREEN, -1)
