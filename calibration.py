"""Calibration module for personalized hand geometry and neutral center baseline.

Captures open-hand and pinched-hand reference samples, computes optimal
hysteresis thresholds, calibrates deadzone anchor, and persists to config.json.
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from config import AppConfig
from hand_tracker import HandData
from utils import LandmarkIndex, euclidean_distance_3d

logger = logging.getLogger(__name__)


class CalibrationStep(enum.Enum):
    IDLE = "IDLE"
    INSTRUCT_OPEN = "INSTRUCT_OPEN"
    CAPTURING_OPEN = "CAPTURING_OPEN"
    INSTRUCT_PINCH = "INSTRUCT_PINCH"
    CAPTURING_PINCH = "CAPTURING_PINCH"
    COMPLETED = "COMPLETED"


@dataclass
class CalibrationSample:
    palm_scale: float
    palm_center: Tuple[float, float]
    thumb_index_ratio: float


class Calibrator:
    """Interactive multi-step calibrator."""

    def __init__(self, config: AppConfig, samples_per_step: int = 45):
        self.config = config
        self.samples_per_step = samples_per_step
        self.step: CalibrationStep = CalibrationStep.IDLE

        self._open_samples: List[CalibrationSample] = []
        self._pinch_samples: List[CalibrationSample] = []
        self._step_start_time: float = 0.0
        self._instruction_duration: float = 2.0  # seconds to read prompt before capturing

        self.status_message: str = ""
        self.progress_pct: float = 0.0

    def start(self) -> None:
        """Begin calibration wizard."""
        logger.info("Starting hand calibration wizard...")
        self.step = CalibrationStep.INSTRUCT_OPEN
        self._step_start_time = time.perf_counter()
        self._open_samples.clear()
        self._pinch_samples.clear()
        self.status_message = "Keep hand OPEN and RELAXED in center of view..."
        self.progress_pct = 0.0

    def cancel(self) -> None:
        """Cancel calibration and revert to IDLE."""
        self.step = CalibrationStep.IDLE
        self.status_message = "Calibration cancelled."
        self.progress_pct = 0.0

    def is_active(self) -> bool:
        return self.step != CalibrationStep.IDLE and self.step != CalibrationStep.COMPLETED

    def update(self, hand_data: HandData) -> None:
        """Update calibration state with current frame's hand data."""
        if not self.is_active():
            return

        now = time.perf_counter()
        elapsed = now - self._step_start_time

        # Step 1: Instruction for Open Hand
        if self.step == CalibrationStep.INSTRUCT_OPEN:
            self.status_message = "Step 1/2: Hold hand OPEN in center (Starting soon...)"
            self.progress_pct = min(1.0, elapsed / self._instruction_duration)
            if elapsed >= self._instruction_duration:
                self.step = CalibrationStep.CAPTURING_OPEN
                self._open_samples.clear()
                self.status_message = "Capturing OPEN hand samples... Hold still."

        # Step 2: Capturing Open Hand
        elif self.step == CalibrationStep.CAPTURING_OPEN:
            if hand_data.detected and len(hand_data.landmarks_norm) >= 21:
                sample = self._extract_sample(hand_data)
                self._open_samples.append(sample)
                self.progress_pct = len(self._open_samples) / float(self.samples_per_step)

            if len(self._open_samples) >= self.samples_per_step:
                self.step = CalibrationStep.INSTRUCT_PINCH
                self._step_start_time = now
                self.status_message = "Step 2/2: Now PINCH thumb and index together firmly..."
                self.progress_pct = 0.0

        # Step 3: Instruction for Pinch
        elif self.step == CalibrationStep.INSTRUCT_PINCH:
            self.status_message = "Step 2/2: Prepare to PINCH thumb and index finger..."
            self.progress_pct = min(1.0, elapsed / self._instruction_duration)
            if elapsed >= self._instruction_duration:
                self.step = CalibrationStep.CAPTURING_PINCH
                self._pinch_samples.clear()
                self.status_message = "Capturing PINCH samples... Hold pinch."

        # Step 4: Capturing Pinch
        elif self.step == CalibrationStep.CAPTURING_PINCH:
            if hand_data.detected and len(hand_data.landmarks_norm) >= 21:
                sample = self._extract_sample(hand_data)
                self._pinch_samples.append(sample)
                self.progress_pct = len(self._pinch_samples) / float(self.samples_per_step)

            if len(self._pinch_samples) >= self.samples_per_step:
                self._compute_and_save()
                self.step = CalibrationStep.COMPLETED
                self.status_message = "Calibration Complete! Thresholds updated."
                self.progress_pct = 1.0

    def _extract_sample(self, hand_data: HandData) -> CalibrationSample:
        """Extract measurement metrics from hand landmarks."""
        lms = hand_data.landmarks_norm
        scale = hand_data.palm_scale
        thumb_tip = lms[LandmarkIndex.THUMB_TIP]
        index_tip = lms[LandmarkIndex.INDEX_TIP]
        ratio = euclidean_distance_3d(thumb_tip, index_tip) / max(0.01, scale)

        return CalibrationSample(
            palm_scale=scale,
            palm_center=hand_data.palm_center_norm,
            thumb_index_ratio=ratio,
        )

    def _compute_and_save(self) -> None:
        """Calculate optimal thresholds from gathered samples and save to config."""
        if not self._open_samples or not self._pinch_samples:
            logger.warning("Insufficient calibration samples.")
            return

        avg_open_ratio = float(np.mean([s.thumb_index_ratio for s in self._open_samples]))
        avg_pinch_ratio = float(np.mean([s.thumb_index_ratio for s in self._pinch_samples]))
        avg_scale = float(np.mean([s.palm_scale for s in self._open_samples]))

        avg_center_x = float(np.mean([s.palm_center[0] for s in self._open_samples]))
        avg_center_y = float(np.mean([s.palm_center[1] for s in self._open_samples]))

        logger.info(
            "Calibration Results -> Open Ratio: %.3f, Pinch Ratio: %.3f, Scale: %.3f, Center: (%.2f, %.2f)",
            avg_open_ratio, avg_pinch_ratio, avg_scale, avg_center_x, avg_center_y
        )

        # Spread: distance between open ratio and pinched ratio
        spread = max(0.15, avg_open_ratio - avg_pinch_ratio)
        calc_on = avg_pinch_ratio + 0.35 * spread
        calc_off = avg_pinch_ratio + 0.65 * spread

        # Clamping to sane bounds
        calc_on = float(np.clip(calc_on, 0.20, 0.50))
        calc_off = float(np.clip(calc_off, calc_on + 0.08, 0.70))

        # Update active config
        calib = self.config.calibration
        calib.neutral_center_x = avg_center_x
        calib.neutral_center_y = avg_center_y
        calib.calibrated_palm_scale = avg_scale
        calib.calibrated_pinch_on = calc_on
        calib.calibrated_pinch_off = calc_off
        calib.is_calibrated = True

        self.config.save()
        logger.info("Saved calibrated settings: ON=%.2f, OFF=%.2f", calc_on, calc_off)
