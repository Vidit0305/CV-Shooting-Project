"""Utility functions, geometric math, landmark constants, and FPS tracking.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple


# MediaPipe Hand 21 Landmark Indices
class LandmarkIndex:
    WRIST = 0
    
    # Thumb
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    
    # Index Finger
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    
    # Middle Finger
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    
    # Ring Finger
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    
    # Pinky Finger
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


@dataclass
class Point3D:
    x: float
    y: float
    z: float = 0.0

    @classmethod
    def from_landmark(cls, lm: Any) -> Point3D:
        return cls(x=float(lm.x), y=float(lm.y), z=float(getattr(lm, "z", 0.0)))


def euclidean_distance_2d(p1: Point3D | Sequence[float], p2: Point3D | Sequence[float]) -> float:
    """Calculate 2D Euclidean distance between two points."""
    x1, y1 = (p1.x, p1.y) if isinstance(p1, Point3D) else (p1[0], p1[1])
    x2, y2 = (p2.x, p2.y) if isinstance(p2, Point3D) else (p2[0], p2[1])
    return math.hypot(x2 - x1, y2 - y1)


def euclidean_distance_3d(p1: Point3D | Sequence[float], p2: Point3D | Sequence[float]) -> float:
    """Calculate 3D Euclidean distance between two points."""
    if isinstance(p1, Point3D) and isinstance(p2, Point3D):
        return math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2 + (p2.z - p1.z) ** 2)
    x1, y1, z1 = (p1[0], p1[1], p1[2] if len(p1) > 2 else 0.0)
    x2, y2, z2 = (p2[0], p2[1], p2[2] if len(p2) > 2 else 0.0)
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


def calculate_angle_3points(
    a: Point3D | Sequence[float],
    b: Point3D | Sequence[float],
    c: Point3D | Sequence[float],
) -> float:
    """Calculate the interior angle (in degrees) at joint vertex b between a and c."""
    ax, ay = (a.x, a.y) if isinstance(a, Point3D) else (a[0], a[1])
    bx, by = (b.x, b.y) if isinstance(b, Point3D) else (b[0], b[1])
    cx, cy = (c.x, c.y) if isinstance(c, Point3D) else (c[0], c[1])

    v1 = (ax - bx, ay - by)
    v2 = (cx - bx, cy - by)

    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.hypot(v1[0], v1[1])
    mag2 = math.hypot(v2[0], v2[1])

    if mag1 < 1e-6 or mag2 < 1e-6:
        return 180.0

    cosine = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cosine))


def calculate_palm_scale(landmarks: List[Point3D]) -> float:
    """Calculate normalized palm reference scale from hand landmarks.
    
    Uses distance between Wrist (0) and Middle MCP (9) combined with
    Index MCP (5) to Pinky MCP (17) width for high robustness to hand rotation.
    """
    if len(landmarks) < 21:
        return 0.20

    wrist = landmarks[LandmarkIndex.WRIST]
    middle_mcp = landmarks[LandmarkIndex.MIDDLE_MCP]
    index_mcp = landmarks[LandmarkIndex.INDEX_MCP]
    pinky_mcp = landmarks[LandmarkIndex.PINKY_MCP]

    palm_length = euclidean_distance_2d(wrist, middle_mcp)
    palm_width = euclidean_distance_2d(index_mcp, pinky_mcp)

    scale = (palm_length + palm_width) / 2.0
    return max(0.05, scale)


class FPSCounter:
    """Smooth real-time FPS counter using exponential moving average."""

    def __init__(self, smoothing_factor: float = 0.90):
        self.smoothing = smoothing_factor
        self.fps: float = 0.0
        self._last_time: float = time.perf_counter()
        self._frame_count: int = 0

    def update(self) -> float:
        current_time = time.perf_counter()
        delta = current_time - self._last_time
        self._last_time = current_time
        self._frame_count += 1

        if delta > 0.0:
            current_fps = 1.0 / delta
            if self.fps == 0.0:
                self.fps = current_fps
            else:
                self.fps = self.smoothing * self.fps + (1.0 - self.smoothing) * current_fps

        return self.fps
