# -*- coding: utf-8 -*-
"""Small geometry and interpolation utilities."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    """Return a unit vector; return the original vector if it is near zero."""
    n = float(np.linalg.norm(v))
    return v if n < 1e-12 else v / n


def point_in_polygon_xy(point: np.ndarray, polygon: List[Tuple[float, float]]) -> bool:
    """Return True if a 2D point is inside a polygon using ray casting."""
    x, y = float(point[0]), float(point[1])
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1):
            inside = not inside
    return inside


def sample_points_in_polygon(polygon: List[Tuple[float, float]], n_axis: int) -> List[np.ndarray]:
    """Sample a regular XY grid and keep points inside the polygon."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    grid_x = np.linspace(min(xs), max(xs), n_axis)
    grid_y = np.linspace(min(ys), max(ys), n_axis)
    pts: List[np.ndarray] = []
    for x in grid_x:
        for y in grid_y:
            p = np.array([x, y], dtype=float)
            if point_in_polygon_xy(p, polygon):
                pts.append(p)
    return pts


def smoothstep5(t: float) -> float:
    """Quintic smoothstep with zero slope and acceleration at endpoints."""
    t = float(np.clip(t, 0.0, 1.0))
    return 6.0 * t**5 - 15.0 * t**4 + 10.0 * t**3


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi
