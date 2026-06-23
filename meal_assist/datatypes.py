# -*- coding: utf-8 -*-
"""Shared dataclasses for planning, IK targets, replay status, and LUT entries."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class PoseTarget:
    """One task-space IK target for a phase of the eating motion."""

    name: str
    pos: Tuple[float, float, float]
    forward: Tuple[float, float, float]
    normal: Tuple[float, float, float]
    normal_weight: float
    forward_weight: float
    normal_hard: bool = False
    forward_hard: bool = True
    joint6_pref: Optional[float] = None
    joint6_weight: float = 0.0
    joint6_max: Optional[float] = None
    joint6_hard: bool = False
    head_drop_min: Optional[float] = None
    head_drop_weight: float = 0.0
    head_drop_hard_min: Optional[float] = None


@dataclass
class FoodRegion:
    """Tray subregion used to generate feasible scoop primitives."""

    region_id: int
    name: str
    polygon_xy: List[Tuple[float, float]]
    barrier_x: float
    barrier_y_min: float
    barrier_y_max: float
    barrier_height: float
    barrier_thickness: float


@dataclass
class StepResult:
    """Structured status returned by replay and motion helper routines."""

    label: str
    ok: bool
    reason: str
    target_pos: Optional[Tuple[float, float, float]] = None
    actual_pos: Optional[Tuple[float, float, float]] = None
    pos_error: Optional[float] = None
    q_error: Optional[float] = None
    tilt_error: Optional[float] = None
    head_drop: Optional[float] = None
    contact: int = 0
    extra_frames: int = 0

    def summary(self) -> str:
        """Return a compact one-line status string for logs."""
        parts = [f"label={self.label}", f"ok={self.ok}", f"reason={self.reason}"]
        if self.pos_error is not None:
            parts.append(f"pos_err={self.pos_error * 1000:.1f}mm")
        if self.q_error is not None:
            parts.append(f"q_err={self.q_error:.4f}rad")
        if self.tilt_error is not None:
            parts.append(f"tilt={np.degrees(self.tilt_error):.2f}deg")
        if self.head_drop is not None:
            parts.append(f"head_drop={self.head_drop * 1000:.1f}mm")
        if self.contact:
            parts.append(f"contact={self.contact}")
        if self.extra_frames:
            parts.append(f"extra={self.extra_frames}")
        return "[STEP_RESULT] " + ", ".join(parts)


@dataclass
class ScoopPrimitive:
    """Stored feasible scoop motion candidate in the LUT."""

    primitive_id: str
    region_id: int
    food_xy: Tuple[float, float]
    drag_length: float
    score: float

    pre_scoop_pos: Tuple[float, float, float]
    engage_pos: Tuple[float, float, float]
    drag_start_pos: Tuple[float, float, float]
    drag_end_pos: Tuple[float, float, float]
    lift_pos: Tuple[float, float, float]

    q_pre: List[float]
    q_engage: List[float]
    q_drag_start: List[float]
    q_drag_end: List[float]
    q_lift: List[float]

    max_pos_error: float
    max_tilt_error: float
    max_forward_error: float
    min_sigma: float
    max_condition: float
    max_contact: int
    preview_max_tilt: float = 0.0
    head_drop_pre: float = 0.0
    head_drop_engage: float = 0.0
    head_drop_drag_start: float = 0.0
    head_drop_drag_end: float = 0.0
    min_head_drop: float = 0.0
    max_head_drop_error: float = 0.0


@dataclass
class MouthConnector:
    """Cached neutral/pre-approach/mouth delivery connector."""

    connector_id: str
    mouth_pos: Tuple[float, float, float]
    pre_pos: Tuple[float, float, float]
    retreat_pos: Tuple[float, float, float]
    q_pre: List[float]
    q_delivery: List[float]
    q_retreat: List[float]
    pos_error: float
    tilt_error: float
    forward_error: float
    min_joint_margin_ratio: float
    attempts: int


@dataclass
class NeutralConnector:
    """Cached neutral pose selected from the neutral search region."""

    q_neutral: List[float]
    neutral_pos: Tuple[float, float, float]
    pos_error: float
    tilt_error: float
    head_drop: float
