# -*- coding: utf-8 -*-
"""Minimal core API for the meal-assist trajectory-optimization tutorials.

This module intentionally keeps only the pieces needed by the tutorial and
trajectory-optimization workflow:

- configuration values
- shared dataclasses
- small math/geometry utilities
- MuJoCo robot FK/Jacobian helpers
- JSON/CSV primitive and connector storage

Older LUT builders, replay runners, online IK search, and viewer utilities were
removed from the main package surface. Keep them in historical scripts if they
are needed for reference, but new work should build from this compact core and
the ``tutorial_drag_opt_*`` files.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import mujoco
    import mujoco.viewer
except Exception as exc:  # pragma: no cover
    mujoco = None
    print("[WARN] Failed to import MuJoCo. Install it with `pip install mujoco`.", exc)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SystemConfig:
    """Runtime and planning parameters shared by the compact tutorial pipeline."""

    base_dir: Path = Path(__file__).resolve().parent.parent
    xml_name: str = "robot_model_v5_scene.xml"
    out_dir_name: str = "scoop_lut_output_v11"

    # Tray frame and scene geometry. These values match the current XML tray.
    tray_x_length: float = 0.21
    tray_y_length: float = 0.28
    base_x_offset: float = 0.075
    base_y_offset: float = 0.13
    base_z_offset: float = 0.0
    tray_surface_z: float = 0.04

    # Neutral and mouth anchors retained for loading existing connector caches.
    neutral_radius: float = 0.08
    neutral_z_min: float = 0.22
    neutral_z_max: float = 0.28
    neutral_grid_xy: int = 7
    neutral_grid_z: int = 3
    neutral_center_tray: Tuple[float, float] = (0.075, -0.02)
    neutral_pos_tray: Tuple[float, float, float] = (0.105, 0.046, 0.32)
    neutral_position_tol: float = 0.025
    neutral_normal_weight: float = 0.20
    neutral_head_drop_min: float = 0.0
    neutral_head_drop_weight: float = 0.0
    neutral_tilt_score_weight: float = 1.0
    neutral_joint_margin_ratio: float = 0.015
    neutral_margin_score_weight: float = 3.0

    default_mouth_pos_world: Tuple[float, float, float] = (0.0, 0.26, 0.45)
    mouth_y_range: Tuple[float, float] = (0.10, 0.25)
    mouth_candidate_z_range: Tuple[float, float] = (0.30, 0.60)
    mouth_forward_world: Tuple[float, float, float] = (-1.0, 0.0, 0.0)

    # Scoop primitive generation parameters retained for interpreting LUTs.
    scoop_drag_direction_world: Tuple[float, float, float] = (-1.0, 0.0, 0.0)
    drag_lengths: Tuple[float, ...] = (0.035, 0.050, 0.070)
    pre_scoop_height: float = 0.075
    engage_height: float = 0.018
    lift_height: float = 0.110
    scoop_start_offsets_x: Tuple[float, ...] = (0.020, 0.035, 0.050)
    scoop_y_offsets: Tuple[float, ...] = (-0.015, 0.0, 0.015)
    food_samples_per_region_axis: int = 5

    # IK/validation thresholds used by tutorial costs and diagnostics.
    joint_limit_margin_ratio: float = 0.015
    contact_allowed: int = 0
    min_sigma: float = 0.004
    max_condition: float = 900.0
    min_forward_dot: float = 0.85
    max_pos_error: float = 0.018
    max_tilt_error: float = 0.10
    max_forward_error: float = 0.40

    # Spoon geometry and posture metrics.
    joint6_index: int = 5
    spoon_head_local: Tuple[float, float, float] = (0.014529, 0.104125, 0.036995)
    head_drop_hard_min_drag_start: float = 0.050
    head_drop_hard_min_drag_end: float = 0.015
    spoon_normal_local: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    spoon_forward_local: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    world_up: Tuple[float, float, float] = (0.0, 0.0, -1.0)

    # Region guards used when checking drag_start/drag_end validity.
    scoop_boundary_max_overshoot: float = 0.005
    spoon_outer_margin: float = 0.005

    # Tutorial/replay timing parameters kept for compatibility.
    frames_per_segment: int = 240
    random_seed: int = 13

    @property
    def xml_path(self) -> Path:
        return self.base_dir / self.xml_name

    @property
    def out_dir(self) -> Path:
        return self.base_dir / self.out_dir_name


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass
class PoseTarget:
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
    region_id: int
    name: str
    polygon_xy: List[Tuple[float, float]]
    barrier_x: float
    barrier_y_min: float
    barrier_y_max: float
    barrier_height: float
    barrier_thickness: float


@dataclass
class ScoopPrimitive:
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
    q_neutral: List[float]
    neutral_pos: Tuple[float, float, float]
    pos_error: float
    tilt_error: float
    head_drop: float


@dataclass
class StepResult:
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


# =============================================================================
# Math and region helpers
# =============================================================================


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n < 1e-12 else v / n


def smoothstep5(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return 6.0 * t**5 - 15.0 * t**4 + 10.0 * t**3


def wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def point_in_polygon_xy(point: np.ndarray, polygon: List[Tuple[float, float]]) -> bool:
    x, y = float(point[0]), float(point[1])
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        crosses = (y1 > y) != (y2 > y)
        if crosses and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
            inside = not inside
    return inside


def sample_points_in_polygon(polygon: List[Tuple[float, float]], n_axis: int) -> List[np.ndarray]:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    pts: List[np.ndarray] = []
    for x in np.linspace(min(xs), max(xs), n_axis):
        for y in np.linspace(min(ys), max(ys), n_axis):
            p = np.array([x, y], dtype=float)
            if point_in_polygon_xy(p, polygon):
                pts.append(p)
    return pts


class TrayGeometry:
    """Tray/world coordinate conversion plus XML-matching region definitions."""

    def __init__(self, cfg: SystemConfig):
        self.cfg = cfg
        self.p_world_tray_origin = np.array(
            [cfg.base_x_offset, cfg.base_y_offset, cfg.base_z_offset],
            dtype=float,
        )
        self.R_world_tray = np.eye(3)

    def tray_to_world(self, p_tray: np.ndarray) -> np.ndarray:
        return self.p_world_tray_origin + self.R_world_tray @ p_tray

    def world_to_tray(self, p_world: np.ndarray) -> np.ndarray:
        return self.R_world_tray.T @ (p_world - self.p_world_tray_origin)

    def default_regions(self) -> List[FoodRegion]:
        Lx = self.cfg.tray_x_length
        Ly = self.cfg.tray_y_length
        return [
            FoodRegion(1, "rice_or_main", [(0.00*Lx,0.00*Ly),(0.52*Lx,0.00*Ly),(0.52*Lx,0.48*Ly),(0.00*Lx,0.48*Ly)], 0.00*Lx, 0.00*Ly, 0.48*Ly, 0.025, 0.006),
            FoodRegion(2, "side_1",       [(0.52*Lx,0.00*Ly),(1.00*Lx,0.00*Ly),(1.00*Lx,0.32*Ly),(0.52*Lx,0.32*Ly)], 0.52*Lx, 0.00*Ly, 0.32*Ly, 0.025, 0.006),
            FoodRegion(3, "side_2",       [(0.52*Lx,0.32*Ly),(1.00*Lx,0.32*Ly),(1.00*Lx,0.64*Ly),(0.52*Lx,0.64*Ly)], 0.52*Lx, 0.32*Ly, 0.64*Ly, 0.025, 0.006),
            FoodRegion(4, "side_3",       [(0.52*Lx,0.64*Ly),(1.00*Lx,0.64*Ly),(1.00*Lx,1.00*Ly),(0.52*Lx,1.00*Ly)], 0.52*Lx, 0.64*Ly, 1.00*Ly, 0.025, 0.006),
            FoodRegion(5, "soup_or_extra",[(0.00*Lx,0.48*Ly),(0.52*Lx,0.48*Ly),(0.52*Lx,1.00*Ly),(0.00*Lx,1.00*Ly)], 0.00*Lx, 0.48*Ly, 1.00*Ly, 0.025, 0.006),
        ]

    def drag_endpoints_inside_region(self, primitive: ScoopPrimitive) -> Tuple[bool, Dict[str, object]]:
        """Check whether drag_start and drag_end lie inside the primitive region."""
        regions = {r.region_id: r for r in self.default_regions()}
        region = regions.get(int(primitive.region_id))
        if region is None:
            return False, {"reason": "unknown_region"}
        start_tray = self.world_to_tray(np.array(primitive.drag_start_pos, dtype=float))
        end_tray = self.world_to_tray(np.array(primitive.drag_end_pos, dtype=float))
        start_ok = point_in_polygon_xy(start_tray[:2], region.polygon_xy)
        end_ok = point_in_polygon_xy(end_tray[:2], region.polygon_xy)
        return bool(start_ok and end_ok), {
            "region_id": primitive.region_id,
            "start_tray": start_tray.tolist(),
            "end_tray": end_tray.tolist(),
            "start_ok": bool(start_ok),
            "end_ok": bool(end_ok),
        }


# =============================================================================
# Robot wrapper
# =============================================================================


class RobotModel:
    """Small MuJoCo wrapper for FK, Jacobians, limits, and spoon metrics."""

    def __init__(self, cfg: SystemConfig):
        if mujoco is None:
            raise RuntimeError("MuJoCo is required. Install it with `pip install mujoco`.")
        if not cfg.xml_path.exists():
            raise FileNotFoundError(f"MuJoCo XML file not found: {cfg.xml_path}")

        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(str(cfg.xml_path))
        self.data = mujoco.MjData(self.model)

        self.spoon_tip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "spoon_tip")
        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link7")
        if self.spoon_tip_id < 0:
            raise RuntimeError("MuJoCo site 'spoon_tip' was not found.")
        if self.ee_id < 0:
            raise RuntimeError("MuJoCo body 'link7' was not found.")

        self.q_center, self.q_half, self.q_lower, self.q_upper = self._joint_bounds()

    def _joint_bounds(self):
        nq = self.model.nq
        center = np.zeros(nq)
        half = np.ones(nq)
        lower = np.full(nq, -np.pi)
        upper = np.full(nq, np.pi)
        for j in range(self.model.njnt):
            qadr = int(self.model.jnt_qposadr[j])
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and self.model.jnt_limited[j]:
                qmin, qmax = self.model.jnt_range[j]
                lower[qadr] = qmin
                upper[qadr] = qmax
                center[qadr] = 0.5 * (qmin + qmax)
                half[qadr] = max(0.5 * (qmax - qmin), 1e-6)
        return center, half, lower, upper

    def enforce_joint_limits(self, d, model=None):
        if model is None:
            model = self.model
        for j in range(model.njnt):
            qadr = int(model.jnt_qposadr[j])
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and model.jnt_limited[j]:
                qmin, qmax = model.jnt_range[j]
                d.qpos[qadr] = np.clip(d.qpos[qadr], qmin, qmax)

    def set_q(self, d, q: np.ndarray, model=None):
        if model is None:
            model = self.model
        d.qpos[:model.nq] = q[:model.nq]
        d.qvel[:] = 0.0
        self.enforce_joint_limits(d, model)
        mujoco.mj_forward(model, d)

    def is_joint_limit_safe(self, q: np.ndarray) -> bool:
        return self.is_joint_limit_safe_with_margin(q, self.cfg.joint_limit_margin_ratio)

    def is_joint_limit_safe_with_margin(self, q: np.ndarray, margin_ratio: float) -> bool:
        for i in range(self.model.nq):
            lo, hi = self.q_lower[i], self.q_upper[i]
            margin = margin_ratio * max(hi - lo, 1e-6)
            if q[i] < lo + margin or q[i] > hi - margin:
                return False
        return True

    def min_joint_limit_margin_ratio(self, q: np.ndarray) -> float:
        min_margin = float("inf")
        for i in range(self.model.nq):
            lo, hi = self.q_lower[i], self.q_upper[i]
            span = max(hi - lo, 1e-6)
            min_margin = min(min_margin, float(min((q[i] - lo) / span, (hi - q[i]) / span)))
        return min_margin

    def current_body_axis_world(self, d, local_axis: np.ndarray) -> np.ndarray:
        R = d.body(self.ee_id).xmat.reshape(3, 3)
        return normalize(R @ normalize(local_axis))

    def local_point_world(self, d, local_point: np.ndarray) -> np.ndarray:
        R = d.body(self.ee_id).xmat.reshape(3, 3)
        p = d.body(self.ee_id).xpos.copy()
        return p + R @ np.array(local_point, dtype=float)

    def link7_origin_world(self, d) -> np.ndarray:
        return d.body(self.ee_id).xpos.copy()

    def spoon_head_world(self, d) -> np.ndarray:
        return self.local_point_world(d, np.array(self.cfg.spoon_head_local, dtype=float))

    def spoon_head_drop(self, d) -> float:
        return float(self.link7_origin_world(d)[2] - self.spoon_head_world(d)[2])

    def spoon_pitch_deg(self, d) -> float:
        L = float(np.linalg.norm(np.array(self.cfg.spoon_head_local, dtype=float)))
        ratio = self.spoon_head_drop(d) / max(L, 1e-9)
        return float(np.degrees(np.arcsin(np.clip(ratio, -1.0, 1.0))))

    def tip_pos(self, d) -> np.ndarray:
        return d.site(self.spoon_tip_id).xpos.copy()

    def jacobians(self, d, model=None):
        if model is None:
            model = self.model
        jacp_site = np.zeros((3, model.nv))
        jacr_site = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, d, jacp_site, jacr_site, self.spoon_tip_id)
        jacp_body = np.zeros((3, model.nv))
        jacr_body = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, d, jacp_body, jacr_body, self.ee_id)
        return jacp_site[:, :model.nq], jacr_body[:, :model.nq]

    def singularity_metrics(self, d, model=None) -> Tuple[float, float]:
        Jp, Jr = self.jacobians(d, model)
        J = np.vstack([Jp, 0.15 * Jr])
        try:
            S = np.linalg.svd(J, compute_uv=False)
            sigma_min = float(np.min(S))
            sigma_max = float(np.max(S))
            return sigma_min, sigma_max / (sigma_min + 1e-12)
        except Exception:
            return 0.0, 1e12

    def orientation_errors(
        self,
        d,
        target_normal: np.ndarray,
        target_forward: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        n_cur = self.current_body_axis_world(d, np.array(self.cfg.spoon_normal_local, dtype=float))
        f_cur = self.current_body_axis_world(d, np.array(self.cfg.spoon_forward_local, dtype=float))
        n_tgt = normalize(target_normal)
        f_tgt = target_forward.copy()
        f_tgt[2] = 0.0
        f_tgt = normalize(f_tgt)
        f_cur_xy = f_cur.copy()
        f_cur_xy[2] = 0.0
        f_cur_xy = normalize(f_cur_xy)
        tilt_vec = np.cross(n_cur, n_tgt)
        forward_vec = np.cross(f_cur_xy, f_tgt)
        return (
            float(np.linalg.norm(tilt_vec)),
            float(np.linalg.norm(forward_vec)),
            float(np.dot(n_cur, n_tgt)),
            float(np.dot(f_cur_xy, f_tgt)),
        )


# =============================================================================
# Storage
# =============================================================================


class PrimitiveDatabase:
    """Persist and query generated scoop primitives."""

    def __init__(self, cfg: SystemConfig):
        self.cfg = cfg
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.cfg.out_dir / "scoop_primitives.json"
        self.csv_path = self.cfg.out_dir / "scoop_primitives_summary.csv"

    def save(self, primitives: List[ScoopPrimitive], regions: List[FoodRegion]):
        payload = {
            "config": {
                "tray_x_length": self.cfg.tray_x_length,
                "tray_y_length": self.cfg.tray_y_length,
                "base_y_offset": self.cfg.base_y_offset,
                "tray_surface_z": self.cfg.tray_surface_z,
            },
            "regions": [asdict(r) for r in regions],
            "primitives": [asdict(p) for p in primitives],
        }
        self.json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        with self.csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "primitive_id", "region_id", "food_x", "food_y", "drag_length", "score",
                "max_pos_error", "max_tilt_error", "preview_max_tilt", "min_sigma",
                "max_condition", "max_contact", "head_drop_pre", "head_drop_engage",
                "head_drop_drag_start", "head_drop_drag_end", "min_head_drop",
                "max_head_drop_error",
            ])
            for p in primitives:
                writer.writerow([
                    p.primitive_id, p.region_id, p.food_xy[0], p.food_xy[1], p.drag_length,
                    p.score, p.max_pos_error, p.max_tilt_error, p.preview_max_tilt,
                    p.min_sigma, p.max_condition, p.max_contact, p.head_drop_pre,
                    p.head_drop_engage, p.head_drop_drag_start, p.head_drop_drag_end,
                    p.min_head_drop, p.max_head_drop_error,
                ])

    def load(self) -> List[ScoopPrimitive]:
        if not self.json_path.exists():
            raise FileNotFoundError(f"LUT JSON file not found: {self.json_path}")
        payload = json.loads(self.json_path.read_text(encoding="utf-8"))
        return [ScoopPrimitive(**p) for p in payload["primitives"]]

    def select(self, region_id: int, food_xy: Tuple[float, float], top_k: int = 5) -> List[ScoopPrimitive]:
        primitives = [p for p in self.load() if p.region_id == int(region_id)]
        f = np.array(food_xy, dtype=float)
        return sorted(primitives, key=lambda p: (float(np.linalg.norm(np.array(p.food_xy) - f)), p.score))[:top_k]


class MouthConnectorDatabase:
    """Persist and load the mouth connector used by step9."""

    def __init__(self, cfg: SystemConfig):
        self.cfg = cfg
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.cfg.out_dir / "mouth_connector.json"

    def save(self, connector: MouthConnector):
        payload = {
            "config": {
                "mouth_y_range": self.cfg.mouth_y_range,
                "mouth_candidate_z_range": self.cfg.mouth_candidate_z_range,
                "mouth_forward_world": self.cfg.mouth_forward_world,
            },
            "connector": asdict(connector),
        }
        self.json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self) -> MouthConnector:
        connector, _config = self.load_payload()
        return connector

    def load_payload(self) -> Tuple[MouthConnector, Dict[str, object]]:
        if not self.json_path.exists():
            raise FileNotFoundError(f"Mouth connector file not found: {self.json_path}")
        payload = json.loads(self.json_path.read_text(encoding="utf-8"))
        return MouthConnector(**payload["connector"]), dict(payload.get("config", {}))


class NeutralDatabase:
    """Persist and load a neutral pose cache, kept for old result compatibility."""

    def __init__(self, cfg: SystemConfig):
        self.cfg = cfg
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.cfg.out_dir / "neutral.json"

    def save(self, connector: NeutralConnector):
        self.json_path.write_text(
            json.dumps({"connector": asdict(connector)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load_payload(self) -> Tuple[NeutralConnector, Dict[str, object]]:
        if not self.json_path.exists():
            raise FileNotFoundError(f"Neutral file not found: {self.json_path}")
        payload = json.loads(self.json_path.read_text(encoding="utf-8"))
        return NeutralConnector(**payload["connector"]), dict(payload.get("config", {}))
