# -*- coding: utf-8 -*-
"""
eating_scoop_system_v11.py

Single-file snapshot of the meal-assist robot pipeline.

This file is kept as a self-contained historical/integration version. The
modular implementation under `meal_assist/` is easier to read and modify for
new work.

Current pipeline:
1. Build feasible scoop primitives for tray food regions.
2. Use phase-aware IK targets for pre, engage, drag_start, drag_end, and lift.
3. Use FK-based spoon metrics:
   - spoon tip position for task tracking,
   - spoon normal/forward axes for posture,
   - head_drop = z_link7 - z_spoon_head for head-down posture.
4. Cache neutral and mouth connector poses to avoid expensive runtime IK.
5. Replay selected primitives with contact, tilt, joint-limit, and singularity
   monitoring.

Important note:
If `robot_model_v5_scene.xml` or major constraints change, rebuild the LUT
before running replay modes:

    python eating_scoop_system_v11.py --mode build_lut --regions 1 2 3 4 5

Common commands:
    python eating_scoop_system_v11.py --mode build_mouth_lut
    python eating_scoop_system_v11.py --mode build_lut --regions 5 --food_samples 3
    python eating_scoop_system_v11.py --mode run_region --region 5 --n_actions 1 --viewer
    python eating_scoop_system_v11.py --mode run_lut --n_actions 3 --viewer
    python eating_scoop_system_v11.py --mode test_run
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

import numpy as np

try:
    import mujoco
    import mujoco.viewer
except Exception as exc:  # pragma: no cover
    mujoco = None
    print("[WARN] Failed to import MuJoCo. Install it with `pip install mujoco`.", exc)


# =============================================================================
# 1. Configuration
# =============================================================================

@dataclass
class SystemConfig:
    """Central settings for geometry, IK, validation, cache, and replay."""

    # File paths.
    base_dir: Path = Path(__file__).resolve().parent
    xml_name: str = "robot_model_v5_scene.xml"
    # xml_name: str = "robot_model_ur5_scaled_scene.xml"
    # out_dir_name: str = "scoop_lut_output_v11_260610"
    out_dir_name: str = "scoop_lut_output_v11"

    # Tray frame. Origin is the lower-right tray corner; +X points away from
    # the user and the robot base is offset from this origin in world frame.
    tray_x_length: float = 0.21
    tray_y_length: float = 0.28
    base_x_offset: float = 0.075
    base_y_offset: float = 0.13
    base_z_offset: float = 0.0
    tray_surface_z: float = 0.04

    # Neutral region. IK samples this small cylindrical region and selects a
    # reachable, level, collision-free, joint-safe neutral posture.
    neutral_radius: float = 0.08
    neutral_z_min: float = 0.22                     # world z = tray_surface_z + this = 0.26
    neutral_z_max: float = 0.28                     # world z = 0.32
    neutral_grid_xy: int = 7
    neutral_grid_z: int = 3
    neutral_center_tray: Tuple[float, float] = (0.075, -0.02)
    # Legacy single neutral point used as a fallback/reference target.
    neutral_pos_tray: Tuple[float, float, float] = (0.105, 0.046, 0.32)
    neutral_pause_frames: int = 10
    neutral_position_tol: float = 0.025
    neutral_max_extra_frames: int = 1200
    # Neutral is a carry/return pose, so it prefers level posture rather than
    # head-down posture.
    neutral_normal_weight: float = 0.20
    neutral_head_drop_min: float = 0.0
    neutral_head_drop_weight: float = 0.0
    neutral_max_tilt: float = 0.10
    neutral_tilt_score_weight: float = 1.0
    neutral_joint_margin_ratio: float = 0.015
    neutral_margin_score_weight: float = 3.0
    neutral_good_margin: float = 0.10

    # Mouth target search region.
    mouth_grid_xyz: int = 20
    mouth_radius_xy: float = 0.025
    mouth_radius_z: float = 0.04
    default_mouth_pos_world: Tuple[float, float, float] = (0.0, 0.26, 0.45)
    mouth_forward_world: Tuple[float, float, float] = (-1.0, 0.0, 0.0)
    mouth_normal_weight: float = 0.12
    mouth_forward_weight: float = 0.03
    mouth_approach_frames: int = 960
    mouth_pause_frames: int = 30
    # Mouth delivery prioritizes tip position, so these validation thresholds
    # are looser than scoop-phase thresholds.
    mouth_max_pos_error: float = 0.020
    mouth_max_tilt_error: float = 0.04
    mouth_max_forward_error: float = 0.40

    # Mouth-specific joint-limit margin.
    mouth_joint_limit_margin_ratio: float = 0.015

    # Candidate grid for mouth-position multi-start IK.
    mouth_x: float = 0.0
    mouth_y_range: Tuple[float, float] = (0.10, 0.25)
    mouth_y_step: float = 0.02
    mouth_candidate_z_range: Tuple[float, float] = (0.30, 0.60)
    mouth_candidate_z_step: float = 0.05
    mouth_multi_start_seeds: int = 8
    mouth_pre_approach_distance: float = 0.050
    mouth_position_tol: float = 0.010
    mouth_max_extra_frames: int = 1800
    # Cache the expensive mouth and neutral IK searches.
    use_mouth_lut: bool = True
    use_neutral_lut: bool = True

    # Runtime monitoring during trajectory replay.
    monitor_contacts_during_replay: bool = True
    monitor_tilt_during_replay: bool = True
    abort_on_scoop_level_tilt: bool = False
    tilt_warn_threshold: float = 0.10

    # Scoop primitive parameters.
    scoop_drag_direction_world: Tuple[float, float, float] = (-1.0, 0.0, 0.0)
    drag_lengths: Tuple[float, ...] = (0.035, 0.050, 0.070)
    pre_scoop_height: float = 0.075
    engage_height: float = 0.018
    lift_height: float = 0.110
    scoop_start_offsets_x: Tuple[float, ...] = (0.020, 0.035, 0.050)
    scoop_y_offsets: Tuple[float, ...] = (-0.015, 0.0, 0.015)

    # Region sampling
    food_samples_per_region_axis: int = 5

    # IK parameters.
    ik_iters: int = 800
    ik_step_size: float = 0.05
    ik_dq_clip: float = 0.05
    posture_gain: float = 0.012
    normal_weight: float = 0.18
    forward_weight: float = 0.03
    multi_start_trials: int = 12
    random_seed: int = 13

    # Phase-aware spoon posture preferences.
    joint6_index: int = 5
    joint6_soft_enabled: bool = True
    joint6_soft_gain: float = 0.12
    joint6_pref_pre: float = -0.70
    joint6_pref_engage: float = -0.70
    joint6_pref_drag_start: float = -0.55
    joint6_pref_drag_end: float = -0.25
    joint6_pref_lift: float = -0.05
    # Joint 6 is a soft branch preference; FK head_drop is the posture proof.
    joint6_pitch_hard_enabled: bool = False
    joint6_head_down_max: float = -0.25
    joint6_pitch_soft_weight: float = 1.5
    spoon_head_local: Tuple[float, float, float] = (0.014529, 0.104125, 0.036995)
    head_drop_enabled: bool = True
    # head_drop = z_link7 - z_spoon_head. Positive means spoon head is lower
    # than the wrist. With |spoon_head_local| ~= 111.5 mm, 40 deg head-down is
    # roughly 72 mm head_drop.
    head_drop_min_pre: float = 0.070
    head_drop_min_engage: float = 0.075
    head_drop_min_drag_start: float = 0.065
    head_drop_min_drag_end: float = 0.030
    head_drop_weight_pre: float = 30.0
    head_drop_weight_engage: float = 50.0
    head_drop_weight_drag: float = 30.0
    head_drop_hard_min_pre: float = 0.055
    head_drop_hard_min_engage: float = 0.062
    head_drop_hard_min_drag_start: float = 0.050
    head_drop_hard_min_drag_end: float = 0.015
    # Lift disables head-down and enforces level posture via normal_hard.
    head_drop_min_lift: float = 0.0
    head_drop_weight_lift: float = 0.0
    head_drop_hard_min_lift: float = -1.0
    head_drop_ik_margin: float = 0.005
    head_drop_score_weight: float = 24.0
    normal_weight_pre: float = 0.00
    normal_weight_engage: float = 0.02
    normal_weight_drag: float = 0.06
    normal_weight_lift: float = 0.45

    # Debug logging
    debug_reject_log: bool = True
    debug_print_every_try: int = 5
    debug_print_first_n_failures: int = 12

    # Spoon local axes used for orientation errors.
    spoon_normal_local: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    spoon_forward_local: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    world_up: Tuple[float, float, float] = (0.0, 0.0, -1.0)

    # Validation thresholds.
    max_pos_error: float = 0.018
    # max_tilt_error applies only to targets with normal_hard=True.
    max_tilt_error: float = 0.10
    max_forward_error: float = 0.40
    min_forward_dot: float = 0.85
    min_sigma: float = 0.004
    max_condition: float = 900.0
    joint_limit_margin_ratio: float = 0.015
    contact_allowed: int = 0

    # Geometry margins for keeping the drag motion inside the tray/region.
    scoop_boundary_max_overshoot: float = 0.005
    spoon_outer_margin: float = 0.005

    # Replay
    frames_per_segment: int = 240
    ctrl_filter_alpha: float = 0.05
    move_max_extra_frames: int = 2400
    move_q_tolerance: float = 0.05
    pre_move_q_tolerance: float = 0.15
    pre_runtime_head_drop_min: float = 0.010
    mouth_q_tolerance: float = 0.010
    initial_random_radius: float = 0.35
    initial_random_candidates: int = 80
    initial_random_max_tip_distance: float = 0.18

    @property
    def xml_path(self) -> Path:
        """MuJoCo XML 모델 파일의 절대 경로를 반환한다."""
        return self.base_dir / self.xml_name

    @property
    def out_dir(self) -> Path:
        """LUT와 connector 캐시를 저장할 출력 디렉터리 경로를 반환한다."""
        return self.base_dir / self.out_dir_name


# =============================================================================
# 2. 데이터 구조
# =============================================================================

@dataclass
class PoseTarget:
    """IK가 맞춰야 하는 숟가락 tip 위치와 자세 제약을 묶은 목표 pose."""
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
    """식판 안의 음식 구역 polygon과 칸막이/경계 정보를 나타낸다."""
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
    """한 이동 또는 재생 단계의 성공 여부와 추적 오차, 접촉, 자세 지표를 기록한다."""
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
        """로그 출력용으로 핵심 실행 결과를 한 줄 문자열로 요약한다."""
        parts = [f"label={self.label}", f"ok={self.ok}", f"reason={self.reason}"]
        if self.pos_error is not None:
            parts.append(f"pos_err={self.pos_error*1000:.1f}mm")
        if self.q_error is not None:
            parts.append(f"q_err={self.q_error:.4f}rad")
        if self.tilt_error is not None:
            parts.append(f"tilt={np.degrees(self.tilt_error):.2f}deg")
        if self.head_drop is not None:
            parts.append(f"head_drop={self.head_drop*1000:.1f}mm")
        if self.contact:
            parts.append(f"contact={self.contact}")
        if self.extra_frames:
            parts.append(f"extra={self.extra_frames}")
        return "[STEP_RESULT] " + ", ".join(parts)


@dataclass
class ScoopPrimitive:
    """한 번의 scoop 동작을 구성하는 주요 waypoint, 관절각, 품질 점수를 저장한다."""
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
    """Neutral 이후 입 위치로 이동하기 위한 pre/delivery/retreat 관절 경로 캐시."""
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
    """Cached neutral-pose entry saved in ``neutral.json``."""
    q_neutral: List[float]
    neutral_pos: Tuple[float, float, float]
    pos_error: float
    tilt_error: float
    head_drop: float


# =============================================================================
# 3. 유틸
# =============================================================================

def normalize(v: np.ndarray) -> np.ndarray:
    """벡터를 0 나눗셈 없이 단위 벡터로 정규화한다."""
    return v / (np.linalg.norm(v) + 1e-12)


def point_in_polygon_xy(point: np.ndarray, polygon: List[Tuple[float, float]]) -> bool:
    """Ray casting point-in-polygon."""
    x, y = float(point[0]), float(point[1])
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if ((y1 > y) != (y2 > y)):
            x_intersect = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
            if x < x_intersect:
                inside = not inside
    return inside


def sample_points_in_polygon(polygon: List[Tuple[float, float]], n_axis: int) -> List[np.ndarray]:
    """Polygon 내부 격자점을 샘플링한다. boundary 꼭짓점을 피하기 위해 양 끝점을 제외한다."""
    arr = np.array(polygon, dtype=float)
    xmin, ymin = np.min(arr, axis=0)
    xmax, ymax = np.max(arr, axis=0)
    xs = np.linspace(xmin, xmax, n_axis + 2)[1:-1]
    ys = np.linspace(ymin, ymax, n_axis + 2)[1:-1]
    pts = []
    for x in xs:
        for y in ys:
            p = np.array([x, y], dtype=float)
            if point_in_polygon_xy(p, polygon):
                pts.append(p)
    return pts


def smoothstep5(t: float) -> float:
    """0~1 보간 인자를 양 끝 속도가 0인 5차 smoothstep 곡선으로 변환한다."""
    t = float(np.clip(t, 0.0, 1.0))
    return 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5


def wrap_angle(angle: float) -> float:
    """임의의 각도를 -pi~pi 범위로 감싼다."""
    return float(math.atan2(math.sin(angle), math.cos(angle)))


# =============================================================================
# 4. 식판 geometry
# =============================================================================

class TrayGeometry:
    """
    식판 frame과 robot/world frame 사이 변환을 담당한다.

    현재 가정:
    - MuJoCo world frame의 robot base를 [0,0,0]으로 둔다.
    - 식판 원점은 robot base 기준 [0, base_y_offset, base_z_offset]에 있다.
    - robot base는 식판 원점 기준 -Y축 상에 있으므로, tray origin은 robot 기준 +Y 쪽이다.
    - tray frame axis와 world frame axis는 일단 평행하다고 둔다.

    실제 설치에서 yaw/roll/pitch가 있으면 R_world_tray를 수정하면 된다.
    """

    def __init__(self, cfg: SystemConfig):
        """설정값으로 식판 원점과 world 변환 행렬을 초기화한다."""
        self.cfg = cfg
        self.p_world_tray_origin = np.array([cfg.base_x_offset, cfg.base_y_offset, cfg.base_z_offset], dtype=float)
        self.R_world_tray = np.eye(3)

    def tray_to_world(self, p_tray: np.ndarray) -> np.ndarray:
        """식판 좌표계의 점을 MuJoCo world 좌표계로 변환한다."""
        return self.p_world_tray_origin + self.R_world_tray @ p_tray

    def world_to_tray(self, p_world: np.ndarray) -> np.ndarray:
        """MuJoCo world 좌표계의 점을 식판 좌표계로 변환한다."""
        return self.R_world_tray.T @ (p_world - self.p_world_tray_origin)

    def default_regions(self) -> List[FoodRegion]:
        """Return the provisional five-region tray layout.

        Replace these polygons with measured tray/food-compartment boundaries
        when real hardware dimensions are available.
        """
        Lx = self.cfg.tray_x_length
        Ly = self.cfg.tray_y_length
        # Provisional five-way split of the tray.
        regions = [
            FoodRegion(1, "rice_or_main", [(0.00*Lx,0.00*Ly),(0.52*Lx,0.00*Ly),(0.52*Lx,0.48*Ly),(0.00*Lx,0.48*Ly)],
                       barrier_x=0.00*Lx, barrier_y_min=0.00*Ly, barrier_y_max=0.48*Ly, barrier_height=0.025, barrier_thickness=0.006),
            FoodRegion(2, "side_1", [(0.52*Lx,0.00*Ly),(1.00*Lx,0.00*Ly),(1.00*Lx,0.32*Ly),(0.52*Lx,0.32*Ly)],
                       barrier_x=0.52*Lx, barrier_y_min=0.00*Ly, barrier_y_max=0.32*Ly, barrier_height=0.025, barrier_thickness=0.006),
            FoodRegion(3, "side_2", [(0.52*Lx,0.32*Ly),(1.00*Lx,0.32*Ly),(1.00*Lx,0.64*Ly),(0.52*Lx,0.64*Ly)],
                       barrier_x=0.52*Lx, barrier_y_min=0.32*Ly, barrier_y_max=0.64*Ly, barrier_height=0.025, barrier_thickness=0.006),
            FoodRegion(4, "side_3", [(0.52*Lx,0.64*Ly),(1.00*Lx,0.64*Ly),(1.00*Lx,1.00*Ly),(0.52*Lx,1.00*Ly)],
                       barrier_x=0.52*Lx, barrier_y_min=0.64*Ly, barrier_y_max=1.00*Ly, barrier_height=0.025, barrier_thickness=0.006),
            FoodRegion(5, "soup_or_extra", [(0.00*Lx,0.48*Ly),(0.52*Lx,0.48*Ly),(0.52*Lx,1.00*Ly),(0.00*Lx,1.00*Ly)],
                       barrier_x=0.00*Lx, barrier_y_min=0.48*Ly, barrier_y_max=1.00*Ly, barrier_height=0.025, barrier_thickness=0.006),
        ]
        return regions

    def neutral_points_world(self) -> List[np.ndarray]:
        """Sample neutral candidate positions and return them in world coordinates."""
        cx, cy = self.cfg.neutral_center_tray
        center_tray = np.array([cx, cy, 0.0])
        pts = []
        xy = np.linspace(-self.cfg.neutral_radius, self.cfg.neutral_radius, self.cfg.neutral_grid_xy)
        zs = np.linspace(self.cfg.neutral_z_min, self.cfg.neutral_z_max, self.cfg.neutral_grid_z)
        for dx in xy:
            for dy in xy:
                if dx*dx + dy*dy <= self.cfg.neutral_radius**2:
                    for z in zs:
                        p_tray = center_tray + np.array([dx, dy, self.cfg.tray_surface_z + z])
                        pts.append(self.tray_to_world(p_tray))
        return pts

    # v4 신규: tray frame 기준 단일 Neutral point의 world 좌표.
    def neutral_pos_world(self) -> np.ndarray:
        """legacy 단일 neutral 위치를 world 좌표로 반환한다."""
        p_tray = np.array(self.cfg.neutral_pos_tray, dtype=float)
        return self.tray_to_world(p_tray)


# =============================================================================
# 5. Robot wrapper
# =============================================================================

class RobotModel:
    """MuJoCo 로봇 모델과 숟가락 FK/Jacobian/관절 제한 유틸리티를 감싼 래퍼."""
    def __init__(self, cfg: SystemConfig):
        """MuJoCo 모델, 데이터, 주요 site/body id, IK 전용 no-contact 모델을 준비한다."""
        if mujoco is None:
            raise RuntimeError("mujoco가 필요합니다. `pip install mujoco` 후 실행하세요.")
        if not cfg.xml_path.exists():
            raise FileNotFoundError(f"XML 파일을 찾지 못했습니다: {cfg.xml_path}")

        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(str(cfg.xml_path))
        self.data = mujoco.MjData(self.model)

        self.spoon_tip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "spoon_tip")
        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link7")
        if self.spoon_tip_id < 0:
            raise RuntimeError("site 'spoon_tip'을 찾지 못했습니다.")
        if self.ee_id < 0:
            raise RuntimeError("body 'link7'을 찾지 못했습니다.")

        self.q_center, self.q_half, self.q_lower, self.q_upper = self._joint_bounds()

        # IK 전용 no-contact 클론: iteration 중 지면·자기충돌 간섭 제거 (속도 향상 포함)
        self.ik_model = mujoco.MjModel.from_xml_path(str(cfg.xml_path))
        for i in range(self.ik_model.ngeom):
            self.ik_model.geom_contype[i] = 0
            self.ik_model.geom_conaffinity[i] = 0

        # v4 신규: q_neutral은 lazy init. compute_q_neutral()을 외부에서 호출하여 채운다.
        # (RobotModel 단독으로는 IKSolver 의존성을 갖지 않기 위해 외부에서 채움.)
        self.q_neutral: Optional[np.ndarray] = None
        self.neutral_target_world: Optional[np.ndarray] = None  # compute_q_neutral이 영역에서 선택한 위치

    def _joint_bounds(self):
        """MuJoCo hinge joint range에서 q 중심, 반폭, 하한, 상한 배열을 계산한다."""
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
        """data.qpos가 모델의 hinge joint range 밖으로 나가지 않도록 클램프한다."""
        if model is None:
            model = self.model
        for j in range(model.njnt):
            qadr = int(model.jnt_qposadr[j])
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and model.jnt_limited[j]:
                qmin, qmax = model.jnt_range[j]
                d.qpos[qadr] = np.clip(d.qpos[qadr], qmin, qmax)

    def set_q(self, d, q: np.ndarray, model=None):
        """지정한 관절각을 data에 넣고 속도 초기화, 제한 적용, forward kinematics를 수행한다."""
        if model is None:
            model = self.model
        d.qpos[:model.nq] = q[:model.nq]
        d.qvel[:] = 0.0
        self.enforce_joint_limits(d, model)
        mujoco.mj_forward(model, d)

    def sample_random_q(self, rng: np.random.Generator) -> np.ndarray:
        """관절 제한 margin 안쪽에서 무작위 관절 자세를 샘플링한다."""
        q = np.zeros(self.model.nq)
        for i in range(self.model.nq):
            lo, hi = self.q_lower[i], self.q_upper[i]
            margin = self.cfg.joint_limit_margin_ratio * max(hi - lo, 1e-6)
            q[i] = rng.uniform(lo + margin, hi - margin)
        return q

    def is_joint_limit_safe(self, q: np.ndarray) -> bool:
        """기본 joint_limit_margin_ratio를 만족하는 관절 자세인지 검사한다."""
        for i in range(self.model.nq):
            lo, hi = self.q_lower[i], self.q_upper[i]
            margin = self.cfg.joint_limit_margin_ratio * max(hi - lo, 1e-6)
            if q[i] <= lo + margin or q[i] >= hi - margin:
                return False
        return True

    def is_joint_limit_safe_with_margin(self, q: np.ndarray, margin_ratio: float) -> bool:
        """주어진 margin_ratio (예: 0.001 = 0.1%)로 관절 한계 안전성 검증."""
        for i in range(self.model.nq):
            lo, hi = self.q_lower[i], self.q_upper[i]
            margin = margin_ratio * max(hi - lo, 1e-6)
            if q[i] < lo + margin or q[i] > hi - margin:
                return False
        return True

    def min_joint_limit_margin_ratio(self, q: np.ndarray) -> float:
        """모든 관절 중 제한 경계까지 남은 최소 상대 margin을 반환한다."""
        min_margin = float("inf")
        for i in range(self.model.nq):
            lo, hi = self.q_lower[i], self.q_upper[i]
            span = max(hi - lo, 1e-6)
            ratio = min((q[i] - lo) / span, (hi - q[i]) / span)
            min_margin = min(min_margin, float(ratio))
        return min_margin

    def current_body_axis_world(self, d, local_axis: np.ndarray) -> np.ndarray:
        """link7 body의 local 축을 현재 자세 기준 world 방향 벡터로 변환한다."""
        R = d.body(self.ee_id).xmat.reshape(3, 3)
        return normalize(R @ normalize(local_axis))

    def local_point_world(self, d, local_point: np.ndarray) -> np.ndarray:
        """link7 local 좌표의 한 점을 현재 FK 기준 world 좌표로 변환한다."""
        R = d.body(self.ee_id).xmat.reshape(3, 3)
        p = d.body(self.ee_id).xpos.copy()
        return p + R @ np.array(local_point, dtype=float)

    def link7_origin_world(self, d) -> np.ndarray:
        """현재 link7 body 원점의 world 좌표를 반환한다."""
        return d.body(self.ee_id).xpos.copy()

    def spoon_head_world(self, d) -> np.ndarray:
        """설정된 spoon_head_local 점의 현재 world 좌표를 반환한다."""
        return self.local_point_world(d, np.array(self.cfg.spoon_head_local, dtype=float))

    def spoon_head_drop(self, d) -> float:
        """link7 원점보다 숟가락 head가 얼마나 아래에 있는지 z 차이를 계산한다."""
        return float(self.link7_origin_world(d)[2] - self.spoon_head_world(d)[2])

    def spoon_pitch_deg(self, d) -> float:
        """head_drop을 스푼 pitch 각도(도)로 환산. 양수=head-down.
        angle = asin(head_drop / L), L = |spoon_head_local| ≈ 111.5mm."""
        L = float(np.linalg.norm(np.array(self.cfg.spoon_head_local, dtype=float)))
        ratio = self.spoon_head_drop(d) / max(L, 1e-9)
        return float(np.degrees(np.arcsin(np.clip(ratio, -1.0, 1.0))))

    def point_jacobian_world(self, d, local_point: np.ndarray, model=None) -> np.ndarray:
        """link7 local point의 world 위치 Jacobian을 계산한다."""
        if model is None:
            model = self.model
        p_world = self.local_point_world(d, np.array(local_point, dtype=float))
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jac(model, d, jacp, jacr, p_world, self.ee_id)
        return jacp[:, :model.nq]

    def link7_origin_jacobian(self, d, model=None) -> np.ndarray:
        """link7 body 원점의 위치 Jacobian을 계산한다."""
        if model is None:
            model = self.model
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, d, jacp, jacr, self.ee_id)
        return jacp[:, :model.nq]

    def tip_pos(self, d) -> np.ndarray:
        """spoon_tip site의 현재 world 위치를 반환한다."""
        return d.site(self.spoon_tip_id).xpos.copy()

    def jacobians(self, d, model=None):
        """spoon_tip 위치 Jacobian과 link7 회전 Jacobian을 함께 계산한다."""
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
        """위치/회전 Jacobian SVD로 최소 특이값과 condition number를 계산한다."""
        Jp, Jr = self.jacobians(d, model)
        J = np.vstack([Jp, 0.15 * Jr])
        try:
            S = np.linalg.svd(J, compute_uv=False)
            sigma_min = float(np.min(S))
            sigma_max = float(np.max(S))
            condition = sigma_max / (sigma_min + 1e-12)
            return sigma_min, condition
        except Exception:
            return 0.0, 1e12

    def orientation_errors(self, d, target_normal: np.ndarray, target_forward: np.ndarray) -> Tuple[float, float, float, float]:
        """현재 숟가락 normal/forward와 목표 방향 사이의 tilt/yaw 오차 지표를 계산한다."""
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
        dot_up = float(np.dot(n_cur, n_tgt))
        dot_forward = float(np.dot(f_cur_xy, f_tgt))
        return float(np.linalg.norm(tilt_vec)), float(np.linalg.norm(forward_vec)), dot_up, dot_forward


# =============================================================================
# 6. Constrained DLS IK
# =============================================================================

class IKSolver:
    """위치, 자세, head-drop, joint preference를 함께 푸는 damped least-squares IK solver."""
    def __init__(self, cfg: SystemConfig, robot: RobotModel):
        """설정과 로봇 래퍼 참조를 보관한다."""
        self.cfg = cfg
        self.robot = robot

    def damped_pinv(self, J: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Jacobian 상태에 따라 damping을 조절한 의사역행렬과 특이성 지표를 반환한다."""
        S = np.linalg.svd(J, compute_uv=False)
        sigma_min = float(np.min(S))
        sigma_max = float(np.max(S))
        condition = sigma_max / (sigma_min + 1e-12)
        # sigma가 작을수록 damping 증가
        threshold = 0.03
        if sigma_min >= threshold:
            damping = 0.001
        else:
            r = 1.0 - sigma_min / threshold
            damping = 0.001 + r * r * (0.08 - 0.001)
        m = J.shape[0]
        J_dls = J.T @ np.linalg.inv(J @ J.T + damping*damping*np.eye(m))
        return J_dls, sigma_min, condition

    def solve_pose(self, target: PoseTarget, seed_q: np.ndarray, posture_ref: Optional[np.ndarray] = None) -> Tuple[bool, np.ndarray, Dict[str, float]]:
        # IK iteration은 no-contact 모델로 수행 — 지면·자기충돌 간섭 없이 빠르게 수렴
        """seed 자세에서 시작해 PoseTarget을 만족하는 관절해를 반복 IK로 찾는다."""
        ik_m = self.robot.ik_model
        d = mujoco.MjData(ik_m)
        self.robot.set_q(d, seed_q.copy(), ik_m)
        # posture ref: 항상 관절 중심값 기준 (null-space 안정화)
        if posture_ref is None:
            posture_ref = self.robot.q_center.copy()

        target_pos = np.array(target.pos, dtype=float)
        target_normal = np.array(target.normal, dtype=float)
        target_forward = np.array(target.forward, dtype=float)

        last = {}
        for _ in range(self.cfg.ik_iters):
            cur_pos = self.robot.tip_pos(d)
            pos_err = target_pos - cur_pos

            n_cur = self.robot.current_body_axis_world(d, np.array(self.cfg.spoon_normal_local, dtype=float))
            f_cur = self.robot.current_body_axis_world(d, np.array(self.cfg.spoon_forward_local, dtype=float))
            f_cur_xy = f_cur.copy(); f_cur_xy[2] = 0.0; f_cur_xy = normalize(f_cur_xy)
            f_tgt = target_forward.copy(); f_tgt[2] = 0.0; f_tgt = normalize(f_tgt)

            normal_err = np.cross(n_cur, normalize(target_normal))
            forward_err = np.cross(f_cur_xy, f_tgt)

            Jp, Jr = self.robot.jacobians(d, ik_m)
            J = np.vstack([
                Jp,
                target.normal_weight * Jr,
                target.forward_weight * Jr,
            ])
            e = np.hstack([
                pos_err,
                target.normal_weight * normal_err,
                target.forward_weight * forward_err,
            ])

            head_drop = self.robot.spoon_head_drop(d)
            head_drop_error = 0.0
            if (
                self.cfg.head_drop_enabled
                and target.head_drop_min is not None
                and target.head_drop_weight > 0.0
            ):
                # IK margin: head_drop_hard_min이 있는 phase에서만 margin 적용.
                # head_drop이 min을 넘어도 min+margin까지는 계속 밀어줌 → 다른 objective가
                # 끌어내리는 드리프트를 방지. min+margin 이상에서야 비활성화됨.
                # hard_min이 없는 phase(neutral 등)는 soft preference만 → margin 불필요.
                margin = self.cfg.head_drop_ik_margin if target.head_drop_hard_min is not None else 0.0
                effective_min = float(target.head_drop_min) + margin
                head_drop_error = max(0.0, effective_min - head_drop)
                if head_drop_error > 0.0:
                    J_link7 = self.robot.link7_origin_jacobian(d, ik_m)
                    J_head = self.robot.point_jacobian_world(
                        d, np.array(self.cfg.spoon_head_local, dtype=float), ik_m
                    )
                    J_drop = J_link7[2:3, :] - J_head[2:3, :]
                    J = np.vstack([J, float(target.head_drop_weight) * J_drop])
                    e = np.hstack([e, float(target.head_drop_weight) * head_drop_error])

            J_dls, sigma_min, condition = self.damped_pinv(J)
            dq_task = self.cfg.ik_step_size * (J_dls @ e)

            q_cur = d.qpos[:ik_m.nq].copy()
            q_ref = posture_ref[:ik_m.nq].copy()
            N = np.eye(ik_m.nq) - J_dls @ J
            posture_vec = self.cfg.posture_gain * (q_ref - q_cur)
            if (
                self.cfg.joint6_soft_enabled
                and target.joint6_pref is not None
                and 0 <= self.cfg.joint6_index < ik_m.nq
            ):
                j6 = self.cfg.joint6_index
                err6 = wrap_angle(float(target.joint6_pref) - float(q_cur[j6]))
                posture_vec[j6] += float(target.joint6_weight) * err6
            dq_posture = N @ posture_vec
            dq = dq_task + dq_posture
            dq = np.clip(dq, -self.cfg.ik_dq_clip, self.cfg.ik_dq_clip)

            d.qpos[:ik_m.nq] += dq
            self.robot.enforce_joint_limits(d, ik_m)
            mujoco.mj_forward(ik_m, d)

            tilt, fwd, dot_up, dot_forward = self.robot.orientation_errors(d, target_normal, target_forward)
            head_drop = self.robot.spoon_head_drop(d)
            head_drop_error = 0.0
            if target.head_drop_min is not None:
                head_drop_error = max(0.0, float(target.head_drop_min) - head_drop)
            head_drop_hard_error = 0.0
            if target.head_drop_hard_min is not None:
                head_drop_hard_error = max(0.0, float(target.head_drop_hard_min) - head_drop)
            joint6 = float(d.qpos[self.cfg.joint6_index]) if 0 <= self.cfg.joint6_index < ik_m.nq else 0.0
            joint6_pref = float(target.joint6_pref) if target.joint6_pref is not None else float("nan")
            joint6_error = (
                abs(wrap_angle(joint6_pref - joint6))
                if target.joint6_pref is not None else 0.0
            )
            joint6_hard_error = (
                max(0.0, joint6 - float(target.joint6_max))
                if target.joint6_hard and target.joint6_max is not None else 0.0
            )
            sigma_val, cond_val = self.robot.singularity_metrics(d, ik_m)
            last = {
                "pos_error": float(np.linalg.norm(target_pos - self.robot.tip_pos(d))),
                "tilt_error": tilt,
                "forward_error": fwd,
                "dot_up": dot_up,
                "dot_forward": dot_forward,
                "normal_hard": float(target.normal_hard),
                "forward_hard": float(target.forward_hard),
                "head_drop": head_drop,
                "head_drop_min": float(target.head_drop_min) if target.head_drop_min is not None else float("nan"),
                "head_drop_error": head_drop_error,
                "head_drop_hard_min": float(target.head_drop_hard_min) if target.head_drop_hard_min is not None else float("nan"),
                "head_drop_hard_error": head_drop_hard_error,
                "joint6": joint6,
                "joint6_pref": joint6_pref,
                "joint6_error": joint6_error,
                "joint6_max": float(target.joint6_max) if target.joint6_max is not None else float("nan"),
                "joint6_hard": float(target.joint6_hard),
                "joint6_hard_error": joint6_hard_error,
                "sigma_min": sigma_val,
                "condition": cond_val,
                "contact": 0,  # iteration 중 contact는 검사하지 않음 (validate_q_for_target에서 체크)
            }

            if (last["pos_error"] < self.cfg.max_pos_error and
                ((not target.normal_hard) or last["tilt_error"] < self.cfg.max_tilt_error) and
                (
                    target.forward_weight <= 0.0 or not target.forward_hard
                    or (
                        last["forward_error"] < self.cfg.max_forward_error
                        and last["dot_forward"] >= self.cfg.min_forward_dot
                    )
                ) and
                ((not target.joint6_hard) or last["joint6_hard_error"] <= 0.0) and
                (target.head_drop_hard_min is None or last["head_drop_hard_error"] <= 0.0)):
                break

        q = d.qpos[:ik_m.nq].copy()
        ok, val_metrics = self.validate_q_for_target(q, target)
        # 검증 결과의 contact 값으로 last 업데이트
        last["contact"] = val_metrics.get("contact", 0)
        return ok, q, last

    def validate_q_for_target(self, q: np.ndarray, target: PoseTarget) -> Tuple[bool, Dict[str, float]]:
        """scoop용 PoseTarget 기준으로 IK 결과의 위치, 자세, 접촉, 특이성, head-drop을 검증한다."""
        d = mujoco.MjData(self.robot.model)
        self.robot.set_q(d, q)
        target_pos = np.array(target.pos, dtype=float)
        target_normal = np.array(target.normal, dtype=float)
        target_forward = np.array(target.forward, dtype=float)
        pos_error = float(np.linalg.norm(self.robot.tip_pos(d) - target_pos))
        tilt, fwd, dot_up, dot_forward = self.robot.orientation_errors(d, target_normal, target_forward)
        head_drop = self.robot.spoon_head_drop(d)
        head_drop_error = 0.0
        if target.head_drop_min is not None:
            head_drop_error = max(0.0, float(target.head_drop_min) - head_drop)
        head_drop_hard_error = 0.0
        if target.head_drop_hard_min is not None:
            head_drop_hard_error = max(0.0, float(target.head_drop_hard_min) - head_drop)
        joint6 = float(q[self.cfg.joint6_index]) if 0 <= self.cfg.joint6_index < len(q) else 0.0
        joint6_pref = float(target.joint6_pref) if target.joint6_pref is not None else float("nan")
        joint6_error = (
            abs(wrap_angle(joint6_pref - joint6))
            if target.joint6_pref is not None else 0.0
        )
        joint6_hard_error = (
            max(0.0, joint6 - float(target.joint6_max))
            if target.joint6_hard and target.joint6_max is not None else 0.0
        )
        sigma, condition = self.robot.singularity_metrics(d)
        contact = int(d.ncon)
        # target.normal_weight / forward_weight가 0이면 해당 orientation constraint는
        # solve와 validation 양쪽에서 비활성화된 것으로 취급한다.
        # 기존 scoop target은 두 weight가 모두 양수라 기존 동작과 동일하다.
        normal_ok = (not target.normal_hard) or (tilt <= self.cfg.max_tilt_error)
        forward_ok = (
            target.forward_weight <= 0.0 or not target.forward_hard
            or (fwd <= self.cfg.max_forward_error and dot_forward >= self.cfg.min_forward_dot)
        )
        joint6_hard_ok = (not target.joint6_hard) or (joint6_hard_error <= 0.0)
        head_drop_ok = target.head_drop_hard_min is None or head_drop_hard_error <= 0.0
        ok = (
            pos_error <= self.cfg.max_pos_error and
            normal_ok and
            forward_ok and
            joint6_hard_ok and
            head_drop_ok and
            sigma >= self.cfg.min_sigma and
            condition <= self.cfg.max_condition and
            contact <= self.cfg.contact_allowed and
            self.robot.is_joint_limit_safe(q)
        )
        metrics: Dict[str, float] = {
            "pos_error": pos_error,
            "tilt_error": tilt,
            "forward_error": fwd,
            "dot_up": dot_up,
            "dot_forward": dot_forward,
            "normal_hard": float(target.normal_hard),
            "forward_hard": float(target.forward_hard),
            "head_drop": head_drop,
            "head_drop_min": float(target.head_drop_min) if target.head_drop_min is not None else float("nan"),
            "head_drop_error": head_drop_error,
            "head_drop_hard_min": float(target.head_drop_hard_min) if target.head_drop_hard_min is not None else float("nan"),
            "head_drop_hard_error": head_drop_hard_error,
            "joint6": joint6,
            "joint6_pref": joint6_pref,
            "joint6_error": joint6_error,
            "joint6_max": float(target.joint6_max) if target.joint6_max is not None else float("nan"),
            "joint6_hard": float(target.joint6_hard),
            "joint6_hard_error": joint6_hard_error,
            "sigma_min": sigma,
            "condition": condition,
            "contact": float(contact),
        }
        return ok, metrics

    def validate_q_for_mouth(self, q: np.ndarray, target: PoseTarget) -> Tuple[bool, Dict[str, float]]:
        """Mouth 전달 전용 완화된 validation.

        Scoop IK는 식판 위 음식 위치 정확성과 숟가락 자세를 엄격하게 요구하지만,
        Mouth 전달은 (1) 위치가 입 근처에 있고 (2) 숟가락이 대체로 평평하며
        (3) singularity/충돌이 없으면 충분하다. 따라서 cfg.mouth_max_* 한계치를
        사용해 한 단계 더 관대한 검증을 수행한다.
        """
        d = mujoco.MjData(self.robot.model)
        self.robot.set_q(d, q)
        target_pos = np.array(target.pos, dtype=float)
        target_normal = np.array(target.normal, dtype=float)
        target_forward = np.array(target.forward, dtype=float)
        pos_error = float(np.linalg.norm(self.robot.tip_pos(d) - target_pos))
        tilt, fwd, dot_up, dot_forward = self.robot.orientation_errors(d, target_normal, target_forward)
        sigma, condition = self.robot.singularity_metrics(d)
        contact = int(d.ncon)

        normal_ok = (target.normal_weight <= 0.0) or (tilt <= self.cfg.mouth_max_tilt_error)
        forward_ok = (
            target.forward_weight <= 0.0
            or (fwd <= self.cfg.mouth_max_forward_error and dot_forward >= self.cfg.min_forward_dot)
        )
        # Mouth delivery also requires a joint-limit margin.
        joint_ok = self.robot.is_joint_limit_safe_with_margin(q, self.cfg.mouth_joint_limit_margin_ratio)
        min_margin_ratio = self.robot.min_joint_limit_margin_ratio(q)
        ok = (
            pos_error <= self.cfg.mouth_max_pos_error and
            normal_ok and
            forward_ok and
            sigma >= self.cfg.min_sigma and
            condition <= self.cfg.max_condition and
            contact <= self.cfg.contact_allowed and
            joint_ok
        )
        metrics: Dict[str, float] = {
            "pos_error": pos_error,
            "tilt_error": tilt,
            "forward_error": fwd,
            "dot_up": dot_up,
            "dot_forward": dot_forward,
            "sigma_min": sigma,
            "condition": condition,
            "contact": float(contact),
            "joint_limit_ok": float(joint_ok),
            "min_joint_margin_ratio": float(min_margin_ratio),
        }
        return ok, metrics


# =============================================================================
# 7. Scoop primitive 생성
# =============================================================================

class ScoopPrimitiveBuilder:
    """식판 샘플과 IK solver를 이용해 region별 feasible scoop primitive LUT를 만든다."""
    def __init__(self, cfg: SystemConfig, tray: TrayGeometry, robot: RobotModel, ik: IKSolver):
        """primitive 생성에 필요한 설정, 기하, 로봇, IK solver와 난수원을 초기화한다."""
        self.cfg = cfg
        self.tray = tray
        self.robot = robot
        self.ik = ik
        self.rng = np.random.default_rng(cfg.random_seed)
        self.target_normal = np.array(cfg.world_up, dtype=float)
        self.target_forward = np.array(cfg.scoop_drag_direction_world, dtype=float)

    def make_pose_targets(self, food_xy_tray: np.ndarray, drag_len: float, start_offset_x: float, y_offset: float) -> Dict[str, PoseTarget]:
        """음식 위치와 drag 파라미터를 pre/engage/drag/lift PoseTarget 묶음으로 변환한다."""
        x_food, y_food = float(food_xy_tray[0]), float(food_xy_tray[1] + y_offset)
        z_surface = self.cfg.tray_surface_z

        # -X 방향으로 끌어오기 위해 +X 쪽에서 시작해서 -X 쪽으로 끝낸다.
        drag_start_tray = np.array([x_food + start_offset_x, y_food, z_surface + self.cfg.engage_height], dtype=float)
        drag_end_tray   = np.array([x_food + start_offset_x - drag_len, y_food, z_surface + self.cfg.engage_height], dtype=float)
        pre_tray        = drag_start_tray + np.array([0.0, 0.0, self.cfg.pre_scoop_height], dtype=float)
        engage_tray     = drag_start_tray.copy()
        lift_tray       = drag_end_tray + np.array([0.0, 0.0, self.cfg.lift_height], dtype=float)

        def W(p_tray):
            return tuple(self.tray.tray_to_world(p_tray).tolist())

        def T(
            name: str,
            pos,
            normal_weight: float,
            joint6_pref: float,
            joint6_hard: bool,
            head_drop_min: Optional[float],
            head_drop_weight: float,
            head_drop_hard_min: Optional[float] = None,
            normal_hard: bool = False,
        ) -> PoseTarget:
            return PoseTarget(
                name=name,
                pos=W(pos),
                forward=tuple(self.target_forward.tolist()),
                normal=tuple(self.target_normal.tolist()),
                normal_weight=float(normal_weight),
                forward_weight=self.cfg.forward_weight,
                # Scoop phases use soft posture preferences; lift is level-hard.
                normal_hard=bool(normal_hard),
                forward_hard=True,
                joint6_pref=float(joint6_pref),
                joint6_weight=float(self.cfg.joint6_soft_gain),
                joint6_max=float(self.cfg.joint6_head_down_max) if joint6_hard else None,
                joint6_hard=bool(self.cfg.joint6_pitch_hard_enabled and joint6_hard),
                head_drop_min=head_drop_min,
                head_drop_weight=float(head_drop_weight),
                head_drop_hard_min=head_drop_hard_min,
            )
        return {
            "pre": T(
                "pre", pre_tray, self.cfg.normal_weight_pre, self.cfg.joint6_pref_pre, False,
                self.cfg.head_drop_min_pre, self.cfg.head_drop_weight_pre,
                self.cfg.head_drop_hard_min_pre,
            ),
            "engage": T(
                "engage", engage_tray, self.cfg.normal_weight_engage, self.cfg.joint6_pref_engage, False,
                self.cfg.head_drop_min_engage, self.cfg.head_drop_weight_engage,
                self.cfg.head_drop_hard_min_engage,
            ),
            "drag_start": T(
                "drag_start", drag_start_tray, self.cfg.normal_weight_drag, self.cfg.joint6_pref_drag_start, False,
                self.cfg.head_drop_min_drag_start, self.cfg.head_drop_weight_drag,
                self.cfg.head_drop_hard_min_drag_start,
            ),
            "drag_end": T(
                "drag_end", drag_end_tray, self.cfg.normal_weight_drag, self.cfg.joint6_pref_drag_end, False,
                self.cfg.head_drop_min_drag_end, self.cfg.head_drop_weight_drag,
                self.cfg.head_drop_hard_min_drag_end,
            ),
            # Lift starts transport: disable head-down and enforce level posture.
            "lift": T(
                "lift", lift_tray, self.cfg.normal_weight_lift, self.cfg.joint6_pref_lift, False,
                None, 0.0,
                None,
                normal_hard=True,
            ),
        }

    def seed_list(self) -> List[np.ndarray]:
        """multi-start IK에 사용할 기본 자세와 무작위 seed 목록을 만든다."""
        seeds = [np.zeros(self.robot.model.nq), self.robot.q_center.copy()]
        for _ in range(self.cfg.multi_start_trials):
            seeds.append(self.robot.sample_random_q(self.rng))
        return seeds

    def classify_ik_failure(self, metrics: Dict[str, float]) -> str:
        """IK가 실패했을 때 가장 직접적인 reject 이유를 문자열로 반환한다."""
        if not metrics:
            return "ik_no_metrics"
        if metrics.get("contact", 0) > self.cfg.contact_allowed:
            return "ik_contact"
        if metrics.get("pos_error", 1e9) > self.cfg.max_pos_error:
            return "ik_pos_error"
        if metrics.get("head_drop_hard_error", 0.0) > 0.0:
            return "ik_head_drop_error"
        if metrics.get("joint6_hard", 0.0) > 0.0 and metrics.get("joint6_hard_error", 0.0) > 0.0:
            return "ik_joint6_head_down_error"
        if metrics.get("normal_hard", 0.0) > 0.0 and metrics.get("tilt_error", 1e9) > self.cfg.max_tilt_error:
            return "ik_tilt_error"
        if metrics.get("forward_hard", 1.0) > 0.0 and metrics.get("forward_error", 1e9) > self.cfg.max_forward_error:
            return "ik_forward_error"
        if metrics.get("sigma_min", 1e9) < self.cfg.min_sigma:
            return "ik_singularity_sigma"
        if metrics.get("condition", 0.0) > self.cfg.max_condition:
            return "ik_singularity_condition"
        return "ik_unknown"

    def solve_sequence(
        self,
        targets: Dict[str, PoseTarget],
        return_reason: bool = False,
    ):
        """하나의 scoop 후보에 대해 각 phase IK를 순차적으로 풀고 가장 좋은 해를 고른다."""
        best = None
        reject_counter = Counter()
        last_failure = {}

        for seed_idx, seed in enumerate(self.seed_list()):
            q_map = {}
            m_map = {}
            q_cur = seed.copy()
            ok_all = True

            for key in ["pre", "engage", "drag_start", "drag_end", "lift"]:
                ok, q_sol, metrics = self.ik.solve_pose(targets[key], q_cur, posture_ref=q_cur)
                if not ok:
                    ok_all = False
                    reason = f"{key}:{self.classify_ik_failure(metrics)}"
                    reject_counter[reason] += 1
                    last_failure = {
                        "seed_idx": seed_idx,
                        "pose": key,
                        "reason": reason,
                        **metrics,
                    }
                    break
                q_map[key] = q_sol
                m_map[key] = metrics
                q_cur = q_sol.copy()

            if not ok_all:
                continue

            preview_ok, preview_metrics = self.preview_sequence(q_map)
            if not preview_ok:
                reason = f"preview:{preview_metrics.get('reason', 'unknown')}"
                reject_counter[reason] += 1
                last_failure = {"seed_idx": seed_idx, "pose": "preview", "reason": reason, **preview_metrics}
                continue

            # score: 위치/tilt/condition/관절 변화량을 종합
            qs = [q_map[k] for k in ["pre", "engage", "drag_start", "drag_end", "lift"]]
            joint_motion = sum(float(np.linalg.norm(qs[i+1] - qs[i])) for i in range(len(qs)-1))
            max_tilt = max(m["tilt_error"] for m in m_map.values())
            max_pos = max(m["pos_error"] for m in m_map.values())
            max_cond = max(m["condition"] for m in m_map.values())
            max_joint6_error = max(
                abs(wrap_angle(m.get("joint6_pref", m.get("joint6", 0.0)) - m.get("joint6", 0.0)))
                for m in m_map.values()
                if not math.isnan(m.get("joint6_pref", float("nan")))
            )
            max_joint6_hard_error = max(m.get("joint6_hard_error", 0.0) for m in m_map.values())
            max_head_drop_error = max(m.get("head_drop_error", 0.0) for m in m_map.values())
            score = (
                2.0*joint_motion
                + 20.0*max_tilt
                + 10.0*max_pos
                + 0.0005*max_cond
                + self.cfg.joint6_pitch_soft_weight*max_joint6_hard_error
                + self.cfg.head_drop_score_weight*max_head_drop_error
                + 0.5*max_joint6_error
            )
            if best is None or score < best[0]:
                best = (score, q_map, m_map, preview_metrics)

        if best is None:
            if return_reason:
                return None, {"reject_counter": reject_counter, "last_failure": last_failure}
            return None

        if return_reason:
            return (best[1], best[2], best[3]), {"reject_counter": reject_counter, "last_failure": last_failure}
        return best[1], best[2], best[3]

    def preview_sequence(self, q_map: Dict[str, np.ndarray]) -> Tuple[bool, Dict[str, float]]:
        """관절 보간으로 후보 trajectory를 미리 재생하며 접촉, 특이성, tilt를 검사한다."""
        d = mujoco.MjData(self.robot.model)
        keys = ["pre", "engage", "drag_start", "drag_end", "lift"]
        self.robot.set_q(d, q_map[keys[0]])
        max_contact = 0
        min_sigma = 1e9
        max_condition = 0.0
        max_joint_step = 0.0
        max_tilt = 0.0
        target_normal = np.array(self.cfg.world_up, dtype=float)
        target_forward = np.array(self.cfg.scoop_drag_direction_world, dtype=float)
        for a, b in zip(keys[:-1], keys[1:]):
            q0, q1 = q_map[a], q_map[b]
            max_joint_step = max(max_joint_step, float(np.max(np.abs(q1 - q0))))
            for i in range(self.cfg.frames_per_segment):
                s = smoothstep5(i / max(self.cfg.frames_per_segment - 1, 1))
                q = (1.0 - s) * q0 + s * q1
                d.qpos[:self.robot.model.nq] = q[:self.robot.model.nq]
                d.qvel[:] = 0.0
                self.robot.enforce_joint_limits(d)
                mujoco.mj_forward(self.robot.model, d)
                max_contact = max(max_contact, int(d.ncon))
                sigma, cond = self.robot.singularity_metrics(d)
                min_sigma = min(min_sigma, sigma)
                max_condition = max(max_condition, cond)
                tilt, _fwd, _du, _df = self.robot.orientation_errors(d, target_normal, target_forward)
                max_tilt = max(max_tilt, tilt)
                if max_contact > self.cfg.contact_allowed:
                    return False, {"reason": "contact", "max_contact": max_contact, "max_tilt": max_tilt}
                if sigma < self.cfg.min_sigma or cond > self.cfg.max_condition:
                    return False, {"reason": "singularity", "min_sigma": min_sigma, "max_condition": max_condition, "max_tilt": max_tilt}
                if not self.robot.is_joint_limit_safe(d.qpos[:self.robot.model.nq]):
                    return False, {"reason": "joint_limit", "max_tilt": max_tilt}
        return True, {
            "max_contact": max_contact,
            "min_sigma": min_sigma,
            "max_condition": max_condition,
            "max_joint_step": max_joint_step,
            "max_tilt": max_tilt,
        }

    def build_for_region(self, region: FoodRegion) -> List[ScoopPrimitive]:
        """한 food region의 샘플/drag 조합을 전부 탐색해 feasible primitive 목록을 생성한다."""
        primitives: List[ScoopPrimitive] = []
        food_pts = sample_points_in_polygon(region.polygon_xy, self.cfg.food_samples_per_region_axis)
        counter = 0

        total_try = (
            len(food_pts)
            * len(self.cfg.drag_lengths)
            * len(self.cfg.scoop_start_offsets_x)
            * len(self.cfg.scoop_y_offsets)
        )
        try_idx = 0
        reject_counter = Counter()
        first_failures = []

        print(f"[REGION {region.region_id}] food sample count = {len(food_pts)}", flush=True)

        for food_i, food_xy in enumerate(food_pts):
            print(f"[REGION {region.region_id}] food {food_i+1}/{len(food_pts)} xy={food_xy}", flush=True)
            for drag_len in self.cfg.drag_lengths:
                for sx in self.cfg.scoop_start_offsets_x:
                    for sy in self.cfg.scoop_y_offsets:
                        try_idx += 1
                        if self.cfg.debug_reject_log and try_idx % self.cfg.debug_print_every_try == 0:
                            print(f"    try {try_idx}/{total_try}, saved={len(primitives)}", flush=True)

                        # Reject geometry-invalid candidates before running IK.
                        Lx = self.cfg.tray_x_length
                        Ly = self.cfg.tray_y_length
                        outer = self.cfg.spoon_outer_margin
                        overshoot = self.cfg.scoop_boundary_max_overshoot

                        drag_start_xy = np.array([food_xy[0] + sx, food_xy[1] + sy])
                        drag_end_xy = np.array([food_xy[0] + sx - drag_len, food_xy[1] + sy])

                        # (a) drag_start, drag_end 가 식판 외곽을 벗어나면 reject (spoon_outer_margin 내부)
                        def _within_tray(pt):
                            return (outer <= pt[0] <= Lx - outer) and (outer <= pt[1] <= Ly - outer)

                        if not _within_tray(drag_start_xy):
                            reject_counter["geometry:drag_start_out_of_tray"] += 1
                            if len(first_failures) < self.cfg.debug_print_first_n_failures:
                                first_failures.append((try_idx, "geometry:drag_start_out_of_tray", food_xy.copy(), drag_len, sx, sy, drag_start_xy.copy()))
                            continue
                        if not _within_tray(drag_end_xy):
                            reject_counter["geometry:drag_end_out_of_tray"] += 1
                            if len(first_failures) < self.cfg.debug_print_first_n_failures:
                                first_failures.append((try_idx, "geometry:drag_end_out_of_tray", food_xy.copy(), drag_len, sx, sy, drag_end_xy.copy()))
                            continue

                        # (b) y 방향 region 침범: drag_start/drag_end 모두 region barrier_y 범위 ± overshoot 내
                        y_min = region.barrier_y_min - overshoot
                        y_max = region.barrier_y_max + overshoot
                        if not (y_min <= drag_start_xy[1] <= y_max):
                            reject_counter["geometry:drag_start_y_overshoot"] += 1
                            if len(first_failures) < self.cfg.debug_print_first_n_failures:
                                first_failures.append((try_idx, "geometry:drag_start_y_overshoot", food_xy.copy(), drag_len, sx, sy, drag_start_xy.copy()))
                            continue
                        if not (y_min <= drag_end_xy[1] <= y_max):
                            reject_counter["geometry:drag_end_y_overshoot"] += 1
                            if len(first_failures) < self.cfg.debug_print_first_n_failures:
                                first_failures.append((try_idx, "geometry:drag_end_y_overshoot", food_xy.copy(), drag_len, sx, sy, drag_end_xy.copy()))
                            continue

                        # (c) region X 침범: barrier_x를 넘어 -X 쪽 침범 허용은 overshoot까지만
                        if drag_end_xy[0] < region.barrier_x - overshoot:
                            reject_counter["geometry:drag_end_too_far_minus_x"] += 1
                            if len(first_failures) < self.cfg.debug_print_first_n_failures:
                                first_failures.append((try_idx, "geometry:drag_end_too_far_minus_x", food_xy.copy(), drag_len, sx, sy, drag_end_xy.copy()))
                            continue

                        targets = self.make_pose_targets(food_xy, drag_len, sx, sy)
                        result, debug_info = self.solve_sequence(targets, return_reason=True)
                        if result is None:
                            local_counter = debug_info.get("reject_counter", Counter())
                            if len(local_counter) == 0:
                                reject_counter["sequence:unknown"] += 1
                                reason_for_first = "sequence:unknown"
                            else:
                                reject_counter.update(local_counter)
                                reason_for_first = local_counter.most_common(1)[0][0]
                            if len(first_failures) < self.cfg.debug_print_first_n_failures:
                                first_failures.append((try_idx, reason_for_first, food_xy.copy(), drag_len, sx, sy, drag_end_xy.copy(), debug_info.get("last_failure", {})))
                            continue
                        q_map, m_map, preview_metrics = result

                        max_pos = max(m["pos_error"] for m in m_map.values())
                        max_tilt = max(m["tilt_error"] for m in m_map.values())
                        max_fwd = max(m["forward_error"] for m in m_map.values())
                        min_sigma = min(m["sigma_min"] for m in m_map.values())
                        max_cond = max(m["condition"] for m in m_map.values())
                        max_contact = max(m["contact"] for m in m_map.values())
                        head_drop_pre = float(m_map["pre"].get("head_drop", 0.0))
                        head_drop_engage = float(m_map["engage"].get("head_drop", 0.0))
                        head_drop_drag_start = float(m_map["drag_start"].get("head_drop", 0.0))
                        head_drop_drag_end = float(m_map["drag_end"].get("head_drop", 0.0))
                        min_head_drop = min(
                            head_drop_pre,
                            head_drop_engage,
                            head_drop_drag_start,
                            head_drop_drag_end,
                        )
                        max_joint6_error = max(
                            abs(wrap_angle(m.get("joint6_pref", m.get("joint6", 0.0)) - m.get("joint6", 0.0)))
                            for m in m_map.values()
                            if not math.isnan(m.get("joint6_pref", float("nan")))
                        )
                        max_joint6_hard_error = max(m.get("joint6_hard_error", 0.0) for m in m_map.values())
                        max_head_drop_error = max(m.get("head_drop_error", 0.0) for m in m_map.values())
                        joint_motion = sum(
                            float(np.linalg.norm(q_map[b] - q_map[a]))
                            for a, b in zip(["pre","engage","drag_start","drag_end"], ["engage","drag_start","drag_end","lift"])
                        )
                        score = (
                            2.0*joint_motion
                            + 20.0*max_tilt
                            + 10.0*max_pos
                            + 0.0005*max_cond
                            + self.cfg.joint6_pitch_soft_weight*max_joint6_hard_error
                            + self.cfg.head_drop_score_weight*max_head_drop_error
                            + 0.5*max_joint6_error
                        )
                        pid = f"R{region.region_id:02d}_{counter:05d}"
                        counter += 1
                        primitives.append(ScoopPrimitive(
                            primitive_id=pid,
                            region_id=region.region_id,
                            food_xy=(float(food_xy[0]), float(food_xy[1])),
                            drag_length=float(drag_len),
                            score=float(score),
                            pre_scoop_pos=targets["pre"].pos,
                            engage_pos=targets["engage"].pos,
                            drag_start_pos=targets["drag_start"].pos,
                            drag_end_pos=targets["drag_end"].pos,
                            lift_pos=targets["lift"].pos,
                            q_pre=q_map["pre"].tolist(),
                            q_engage=q_map["engage"].tolist(),
                            q_drag_start=q_map["drag_start"].tolist(),
                            q_drag_end=q_map["drag_end"].tolist(),
                            q_lift=q_map["lift"].tolist(),
                            max_pos_error=float(max_pos),
                            max_tilt_error=float(max_tilt),
                            max_forward_error=float(max_fwd),
                            min_sigma=float(min_sigma),
                            max_condition=float(max_cond),
                            max_contact=int(max_contact),
                            preview_max_tilt=float(preview_metrics.get("max_tilt", 0.0)),
                            head_drop_pre=head_drop_pre,
                            head_drop_engage=head_drop_engage,
                            head_drop_drag_start=head_drop_drag_start,
                            head_drop_drag_end=head_drop_drag_end,
                            min_head_drop=float(min_head_drop),
                            max_head_drop_error=float(max_head_drop_error),
                        ))
                        print(
                            f"[OK] region={region.region_id} primitive={pid} food={food_xy} "
                            f"score={score:.4f} min_head_drop={min_head_drop*1000:.1f}mm "
                            f"max_head_drop_err={max_head_drop_error*1000:.1f}mm",
                            flush=True,
                        )

        print(f"[REGION {region.region_id} REJECT SUMMARY]", flush=True)
        if len(reject_counter) == 0:
            print("    no rejects", flush=True)
        else:
            for reason, count in reject_counter.most_common():
                print(f"    {reason}: {count}", flush=True)

        if first_failures:
            print(f"[REGION {region.region_id} FIRST FAILURES]", flush=True)
            for item in first_failures:
                try_no, reason, food_xy, drag_len, sx, sy, drag_end_xy, *rest = item
                print(
                    f"    try={try_no}, reason={reason}, food={food_xy}, "
                    f"drag={drag_len}, sx={sx}, sy={sy}, drag_end={drag_end_xy}",
                    flush=True,
                )
                if rest and rest[0]:
                    lf = rest[0]
                    print(f"        last_failure={lf}", flush=True)

        primitives.sort(key=lambda p: p.score)
        return primitives


# =============================================================================
# 8. LUT 저장/로드/온라인 선택
# =============================================================================

class PrimitiveDatabase:
    """scoop primitive LUT의 JSON/CSV 저장, 로드, 선택을 담당한다."""
    def __init__(self, cfg: SystemConfig):
        """출력 디렉터리와 primitive JSON/CSV 경로를 준비한다."""
        self.cfg = cfg
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.cfg.out_dir / "scoop_primitives.json"
        self.csv_path = self.cfg.out_dir / "scoop_primitives_summary.csv"

    def save(self, primitives: List[ScoopPrimitive], regions: List[FoodRegion]):
        """생성된 primitive와 region 정보를 JSON 원본과 CSV 요약 파일로 저장한다."""
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
        print("[SAVE]", self.json_path)
        print("[SAVE]", self.csv_path)

    def load(self) -> List[ScoopPrimitive]:
        """JSON LUT에서 ScoopPrimitive 목록을 복원한다."""
        if not self.json_path.exists():
            raise FileNotFoundError(f"LUT JSON 파일이 없습니다. 먼저 --mode build_lut 실행: {self.json_path}")
        payload = json.loads(self.json_path.read_text(encoding="utf-8"))
        return [ScoopPrimitive(**p) for p in payload["primitives"]]

    def load_summary_rows(self) -> List[Dict[str, str]]:
        """build_lut가 만든 CSV summary를 읽는다.

        주의: 현재 summary CSV에는 관절각(q_pre~q_lift)이 없으므로,
        CSV는 실행할 primitive_id/score/region/food_xy를 고르는 index로 사용하고
        실제 관절 trajectory는 같은 LUT의 JSON에서 primitive_id로 가져온다.
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(f"LUT CSV 파일이 없습니다. 먼저 --mode build_lut 실행: {self.csv_path}")
        with self.csv_path.open("r", newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))

    def select(self, region_id: int, food_xy: Tuple[float, float], top_k: int = 5) -> List[ScoopPrimitive]:
        """region과 현재 food_xy에 가까운 primitive를 score와 거리 기준으로 고른다."""
        primitives = [p for p in self.load() if p.region_id == region_id]
        if not primitives:
            return []
        f = np.array(food_xy, dtype=float)
        def key(p: ScoopPrimitive):
            dist = float(np.linalg.norm(np.array(p.food_xy) - f))
            return (dist, p.score)
        return sorted(primitives, key=key)[:top_k]

    def auto_select_from_lut(
        self,
        region_id: Optional[int] = None,
        n_actions: Optional[int] = 1,
        seed: Optional[int] = None,
        shuffle: bool = False,
        unique_food_xy: bool = True,
    ) -> List[ScoopPrimitive]:
        """CSV summary를 기준으로 실행 primitive를 자동 선택한다.

        - region_id=None이면 전체 LUT에서 score가 좋은 후보를 자동 선택한다.
        - region_id가 주어지면 해당 region 안에서 자동 선택한다.
        - unique_food_xy=True이면 같은 food_xy에 대해 score 최저 primitive만 남긴다.
        - 선택된 primitive_id의 실제 q trajectory는 JSON에서 로드한다.
        """
        rows = self.load_summary_rows()
        if region_id is not None:
            rows = [r for r in rows if int(r["region_id"]) == int(region_id)]
        if not rows:
            return []

        # CSV summary 기준으로 score가 낮은 후보 우선.
        rows = sorted(rows, key=lambda r: (float(r["score"]), int(r["region_id"]), r["primitive_id"]))

        if unique_food_xy:
            best: Dict[Tuple[int, float, float], Dict[str, str]] = {}
            for r in rows:
                key = (int(r["region_id"]), round(float(r["food_x"]), 6), round(float(r["food_y"]), 6))
                if key not in best or float(r["score"]) < float(best[key]["score"]):
                    best[key] = r
            rows = sorted(best.values(), key=lambda r: (float(r["score"]), int(r["region_id"]), r["primitive_id"]))

        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(rows)

        if n_actions is None:
            n_actions = len(rows)
        selected_rows = [rows[i % len(rows)] for i in range(int(n_actions))]

        primitive_by_id = {p.primitive_id: p for p in self.load()}
        selected: List[ScoopPrimitive] = []
        missing = []
        for r in selected_rows:
            pid = r["primitive_id"]
            p = primitive_by_id.get(pid)
            if p is None:
                missing.append(pid)
            else:
                selected.append(p)
        if missing:
            print(f"[WARN] CSV에는 있지만 JSON에 없는 primitive_id {len(missing)}개: {missing[:5]}")
        return selected


class MouthConnectorDatabase:
    """mouth connector LUT 캐시 파일의 저장과 로드를 담당한다."""
    def __init__(self, cfg: SystemConfig):
        """mouth connector JSON 경로를 준비한다."""
        self.cfg = cfg
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.cfg.out_dir / "mouth_connector.json"

    def save(self, connector: MouthConnector):
        """mouth connector와 관련 config snapshot을 JSON으로 저장한다."""
        payload = {
            "config": {
                "mouth_y_range": self.cfg.mouth_y_range,
                "mouth_candidate_z_range": self.cfg.mouth_candidate_z_range,
                "mouth_forward_world": self.cfg.mouth_forward_world,
            },
            "connector": asdict(connector),
        }
        self.json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[SAVE]", self.json_path)

    def load(self) -> MouthConnector:
        """mouth connector JSON에서 connector 본문만 로드한다."""
        if not self.json_path.exists():
            raise FileNotFoundError(f"Mouth connector LUT 파일이 없습니다: {self.json_path}")
        payload = json.loads(self.json_path.read_text(encoding="utf-8"))
        return MouthConnector(**payload["connector"])

    def load_payload(self) -> Tuple[MouthConnector, Dict[str, object]]:
        """connector와 함께 저장 당시 config 블록을 반환한다 (staleness 검증용)."""
        if not self.json_path.exists():
            raise FileNotFoundError(f"Mouth connector LUT 파일이 없습니다: {self.json_path}")
        payload = json.loads(self.json_path.read_text(encoding="utf-8"))
        return MouthConnector(**payload["connector"]), dict(payload.get("config", {}))


def _neutral_config_snapshot(cfg: SystemConfig) -> Dict[str, object]:
    """q_neutral 결과를 결정하는 cfg 항목 스냅샷 (캐시 staleness 판별용).

    후보점 격자(neutral_*), IK 타깃(normal/head_drop), seed(multi_start/seed),
    그리고 tray->world 변환에 영향을 주는 geometry를 포함한다.
    """
    return {
        "neutral_radius": cfg.neutral_radius,
        "neutral_z_min": cfg.neutral_z_min,
        "neutral_z_max": cfg.neutral_z_max,
        "neutral_grid_xy": cfg.neutral_grid_xy,
        "neutral_grid_z": cfg.neutral_grid_z,
        "neutral_normal_weight": cfg.neutral_normal_weight,
        "neutral_head_drop_min": cfg.neutral_head_drop_min,
        "neutral_head_drop_weight": cfg.neutral_head_drop_weight,
        "neutral_position_tol": cfg.neutral_position_tol,
        "neutral_tilt_score_weight": cfg.neutral_tilt_score_weight,
        "neutral_joint_margin_ratio": cfg.neutral_joint_margin_ratio,
        "neutral_center_tray": cfg.neutral_center_tray,           # v12
        "neutral_margin_score_weight": cfg.neutral_margin_score_weight,  # v12
        "multi_start_trials": cfg.multi_start_trials,
        "random_seed": cfg.random_seed,
        "base_x_offset": cfg.base_x_offset,
        "base_y_offset": cfg.base_y_offset,
        "base_z_offset": cfg.base_z_offset,
        "tray_x_length": cfg.tray_x_length,
        "tray_y_length": cfg.tray_y_length,
        "tray_surface_z": cfg.tray_surface_z,
    }


class NeutralDatabase:
    """Persist the computed neutral configuration in ``neutral.json``."""

    def __init__(self, cfg: SystemConfig):
        """neutral connector JSON 경로를 준비한다."""
        self.cfg = cfg
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.cfg.out_dir / "neutral.json"

    def save(self, connector: NeutralConnector):
        """q_neutral과 관련 config snapshot을 neutral.json으로 저장한다."""
        payload = {
            "config": _neutral_config_snapshot(self.cfg),
            "connector": asdict(connector),
        }
        self.json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[SAVE]", self.json_path)

    def load_payload(self) -> Tuple[NeutralConnector, Dict[str, object]]:
        """connector와 함께 저장 당시 config 블록을 반환한다 (staleness 검증용)."""
        if not self.json_path.exists():
            raise FileNotFoundError(f"Neutral LUT 파일이 없습니다: {self.json_path}")
        payload = json.loads(self.json_path.read_text(encoding="utf-8"))
        return NeutralConnector(**payload["connector"]), dict(payload.get("config", {}))


# =============================================================================
# 9. Replay
# =============================================================================

class SequenceRunner:
    """저장된 primitive와 connector를 실제 MuJoCo data/viewer에서 순차 실행한다."""
    def __init__(self, cfg: SystemConfig, robot: RobotModel):
        """실행 중 필요한 로봇 참조, trail buffer, 동적 mouth marker 상태를 초기화한다."""
        self.cfg = cfg
        self.robot = robot
        # Spoon-tip trail buffer used by viewer replays.
        self.trail_positions: List[np.ndarray] = []
        # Current mouth target used for dynamic viewer markers.
        self.current_mouth_pos: Optional[np.ndarray] = None
        self._load_initial_mouth_marker()

    def _load_initial_mouth_marker(self) -> None:
        """저장된 mouth connector가 있으면 초기 viewer marker 위치로 미리 반영한다."""
        try:
            connector, saved_cfg = MouthConnectorDatabase(self.cfg).load_payload()
        except FileNotFoundError:
            return
        if not self._connector_config_matches(saved_cfg):
            return
        self.current_mouth_pos = np.array(connector.mouth_pos, dtype=float)

    def primitive_q_list(self, p: ScoopPrimitive) -> List[np.ndarray]:
        """ScoopPrimitive의 q_pre부터 q_lift까지를 numpy 배열 목록으로 변환한다."""
        return [
            np.array(p.q_pre, dtype=float),
            np.array(p.q_engage, dtype=float),
            np.array(p.q_drag_start, dtype=float),
            np.array(p.q_drag_end, dtype=float),
            np.array(p.q_lift, dtype=float),
        ]

    def sample_neutral_reachable_initial_q(self, rng: np.random.Generator) -> np.ndarray:
        """Neutral로 실제 제어 연결 가능한 초기 random 자세를 샘플한다.

        v5/v6 초기 구현은 전체 joint limit에서 무작위 자세를 뽑았다. 그 자세는
        시각적으로는 random initial이지만, 짧은 position-control 구간으로는
        neutral까지 도달하지 못할 수 있다. 여기서는 q_neutral 주변의 제한된
        후보만 사용해 "random이지만 neutral-reachable"한 시작 자세를 만든다.
        """
        q_ref = self.robot.q_neutral.copy() if self.robot.q_neutral is not None else self.robot.q_center.copy()
        target_world = self._neutral_target_world()
        best_q = q_ref.copy()
        best_dist = float("inf")

        for _ in range(max(1, self.cfg.initial_random_candidates)):
            dq = rng.uniform(-self.cfg.initial_random_radius, self.cfg.initial_random_radius, size=self.robot.model.nq)
            q = q_ref + dq
            q = np.minimum(np.maximum(q, self.robot.q_lower), self.robot.q_upper)
            if not self.robot.is_joint_limit_safe(q):
                continue
            d = mujoco.MjData(self.robot.model)
            self.robot.set_q(d, q)
            if int(d.ncon) > self.cfg.contact_allowed:
                continue
            dist = float(np.linalg.norm(self.robot.tip_pos(d) - target_world))
            if dist <= self.cfg.initial_random_max_tip_distance:
                return q
            if dist < best_dist:
                best_dist = dist
                best_q = q.copy()

        print(
            "[WARN] neutral-reachable random 후보를 엄격 조건으로 찾지 못해 "
            f"가장 가까운 후보를 사용합니다. tip_dist={best_dist*1000:.1f} mm"
        )
        return best_q

    def _step_to(self, d, q_target: np.ndarray):
        """1 스텝: 제어 명령을 1차 low-pass 필터로 업데이트 후 mj_step."""
        if self.robot.model.nu > 0:
            d.ctrl[:self.robot.model.nu] = (
                (1.0 - self.cfg.ctrl_filter_alpha) * d.ctrl[:self.robot.model.nu]
                + self.cfg.ctrl_filter_alpha * q_target[:self.robot.model.nu]
            )
        else:
            d.qpos[:self.robot.model.nq] = q_target[:self.robot.model.nq]
        mujoco.mj_step(self.robot.model, d)

    def _step_to_with_level(self, d, q_target: np.ndarray, level_weight: float = 3.0):
        """carry 구간용 스텝: LPF 이동 + 실시간 spoon level 보정.

        joint-space 보간만으로는 경로 중간 자세의 spoon orientation을 제어할 수 없다.
        매 스텝마다 야코비안 기반 orientation 오차를 계산하고, 위치 야코비안의
        null-space에 level 보정 dq를 추가해 스푼을 수평으로 유지한다.

        null-space 투영: dq_level = N @ (Jr^+ @ level_err)
        → 위치 목표 추종을 방해하지 않으면서 orientation만 보정.
        """
        alpha = self.cfg.ctrl_filter_alpha
        nu = self.robot.model.nu
        nq = self.robot.model.nq

        # 1) 기본 LPF 명령 업데이트
        if nu > 0:
            d.ctrl[:nu] = (1.0 - alpha) * d.ctrl[:nu] + alpha * q_target[:nu]
        else:
            d.qpos[:nq] = q_target[:nq]
            mujoco.mj_step(self.robot.model, d)
            return

        # 2) 현재 스푼 normal 오차 계산
        n_tgt = np.array(self.cfg.world_up, dtype=float)
        n_cur = self.robot.current_body_axis_world(
            d, np.array(self.cfg.spoon_normal_local, dtype=float)
        )
        normal_err = np.cross(n_cur, n_tgt)  # 보정 방향 (외적)
        tilt = float(np.linalg.norm(normal_err))

        if tilt > 0.008:  # ~0.5° 이상 기울어졌을 때만 보정
            Jp, Jr = self.robot.jacobians(d)

            # Jr: (3,nq) rotational jacobian. damped pseudo-inverse
            lam = 0.01
            JrJrt = Jr @ Jr.T
            Jr_pinv = Jr.T @ np.linalg.inv(JrJrt + lam * np.eye(3))  # (nq,3)

            # Jp null-space projector: N = I - Jp^+ @ Jp
            lam_p = 0.001
            JpJpt = Jp @ Jp.T
            Jp_pinv = Jp.T @ np.linalg.inv(JpJpt + lam_p * np.eye(3))  # (nq,3)
            N = np.eye(nq) - Jp_pinv @ Jp  # (nq,nq)

            # level 보정 dq: null-space에 투영
            dq_level = N @ (Jr_pinv @ (level_weight * normal_err))
            dq_level = np.clip(dq_level, -0.03, 0.03)

            # ctrl에 추가 후 joint limit 클램프
            d.ctrl[:nu] += dq_level[:nu]
            for i in range(nu):
                d.ctrl[i] = float(np.clip(
                    d.ctrl[i], self.robot.q_lower[i], self.robot.q_upper[i]
                ))

        mujoco.mj_step(self.robot.model, d)

    def _render_trail(self, v) -> None:
        """
        viewer.user_scn에 self.trail_positions를 작은 구(sphere)로 렌더링.
        오래된 점일수록 투명하게, 최근 점일수록 불투명하게 표시.
        frames_per_segment=240 기준으로 4프레임마다 1개 기록 → 세그먼트당 ~60점.
        """
        if not hasattr(v, 'user_scn'):
            return
        scn = v.user_scn
        pts = self.trail_positions
        if not pts:
            scn.ngeom = 0
            return

        max_g = scn.maxgeom
        n = len(pts)

        # 최대 max_g개를 균등 샘플링
        if n > max_g:
            idxs = np.linspace(0, n - 1, max_g, dtype=int)
            render_pts = [pts[i] for i in idxs]
        else:
            render_pts = pts

        scn.ngeom = 0
        mat9 = np.eye(3, dtype=np.float64).flatten()
        nr = len(render_pts)

        for k, pos in enumerate(render_pts):
            if scn.ngeom >= max_g:
                break
            # 최근 점은 밝은 오렌지, 오래된 점은 반투명 노랑
            t = k / max(nr - 1, 1)
            rgba = np.array([1.0, 0.3 + 0.4 * t, 0.0, 0.3 + 0.7 * t], dtype=np.float32)
            size = np.array([0.005 + 0.003 * t, 0.0, 0.0], dtype=np.float64)
            mujoco.mjv_initGeom(
                scn.geoms[scn.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                size,
                np.array(pos, dtype=np.float64),
                mat9,
                rgba,
            )
            scn.ngeom += 1

        # Draw dynamic mouth and neutral target markers.
        self._render_dynamic_markers(scn, mat9, max_g)

    def _render_dynamic_markers(self, scn, mat9, max_g):
        """Neutral target과 현재 선택된 Mouth target을 viewer scn에 그린다.

        XML에서 mouth_sphere 등 정적 마커를 제거하는 대신, 코드가 실시간 계산한
        target 좌표를 렌더링한다. 둘은 색으로 구분: Neutral=청록, Mouth=핑크.
        """
        # Neutral marker (cyan)
        if scn.ngeom < max_g:
            nt = self._neutral_target_world()
            mujoco.mjv_initGeom(
                scn.geoms[scn.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.012, 0.0, 0.0], dtype=np.float64),
                np.array(nt, dtype=np.float64),
                mat9,
                np.array([0.20, 0.85, 0.85, 0.85], dtype=np.float32),
            )
            scn.ngeom += 1
        # Mouth marker (pink) — 현재 sequence에서 선택된 위치만
        if self.current_mouth_pos is not None and scn.ngeom < max_g:
            mujoco.mjv_initGeom(
                scn.geoms[scn.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.020, 0.0, 0.0], dtype=np.float64),
                np.array(self.current_mouth_pos, dtype=np.float64),
                mat9,
                np.array([1.00, 0.35, 0.55, 0.85], dtype=np.float32),
            )
            scn.ngeom += 1

    def _run_segments(self, d, q_list: List[np.ndarray], v=None, labels: Optional[List[str]] = None) -> StepResult:
        """q_list keyframe 간을 smoothstep5 보간하며 시뮬레이션.
        viewer가 있을 때 4프레임마다 spoon_tip 위치를 trail_positions에 기록한다.

        중요: 여기서는 qpos를 직접 set하지 않고 actuator command를 연속적으로
        업데이트한다. 따라서 Pre-scoop → Engage → Drag start → Drag end → Lift가
        한 trajectory로 이어지고 순간이동이 발생하지 않는다.
        """
        _trail_stride = 4
        _frame_ctr = 0
        if labels is None:
            labels = [f"segment_{i+1}" for i in range(len(q_list) - 1)]
        target_normal = np.array(self.cfg.world_up, dtype=float)
        target_forward = np.array(self.cfg.scoop_drag_direction_world, dtype=float)
        for seg_idx, (q0, q1) in enumerate(zip(q_list[:-1], q_list[1:])):
            print(f"[STEP {seg_idx + 1}/{len(q_list) - 1}] {labels[seg_idx]}")
            seg_contacts = 0
            seg_max_tilt = 0.0
            seg_min_head_drop = float("inf")
            seg_max_head_drop = float("-inf")
            for i in range(self.cfg.frames_per_segment):
                s = smoothstep5(i / max(self.cfg.frames_per_segment - 1, 1))
                q_target = (1.0 - s) * q0 + s * q1
                self._step_to(d, q_target)
                head_drop = self.robot.spoon_head_drop(d)
                if head_drop < seg_min_head_drop:
                    seg_min_head_drop = head_drop
                if head_drop > seg_max_head_drop:
                    seg_max_head_drop = head_drop
                # v5: monitor contacts & tilt during execution
                if self.cfg.monitor_contacts_during_replay and int(d.ncon) > 0:
                    seg_contacts = max(seg_contacts, int(d.ncon))
                if self.cfg.monitor_tilt_during_replay:
                    tilt, _fwd, _du, _df = self.robot.orientation_errors(d, target_normal, target_forward)
                    if tilt > seg_max_tilt:
                        seg_max_tilt = tilt
                    if self.cfg.abort_on_scoop_level_tilt and tilt > self.cfg.tilt_warn_threshold:
                        result = StepResult(
                            label=labels[seg_idx],
                            ok=False,
                            reason="level_tilt_limit_exceeded",
                            actual_pos=tuple(self.robot.tip_pos(d).tolist()),
                            tilt_error=float(tilt),
                            contact=int(d.ncon),
                        )
                        print(result.summary())
                        return result
                if v is not None:
                    _frame_ctr += 1
                    if _frame_ctr % _trail_stride == 0:
                        self.trail_positions.append(self.robot.tip_pos(d).copy())
                    self._render_trail(v)
                    v.sync()
                    time.sleep(self.robot.model.opt.timestep)
            # 세그먼트 종료 시 요약 출력 (contact > 0 또는 큰 tilt면 경고)
            warn_bits = []
            if seg_contacts > 0:
                warn_bits.append(f"contacts={seg_contacts}")
            if seg_max_tilt > self.cfg.tilt_warn_threshold:
                warn_bits.append(f"level_tilt_diag={np.degrees(seg_max_tilt):.1f}deg")
            if warn_bits:
                print(f"  [WARN seg {seg_idx+1}] " + ", ".join(warn_bits))
            if math.isfinite(seg_min_head_drop):
                # Print both head-drop and pitch for posture diagnostics.
                end_pitch = self.robot.spoon_pitch_deg(d)
                end_tilt_deg = float(np.degrees(seg_max_tilt))
                max_pitch = float(np.degrees(np.arcsin(np.clip(
                    seg_max_head_drop / max(np.linalg.norm(np.array(self.cfg.spoon_head_local)), 1e-9), -1.0, 1.0))))
                print(
                    f"  [POSE seg {seg_idx+1}] head_drop min={seg_min_head_drop*1000:.1f}mm "
                    f"max={seg_max_head_drop*1000:.1f}mm | pitch max={max_pitch:.1f}deg "
                    f"end={end_pitch:.1f}deg | end_tilt={np.degrees(self.robot.orientation_errors(d, target_normal, target_forward)[0]):.1f}deg"
                )
        result = StepResult(
            label="RUN_SEGMENTS",
            ok=True,
            reason="completed",
            actual_pos=tuple(self.robot.tip_pos(d).tolist()),
            tilt_error=float(seg_max_tilt),
            head_drop=float(seg_min_head_drop) if math.isfinite(seg_min_head_drop) else None,
            contact=int(d.ncon),
        )
        print(result.summary())
        return result

    def move_to_q(
        self,
        q_target: np.ndarray,
        v=None,
        frames: Optional[int] = None,
        label: str = "MOVE",
        q_tolerance: Optional[float] = None,
        min_head_drop: Optional[float] = None,
        keep_level: bool = False,
        level_weight: float = 3.0,
    ) -> StepResult:
        """현재 자세에서 q_target으로 smoothstep 보간 이동.

        run_region/run_all_regions에서 primitive 시작 자세(q_pre)로 갈 때
        순간 이동(set_q)을 피하기 위한 연결 동작이다.
        keep_level=True면 이동 중 야코비안 null-space level 보정을 적용한다 (carry 구간용).
        """
        d = self.robot.data
        if frames is None:
            frames = max(960, 4 * self.cfg.frames_per_segment)
        if q_tolerance is None:
            q_tolerance = self.cfg.move_q_tolerance
        q_start = d.qpos[:self.robot.model.nq].copy()
        q_target = np.array(q_target, dtype=float)
        trail_stride = 4
        # BUGFIX: 각 lambda를 괄호로 감싼다. 괄호가 없으면 ternary가 첫 lambda의 본문에
        # 흡수되어 식 전체가 하나의 lambda가 되고, keep_level=False일 때 step_fn(d,q)가
        # 안쪽 lambda만 만들어 버린 채 _step_to를 호출하지 않는 no-op이 된다(=로봇 정지).
        step_fn = (
            (lambda d, q: self._step_to_with_level(d, q, level_weight=level_weight))
            if keep_level else
            (lambda d, q: self._step_to(d, q))
        )
        for i in range(frames):
            s = smoothstep5(i / max(frames - 1, 1))
            q_interp = (1.0 - s) * q_start + s * q_target
            step_fn(d, q_interp)
            if v is not None:
                if i % trail_stride == 0:
                    self.trail_positions.append(self.robot.tip_pos(d).copy())
                self._render_trail(v)
                v.sync()
                time.sleep(self.robot.model.opt.timestep)
        q_err = float(np.max(np.abs(d.qpos[:self.robot.model.nq] - q_target[:self.robot.model.nq])))
        extra_used = 0
        if q_err > q_tolerance:
            for j in range(self.cfg.move_max_extra_frames):
                step_fn(d, q_target)
                if v is not None:
                    if j % trail_stride == 0:
                        self.trail_positions.append(self.robot.tip_pos(d).copy())
                    self._render_trail(v)
                    v.sync()
                    time.sleep(self.robot.model.opt.timestep)
                q_err = float(np.max(np.abs(d.qpos[:self.robot.model.nq] - q_target[:self.robot.model.nq])))
                if q_err <= q_tolerance:
                    break
            extra_used = j + 1 if self.cfg.move_max_extra_frames > 0 else 0
        actual_head_drop = float(self.robot.spoon_head_drop(d))
        head_drop_ok = min_head_drop is None or actual_head_drop >= min_head_drop
        ok = q_err <= q_tolerance and head_drop_ok
        result = StepResult(
            label=label,
            ok=ok,
            reason="completed" if ok else ("head_drop_error" if not head_drop_ok else "q_tracking_error"),
            actual_pos=tuple(self.robot.tip_pos(d).tolist()),
            q_error=q_err,
            head_drop=actual_head_drop,
            contact=int(d.ncon),
            extra_frames=extra_used,
        )
        print(result.summary())
        return result

    def make_mouth_target(self, pos: Optional[Tuple[float, float, float]] = None) -> PoseTarget:
        """Mouth delivery 목표 pose.

        - 위치는 인자 pos 우선, 없으면 cfg.default_mouth_pos_world를 사용한다.
        - spoon level은 유지하되, yaw/forward는 초기 안정성을 위해 기본적으로 강제하지 않는다.
          필요하면 cfg.mouth_forward_weight를 0.02~0.05로 올려 테스트한다.
        """
        if pos is None:
            pos = self.cfg.default_mouth_pos_world
        return PoseTarget(
            name="mouth",
            pos=tuple(pos),
            forward=tuple(self.cfg.mouth_forward_world),
            normal=tuple(self.cfg.world_up),
            normal_weight=float(self.cfg.mouth_normal_weight),
            forward_weight=float(self.cfg.mouth_forward_weight),
            normal_hard=True,
            forward_hard=True,
        )

    # v9~: 입 위치 후보군 (x=0, y ∈ [0.10, 0.25], z ∈ [0.3, 0.6])에 대해 multi-start IK로
    # joint margin까지 만족하는 후보를 탐색한다. v10에서는 build_mouth_connector가 이 함수를
    # offline 호출하여 결과를 mouth_connector.json에 캐싱하므로 런타임 호출은 불필요.
    def solve_mouth_q_multi(
        self,
        seed_q: np.ndarray,
        ik: Optional[IKSolver] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[bool, Optional[np.ndarray], Dict[str, float], Optional[Tuple[float, float, float]]]:
        """입 위치 후보군 (x=mouth_x, y ∈ mouth_y_range, z ∈ mouth_candidate_z_range)을
        조밀하게 샘플해 multi-start IK를 풀고, 가장 안전한 후보를 반환한다.

        Returns:
            (ok, q_best, metrics, pos_best)
            ok=False면 q_best=None.
        """
        if ik is None:
            ik = IKSolver(self.cfg, self.robot)
        if rng is None:
            rng = np.random.default_rng(self.cfg.random_seed)

        mx = float(self.cfg.mouth_x)
        y_lo, y_hi = self.cfg.mouth_y_range
        y_step = max(1e-4, float(self.cfg.mouth_y_step))
        y_n = max(2, int(round((y_hi - y_lo) / y_step)) + 1)
        ys = np.linspace(y_lo, y_hi, y_n)
        z_lo, z_hi = self.cfg.mouth_candidate_z_range
        step = max(1e-4, float(self.cfg.mouth_candidate_z_step))
        n = max(2, int(round((z_hi - z_lo) / step)) + 1)
        zs = np.linspace(z_lo, z_hi, n)

        best_q: Optional[np.ndarray] = None
        best_metrics: Dict[str, float] = {}
        best_pos: Optional[Tuple[float, float, float]] = None
        best_score = float("inf")
        best_ok = False

        # Seed pool: q_lift seed + center + random samples (multi-start)
        seeds: List[np.ndarray] = [seed_q.copy(), self.robot.q_center.copy()]
        if self.robot.q_neutral is not None:
            seeds.append(self.robot.q_neutral.copy())
        for _ in range(self.cfg.mouth_multi_start_seeds):
            seeds.append(self.robot.sample_random_q(rng))

        attempts = 0
        for y in ys:
            for z in zs:
                target = self.make_mouth_target(pos=(float(mx), float(y), float(z)))
                for s_idx, q_seed in enumerate(seeds):
                    attempts += 1
                    _, q_sol, metrics = ik.solve_pose(target, q_seed, posture_ref=q_seed)
                    ok, val_metrics = ik.validate_q_for_mouth(q_sol, target)
                    pos_err = val_metrics.get("pos_error", float("inf"))
                    tilt_err = val_metrics.get("tilt_error", float("inf"))
                    fwd_err = val_metrics.get("forward_error", float("inf"))
                    joint_ok = val_metrics.get("joint_limit_ok", 0.0) > 0.0
                    min_margin = val_metrics.get("min_joint_margin_ratio", 0.0)
                    motion = float(np.linalg.norm(q_sol - seed_q[:len(q_sol)]))
                    y_center = 0.5 * (float(y_lo) + float(y_hi))
                    y_penalty = abs(float(y) - y_center)
                    margin_penalty = max(0.0, self.cfg.mouth_joint_limit_margin_ratio - min_margin)
                    score = (
                        1000.0 * pos_err
                        + 10.0 * tilt_err
                        + 2.0 * fwd_err
                        + 0.1 * y_penalty
                        + 0.02 * motion
                        + 200.0 * margin_penalty
                        + (0.0 if joint_ok else 100.0)
                    )
                    # Track best by validity first, then combined safety/accuracy score.
                    better = False
                    if ok and (not best_ok):
                        better = True
                    elif ok == best_ok and score < best_score:
                        better = True
                    if better:
                        best_ok = ok
                        best_q = q_sol.copy()
                        best_metrics = val_metrics
                        best_pos = (float(mx), float(y), float(z))
                        best_score = score

        if best_q is None:
            return False, None, {"attempts": float(attempts)}, None
        best_metrics["attempts"] = float(attempts)
        return best_ok, best_q, best_metrics, best_pos

    def build_mouth_connector(
        self,
        seed_q: np.ndarray,
        ik: Optional[IKSolver] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[bool, Optional[MouthConnector]]:
        """Build a neutral-to-mouth connector and cache-ready pre-approach pose."""
        if ik is None:
            ik = IKSolver(self.cfg, self.robot)
        if rng is None:
            rng = np.random.default_rng(self.cfg.random_seed)

        ok, q_delivery, metrics, mouth_pos = self.solve_mouth_q_multi(seed_q=seed_q, ik=ik, rng=rng)
        if not ok or q_delivery is None or mouth_pos is None:
            print(f"[MOUTH CONNECTOR FAIL] delivery IK failed metrics={metrics}, pos={mouth_pos}")
            return False, None

        mouth_arr = np.array(mouth_pos, dtype=float)
        forward = normalize(np.array(self.cfg.mouth_forward_world, dtype=float))
        pre_pos = mouth_arr - float(self.cfg.mouth_pre_approach_distance) * forward
        pre_target = self.make_mouth_target(pos=tuple(pre_pos.tolist()))

        seed_candidates = [
            seed_q.copy(),
            self.robot.q_neutral.copy() if self.robot.q_neutral is not None else self.robot.q_center.copy(),
            q_delivery.copy(),
        ]
        best_pre_q: Optional[np.ndarray] = None
        best_pre_metrics: Dict[str, float] = {}
        best_pre_score = float("inf")
        best_pre_ok = False
        for q_seed in seed_candidates:
            _, q_pre_candidate, _solve_metrics = ik.solve_pose(pre_target, q_seed, posture_ref=q_seed)
            pre_ok, pre_metrics = ik.validate_q_for_mouth(q_pre_candidate, pre_target)
            score = (
                1000.0 * pre_metrics.get("pos_error", float("inf"))
                + 20.0 * pre_metrics.get("tilt_error", float("inf"))
                + 5.0 * pre_metrics.get("forward_error", float("inf"))
                + 0.05 * float(np.linalg.norm(q_pre_candidate - seed_q[:len(q_pre_candidate)]))
                + (0.0 if pre_ok else 100.0)
            )
            if score < best_pre_score:
                best_pre_score = score
                best_pre_q = q_pre_candidate.copy()
                best_pre_metrics = pre_metrics
                best_pre_ok = bool(pre_ok)

        # F: 유효한 pre 자세를 못 찾으면 의미 없는 midpoint 대신 q_delivery로 직행(안전 폴백).
        if best_pre_q is None or not best_pre_ok:
            print(
                "[MOUTH CONNECTOR WARN] 유효한 pre-approach 자세를 찾지 못해 "
                "q_pre=q_delivery로 폴백합니다 (G1이 delivery로 직행)."
            )
            best_pre_q = np.array(q_delivery, dtype=float)
            pre_pos = mouth_arr.copy()
            best_pre_metrics = {
                "pos_error": float(metrics.get("pos_error", float("nan"))),
                "tilt_error": float(metrics.get("tilt_error", float("nan"))),
                "forward_error": float(metrics.get("forward_error", float("nan"))),
            }

        connector = MouthConnector(
            connector_id="mouth_connector_v10",
            mouth_pos=tuple(float(x) for x in mouth_pos),
            pre_pos=tuple(float(x) for x in pre_pos.tolist()),
            retreat_pos=tuple(float(x) for x in pre_pos.tolist()),
            q_pre=best_pre_q.tolist(),
            q_delivery=q_delivery.tolist(),
            q_retreat=best_pre_q.tolist(),
            pos_error=float(metrics.get("pos_error", float("nan"))),
            tilt_error=float(metrics.get("tilt_error", float("nan"))),
            forward_error=float(metrics.get("forward_error", float("nan"))),
            min_joint_margin_ratio=float(metrics.get("min_joint_margin_ratio", float("nan"))),
            attempts=int(metrics.get("attempts", 0)),
        )
        print(
            "[MOUTH CONNECTOR OK] "
            f"pre_pos={tuple(round(x, 4) for x in connector.pre_pos)}, "
            f"delivery={tuple(round(x, 4) for x in connector.mouth_pos)}, "
            f"delivery_err={connector.pos_error*1000:.1f}mm, "
            f"pre_err={best_pre_metrics.get('pos_error', float('nan'))*1000:.1f}mm"
        )
        return True, connector

    def _connector_config_matches(self, saved_cfg: Dict[str, object]) -> bool:
        """저장 당시 mouth config가 현재 cfg와 같은지 비교 (stale 판별)."""
        def _close(a, b) -> bool:
            try:
                aa = np.array(a, dtype=float).ravel()
                bb = np.array(b, dtype=float).ravel()
                return aa.shape == bb.shape and bool(np.allclose(aa, bb, atol=1e-9))
            except (TypeError, ValueError):
                return a == b

        checks = {
            "mouth_y_range": self.cfg.mouth_y_range,
            "mouth_candidate_z_range": self.cfg.mouth_candidate_z_range,
            "mouth_forward_world": self.cfg.mouth_forward_world,
        }
        for key, current in checks.items():
            if key not in saved_cfg or not _close(saved_cfg[key], current):
                return False
        return True

    def _connector_valid(self, connector: MouthConnector, ik: IKSolver) -> bool:
        """저장된 q_delivery가 현재 모델에서 joint limit / mouth FK 검증을 통과하는지 확인."""
        q_delivery = np.array(connector.q_delivery, dtype=float)
        if not self.robot.is_joint_limit_safe_with_margin(q_delivery, self.cfg.mouth_joint_limit_margin_ratio):
            return False
        delivery_target = self.make_mouth_target(pos=tuple(connector.mouth_pos))
        ok, _ = ik.validate_q_for_mouth(q_delivery, delivery_target)
        return bool(ok)

    def get_mouth_connector(
        self,
        seed_q: np.ndarray,
        ik: Optional[IKSolver] = None,
    ) -> Optional[MouthConnector]:
        """캐시된 mouth connector를 로드/검증하고 필요하면 새로 계산해 반환한다."""
        if ik is None:
            ik = IKSolver(self.cfg, self.robot)
        db = MouthConnectorDatabase(self.cfg)

        # D: use_mouth_lut=False면 캐시를 무시하고 항상 온라인 재계산.
        if self.cfg.use_mouth_lut:
            try:
                connector, saved_cfg = db.load_payload()
            except FileNotFoundError:
                pass
            else:
                # C: config stale 또는 q_delivery validity 실패 시 rebuild.
                stale = not self._connector_config_matches(saved_cfg)
                invalid = not self._connector_valid(connector, ik)
                if stale or invalid:
                    reason = "config가 현재 cfg와 불일치" if stale else "q_delivery validation 실패"
                    print(f"[MOUTH CONNECTOR REBUILD] 캐시 {reason} → 재계산합니다.")
                else:
                    print(
                        "[MOUTH CONNECTOR LOAD] "
                        f"pre={tuple(round(x, 4) for x in connector.pre_pos)}, "
                        f"delivery={tuple(round(x, 4) for x in connector.mouth_pos)}"
                    )
                    return connector
        else:
            print("[MOUTH CONNECTOR] use_mouth_lut=False → 캐시 무시하고 온라인 재계산합니다.")

        ok, connector = self.build_mouth_connector(
            seed_q=seed_q,
            ik=ik,
            rng=np.random.default_rng(self.cfg.random_seed),
        )
        if ok and connector is not None:
            db.save(connector)
            return connector
        return None

    def replay_mouth_connector(self, connector: MouthConnector, v=None) -> StepResult:
        """pre-approach와 delivery 관절 경로를 따라 mouth target까지 이동한다."""
        self.current_mouth_pos = np.array(connector.mouth_pos, dtype=float)
        q_pre = np.array(connector.q_pre, dtype=float)
        q_delivery = np.array(connector.q_delivery, dtype=float)

        pre_result = self.move_to_q(
            q_pre,
            v=v,
            frames=max(self.cfg.mouth_approach_frames, 4 * self.cfg.frames_per_segment),
            label="G1_TO_MOUTH_PRE",
            q_tolerance=self.cfg.move_q_tolerance,
            keep_level=False,
        )
        if not pre_result.ok:
            return pre_result

        delivery_result = self.move_to_q(
            q_delivery,
            v=v,
            frames=max(self.cfg.mouth_approach_frames, 4 * self.cfg.frames_per_segment),
            label="G2_TO_MOUTH_DELIVERY",
            q_tolerance=self.cfg.mouth_q_tolerance,
            keep_level=False,
        )

        tip_now = self.robot.tip_pos(self.robot.data)
        err = float(np.linalg.norm(tip_now - self.current_mouth_pos))
        extra_used = 0
        if err > self.cfg.mouth_position_tol:
            for j in range(self.cfg.mouth_max_extra_frames):
                self._step_to(self.robot.data, q_delivery)
                if v is not None:
                    if j % 4 == 0:
                        self.trail_positions.append(self.robot.tip_pos(self.robot.data).copy())
                    self._render_trail(v)
                    v.sync()
                    time.sleep(self.robot.model.opt.timestep)
                tip_now = self.robot.tip_pos(self.robot.data)
                err = float(np.linalg.norm(tip_now - self.current_mouth_pos))
                if err <= self.cfg.mouth_position_tol:
                    break
            extra_used = j + 1 if self.cfg.mouth_max_extra_frames > 0 else 0

        reached = err <= self.cfg.mouth_position_tol
        if reached:
            print(
                f"[MOUTH REACHED] tip={np.round(tip_now, 4).tolist()}, "
                f"target={np.round(self.current_mouth_pos, 4).tolist()}, err={err*1000:.1f} mm"
            )
        else:
            print(
                f"[MOUTH APPROXIMATE] tip={np.round(tip_now, 4).tolist()}, "
                f"target={np.round(self.current_mouth_pos, 4).tolist()}, err={err*1000:.1f} mm "
                f"(> tol {self.cfg.mouth_position_tol*1000:.0f} mm)"
            )

        result = StepResult(
            label="G_TO_MOUTH_CONNECTOR",
            ok=bool(delivery_result.ok and reached),
            reason="reached" if delivery_result.ok and reached else (
                "mouth_position_error" if not reached else "q_tracking_error"
            ),
            target_pos=tuple(self.current_mouth_pos.tolist()),
            actual_pos=tuple(tip_now.tolist()),
            pos_error=err,
            q_error=delivery_result.q_error,
            tilt_error=float(connector.tilt_error),
            head_drop=float(self.robot.spoon_head_drop(self.robot.data)),
            extra_frames=extra_used,
        )
        print(result.summary())
        return result

    def solve_mouth_q(self, primitive: ScoopPrimitive, ik: Optional[IKSolver] = None) -> Tuple[bool, np.ndarray, Dict[str, float]]:
        """q_lift에서 시작해 mouth pose로 가는 single-target IK 폴백.

        v10에서는 build_mouth_connector/get_mouth_connector를 통한 캐시 경로가
        주 경로이므로, 이 함수는 test_run 등 단독 검증 용도로만 호출된다.
        """
        if ik is None:
            ik = IKSolver(self.cfg, self.robot)
        q_lift = np.array(primitive.q_lift, dtype=float)
        target = self.make_mouth_target()
        ok, q_mouth, metrics = ik.solve_pose(target, q_lift, posture_ref=q_lift)
        if ok:
            print(
                "[MOUTH IK OK] "
                f"pos_error={metrics.get('pos_error', float('nan')):.4f}, "
                f"tilt={metrics.get('tilt_error', float('nan')):.4f}, "
                f"fwd={metrics.get('forward_error', float('nan')):.4f}, "
                f"sigma={metrics.get('sigma_min', float('nan')):.5f}, "
                f"cond={metrics.get('condition', float('nan')):.1f}"
            )
            return True, q_mouth, metrics

        # 엄격한 scoop tolerance에는 못 미쳤지만 mouth용 완화 기준은 만족하는지 확인
        relaxed_ok, relaxed_metrics = ik.validate_q_for_mouth(q_mouth, target)
        if relaxed_ok:
            print(
                "[MOUTH IK OK (relaxed)] "
                f"pos_error={relaxed_metrics.get('pos_error', float('nan')):.4f}, "
                f"tilt={relaxed_metrics.get('tilt_error', float('nan')):.4f}, "
                f"fwd={relaxed_metrics.get('forward_error', float('nan')):.4f}"
            )
            return True, q_mouth, relaxed_metrics

        print(
            "[MOUTH IK FAIL] Lift -> Mouth connector를 찾지 못했습니다. "
            "해당 primitive는 Lift 이후 Neutral로 복귀합니다. "
            f"metrics={metrics}"
        )
        return False, q_lift, metrics

    def pause(self, v=None, frames: Optional[int] = None, label: str = "PAUSE"):
        """현재 actuator command를 유지하며 지정 frame만큼 정지."""
        if frames is None:
            frames = self.cfg.mouth_pause_frames
        d = self.robot.data
        q_hold = d.qpos[:self.robot.model.nq].copy()
        for i in range(frames):
            self._step_to(d, q_hold)
            if v is not None:
                if i % 4 == 0:
                    self.trail_positions.append(self.robot.tip_pos(d).copy())
                self._render_trail(v)
                v.sync()
                time.sleep(self.robot.model.opt.timestep)
        print(f"[{label} DONE]")

    def replay_full_sequence(
        self,
        primitive: ScoopPrimitive,
        v=None,
        approach_frames: Optional[int] = None,
        ik: Optional[IKSolver] = None,
        neutral_after_lift: bool = True,
    ) -> StepResult:
        """제어 코드 계획서 기준 1회 scoop 전체 sequence.

        (호출 직전 자세) -> Pre-scoop -> Engage -> Drag start -> Drag end (-X scoop) -> Lift
        -> [Neutral] -> Mouth(pre->delivery) -> Neutral

        The cached mouth connector avoids online IK during viewer replay.
        """
        q_list = self.primitive_q_list(primitive)

        # A. (현재 자세, 보통 Neutral) -> Pre-scoop
        approach_result = self.move_to_q(
            q_list[0],
            v=v,
            frames=approach_frames,
            label="A_NEUTRAL_TO_PRE_SCOOP",
            q_tolerance=self.cfg.pre_move_q_tolerance,
            min_head_drop=self.cfg.pre_runtime_head_drop_min,
        )
        if not approach_result.ok:
            print("[ABORT] pre-scoop 도달 실패. Neutral retreat 후 sequence를 중단합니다.")
            self.replay_neutral(v=v, frames=max(960, 4 * self.cfg.frames_per_segment))
            return approach_result

        # B~E. Pre -> Engage -> Drag_start -> Drag_end(-X) -> Lift
        labels = [
            "B_PRE_SCOOP_TO_ENGAGE",
            "C_ENGAGE_TO_DRAG_START",
            "D_DRAG_START_TO_DRAG_END_-X_SCOOP",
            "E_DRAG_END_TO_LIFT",
        ]
        scoop_result = self._run_segments(self.robot.data, q_list, v, labels=labels)
        if not scoop_result.ok:
            print("[ABORT] scoop replay 중 안전 한계 위반. Neutral retreat 후 sequence를 중단합니다.")
            self.replay_neutral(v=v, frames=max(960, 4 * self.cfg.frames_per_segment))
            return scoop_result

        # F. Lift -> Neutral — carry 구간: level 유지 보정 ON
        if neutral_after_lift:
            print("[STEP F] LIFT_TO_NEUTRAL")
            # Endpoints are already level IK solutions, so this long carry uses
            # plain joint interpolation for convergence.
            neutral_result = self.replay_neutral(
                v=v, frames=max(960, 4 * self.cfg.frames_per_segment), keep_level=False
            )
            if not neutral_result.ok:
                print("[ABORT] Lift 후 Neutral 도달 실패. Mouth 단계로 진행하지 않습니다.")
                return neutral_result
            seed_for_mouth = self.robot.data.qpos[:self.robot.model.nq].copy()
        else:
            seed_for_mouth = np.array(primitive.q_lift, dtype=float)

        # G. Neutral -> Mouth via cached v10 connector LUT.
        if ik is None:
            ik = IKSolver(self.cfg, self.robot)
        connector = self.get_mouth_connector(seed_q=seed_for_mouth, ik=ik)
        if connector is None:
            result = StepResult(label="G_TO_MOUTH_CONNECTOR", ok=False, reason="mouth_connector_failed")
            print(result.summary())
            return result
        mouth_result = self.replay_mouth_connector(connector, v=v)
        if not mouth_result.ok:
            print("[ABORT] Mouth connector failed. Mouth hold 없이 Neutral retreat합니다.")
            neutral_result = self.replay_neutral(v=v, frames=max(960, 4 * self.cfg.frames_per_segment))
            return mouth_result if not neutral_result.ok else mouth_result
        self.pause(v=v, frames=self.cfg.mouth_pause_frames, label="H_MOUTH_HOLD")

        # I. Mouth -> Neutral — carry 구간: level 유지 보정 ON
        print("[STEP I] MOUTH_TO_NEUTRAL")
        final_neutral_result = self.replay_neutral(
            v=v, frames=max(960, 4 * self.cfg.frames_per_segment), keep_level=False
        )
        print("[FULL SEQUENCE DONE]", primitive.primitive_id)
        return final_neutral_result

    def replay_continuous(self, primitive: ScoopPrimitive, v=None, approach_frames: Optional[int] = None) -> StepResult:
        """현재 자세에서 q_pre로 부드럽게 연결한 뒤 primitive를 재생한다.

        기존 replay()는 단독 검증용으로 q_pre를 즉시 set_q한다.
        이 함수는 여러 동작을 이어서 볼 때 관절각 점프가 생기지 않도록 한다.
        """
        q_list = self.primitive_q_list(primitive)
        approach_result = self.move_to_q(q_list[0], v=v, frames=approach_frames, label="0_NEUTRAL_TO_PRE_SCOOP")
        if not approach_result.ok:
            print("[ABORT] pre-scoop 도달 실패. Neutral retreat합니다.")
            self.replay_neutral(v=v, frames=max(960, 4 * self.cfg.frames_per_segment))
            return approach_result
        labels = [
            "1_PRE_SCOOP_TO_ENGAGE",
            "2_ENGAGE_TO_DRAG_START",
            "3_DRAG_START_TO_DRAG_END",
            "4_DRAG_END_TO_LIFT",
        ]
        result = self._run_segments(self.robot.data, q_list, v, labels=labels)
        if not result.ok:
            print("[ABORT] replay 중 안전 한계 위반. Neutral retreat합니다.")
            self.replay_neutral(v=v, frames=max(960, 4 * self.cfg.frames_per_segment))
            return result
        print("[REPLAY CONTINUOUS DONE]", primitive.primitive_id)
        return result

    def replay(self, primitive: ScoopPrimitive, viewer: bool = False, v=None):
        """
        저장된 primitive를 재생한다.

        Args:
            viewer: True이면 새 viewer 창을 열어 시각화.
            v:      이미 열린 viewer 인스턴스. 제공되면 해당 창을 공유 (run_region 연속 재생용).
                    v가 주어지면 viewer 인자는 무시된다.
        """
        q_list = self.primitive_q_list(primitive)
        d = self.robot.data
        self.robot.set_q(d, q_list[0])
        if self.robot.model.nu > 0:
            d.ctrl[:self.robot.model.nu] = q_list[0][:self.robot.model.nu]

        if v is not None:
            # 외부에서 viewer를 공유받은 경우 (run_region 등)
            result = self._run_segments(d, q_list, v, labels=["pre_to_engage", "engage_to_drag_start", "drag_start_to_drag_end", "drag_end_to_lift"])
        elif viewer:
            with mujoco.viewer.launch_passive(self.robot.model, d) as new_v:
                result = self._run_segments(d, q_list, new_v, labels=["pre_to_engage", "engage_to_drag_start", "drag_start_to_drag_end", "drag_end_to_lift"])
        else:
            result = self._run_segments(d, q_list, None, labels=["pre_to_engage", "engage_to_drag_start", "drag_start_to_drag_end", "drag_end_to_lift"])

        if not result.ok:
            print("[ABORT] replay 중 안전 한계 위반. Neutral retreat합니다.")
            self.replay_neutral(v=v, frames=max(960, 4 * self.cfg.frames_per_segment))
            return
        print("[REPLAY DONE]", primitive.primitive_id)

    def replay_neutral(self, v=None, frames: int = 120, pause: bool = True, keep_level: bool = False) -> StepResult:
        """현재 자세에서 Cartesian neutral_pos_tray에 대응하는 q_neutral로 부드럽게 복귀.

        v5 변경:
          - 보간 종료 후 실제 tip 위치 vs target 위치를 측정하여 출력.
          - 위치 오차가 cfg.neutral_position_tol을 초과하면 cfg.neutral_max_extra_frames
            만큼 추가로 hold하면서 actuator가 수렴할 시간을 준다.
          - 끝까지 수렴 못하면 [NEUTRAL APPROXIMATE] 경고로 명확히 표시한다.
        """
        d = self.robot.data
        q_start = d.qpos[:self.robot.model.nq].copy()
        if self.robot.q_neutral is not None:
            q_neutral = self.robot.q_neutral.copy()
        else:
            q_neutral = self.robot.q_center.copy()

        target_world = self._neutral_target_world()
        _trail_stride = 4
        _frame_ctr = 0
        # Monitor head-drop and level tilt during transport.
        target_normal = np.array(self.cfg.world_up, dtype=float)
        target_forward = np.array(self.cfg.scoop_drag_direction_world, dtype=float)
        min_head_drop = float("inf")
        max_tilt = 0.0

        for i in range(frames):
            s = smoothstep5(i / max(frames - 1, 1))
            q_interp = (1.0 - s) * q_start + s * q_neutral
            # Level correction is intended for carry moves; initial convergence
            # to neutral usually works better with plain interpolation.
            if keep_level:
                self._step_to_with_level(d, q_interp, level_weight=3.0)
            else:
                self._step_to(d, q_interp)
            # 매 프레임 head_drop/tilt 추적
            hd = self.robot.spoon_head_drop(d)
            if hd < min_head_drop:
                min_head_drop = hd
            if self.cfg.monitor_tilt_during_replay:
                tilt_val, _, _, _ = self.robot.orientation_errors(d, target_normal, target_forward)
                if tilt_val > max_tilt:
                    max_tilt = tilt_val
            if v is not None:
                _frame_ctr += 1
                if _frame_ctr % _trail_stride == 0:
                    self.trail_positions.append(self.robot.tip_pos(d).copy())
                self._render_trail(v)
                v.sync()
                time.sleep(self.robot.model.opt.timestep)

        warn_bits = []
        if math.isfinite(min_head_drop) and min_head_drop < 0:
            warn_bits.append(f"head_up={-min_head_drop*1000:.1f}mm")
        if max_tilt > self.cfg.tilt_warn_threshold:
            warn_bits.append(f"level_tilt={np.degrees(max_tilt):.1f}deg")
        if warn_bits:
            print(f"  [WARN NEUTRAL] " + ", ".join(warn_bits))
        if math.isfinite(min_head_drop):
            print(f"  [POSE NEUTRAL] min_head_drop={min_head_drop*1000:.1f}mm")

        # v5: 실제 위치 검증
        tip_now = self.robot.tip_pos(d)
        err = float(np.linalg.norm(tip_now - target_world))
        extra_used = 0
        if err <= self.cfg.neutral_position_tol:
            print(
                f"[NEUTRAL DONE] tip={np.round(tip_now, 4).tolist()}, "
                f"target={np.round(target_world, 4).tolist()}, err={err*1000:.1f} mm"
            )
        else:
            # actuator command를 유지하면서 추가 hold (1차 LPF로 천천히 수렴)
            extra = self.cfg.neutral_max_extra_frames
            for j in range(extra):
                if keep_level:
                    self._step_to_with_level(d, q_neutral, level_weight=3.0)
                else:
                    self._step_to(d, q_neutral)
                if v is not None:
                    if j % _trail_stride == 0:
                        self.trail_positions.append(self.robot.tip_pos(d).copy())
                    self._render_trail(v)
                    v.sync()
                    time.sleep(self.robot.model.opt.timestep)
                tip_now = self.robot.tip_pos(d)
                err = float(np.linalg.norm(tip_now - target_world))
                if err <= self.cfg.neutral_position_tol:
                    break
            extra_used = j + 1 if extra > 0 else 0
            if err <= self.cfg.neutral_position_tol:
                print(
                    f"[NEUTRAL DONE +extra={j+1}] tip={np.round(tip_now, 4).tolist()}, "
                    f"target={np.round(target_world, 4).tolist()}, err={err*1000:.1f} mm"
                )
            else:
                print(
                    f"[NEUTRAL APPROXIMATE] tip={np.round(tip_now, 4).tolist()}, "
                    f"target={np.round(target_world, 4).tolist()}, err={err*1000:.1f} mm "
                    f"(> tol {self.cfg.neutral_position_tol*1000:.0f} mm) — q_neutral IK 부정확 가능"
                )

        if pause and self.cfg.neutral_pause_frames > 0:
            self.pause(v=v, frames=self.cfg.neutral_pause_frames, label="NEUTRAL_HOLD")
        q_neutral = np.array(q_neutral, dtype=float)
        q_err = float(np.max(np.abs(d.qpos[:self.robot.model.nq] - q_neutral[:self.robot.model.nq])))
        result = StepResult(
            label="NEUTRAL",
            ok=err <= self.cfg.neutral_position_tol,
            reason="reached" if err <= self.cfg.neutral_position_tol else "neutral_position_error",
            target_pos=tuple(target_world.tolist()),
            actual_pos=tuple(tip_now.tolist()),
            pos_error=err,
            q_error=q_err,
            contact=int(d.ncon),
            extra_frames=extra_used,
            head_drop=float(min_head_drop) if math.isfinite(min_head_drop) else None,
            tilt_error=float(max_tilt),
        )
        print(result.summary())
        return result

    def _neutral_target_world(self) -> np.ndarray:
        """Neutral target의 world 좌표.
        compute_q_neutral이 영역에서 선택한 위치가 있으면 그것을 사용하고,
        없으면 cfg.neutral_pos_tray 기본값 사용.
        """
        if hasattr(self.robot, 'neutral_target_world') and self.robot.neutral_target_world is not None:
            return self.robot.neutral_target_world.copy()
        p_tray = np.array(self.cfg.neutral_pos_tray, dtype=float)
        origin = np.array([self.cfg.base_x_offset, self.cfg.base_y_offset, self.cfg.base_z_offset], dtype=float)
        return origin + p_tray


# =============================================================================
# 9.5. v4 신규: Neutral q 계산 helper
# =============================================================================

def compute_q_neutral(
    cfg: SystemConfig,
    tray: TrayGeometry,
    robot: RobotModel,
    ik: IKSolver,
    verbose: bool = True,
) -> Optional[np.ndarray]:
    """neutral 영역(원통) 내 후보점들에 대해 multi-start IK를 수행하여
    도달 가능 + level + joint safe한 최적의 q_neutral을 선택한다.

    Candidate positions are sampled from the neutral region; each candidate is
    solved with IK and ranked by position error, tilt, and joint margin.
    실패 시 None을 반환하고 caller가 fallback (q_center)으로 처리한다.
    """
    candidate_positions = tray.neutral_points_world()
    if verbose:
        print(f"[Q_NEUTRAL] 영역 내 후보점: {len(candidate_positions)}개")

    rng = np.random.default_rng(cfg.random_seed)
    seeds_base: List[np.ndarray] = [robot.q_center.copy(), np.zeros(robot.model.nq)]
    for _ in range(cfg.multi_start_trials):
        seeds_base.append(robot.sample_random_q(rng))

    best_q: Optional[np.ndarray] = None
    best_score = float("inf")
    best_pos: Optional[np.ndarray] = None
    best_metrics: Dict[str, float] = {}

    fallback_q: Optional[np.ndarray] = None
    fallback_score = float("inf")
    fallback_pos: Optional[np.ndarray] = None
    fallback_metrics: Dict[str, float] = {}

    found_good = False
    for pos_world in candidate_positions:
        target = PoseTarget(
            name="neutral",
            pos=tuple(pos_world.tolist()),
            forward=tuple(cfg.scoop_drag_direction_world),
            normal=tuple(cfg.world_up),
            normal_weight=cfg.neutral_normal_weight,
            # Neutral prefers level posture, but keeps the normal constraint
            # soft so the search does not return zero candidates.
            normal_hard=False,
            forward_weight=0.0,
            forward_hard=False,
            head_drop_min=cfg.neutral_head_drop_min if cfg.neutral_head_drop_weight > 0 else None,
            head_drop_weight=cfg.neutral_head_drop_weight,
        )

        for q_seed in seeds_base:
            _, q_sol, metrics = ik.solve_pose(target, q_seed, posture_ref=q_seed)
            pos_err = metrics.get("pos_error", float("inf"))
            tilt = metrics.get("tilt_error", float("inf"))
            contact = int(metrics.get("contact", 0))
            sigma = metrics.get("sigma_min", 0.0)
            condition = metrics.get("condition", float("inf"))
            # Include joint-limit margin and head-drop in candidate quality.
            marg = robot.min_joint_limit_margin_ratio(q_sol)
            head_drop = metrics.get("head_drop", 0.0)

            # score = position error + tilt penalty - joint-margin bonus.
            score = (
                pos_err
                + cfg.neutral_tilt_score_weight * tilt
                - cfg.neutral_margin_score_weight * max(0.0, marg)
            )

            # Require the neutral-specific joint-limit margin.
            safe_neutral = robot.is_joint_limit_safe_with_margin(
                q_sol, cfg.neutral_joint_margin_ratio
            )

            # 주 후보: neutral_position_tol(25mm) + (v12) level·head-up·관절여유 게이트.
            primary_ok = (
                pos_err <= cfg.neutral_position_tol and
                safe_neutral and
                tilt <= cfg.neutral_max_tilt and
                head_drop >= 0.0 and
                contact <= cfg.contact_allowed and
                sigma >= cfg.min_sigma and
                condition <= cfg.max_condition
            )
            if primary_ok and score < best_score:
                best_q = q_sol.copy()
                best_score = score
                best_pos = pos_world.copy()
                best_metrics = metrics
                if marg >= cfg.neutral_good_margin:
                    found_good = True

            # fallback: 50mm 이내 + 기계적 안전(마진 0%). primary가 없을 때 최후 수단.
            fallback_ok = (
                pos_err <= 0.050 and
                robot.is_joint_limit_safe_with_margin(q_sol, 0.0) and
                sigma >= cfg.min_sigma and
                condition <= cfg.max_condition
            )
            if fallback_ok and score < fallback_score:
                fallback_q = q_sol.copy()
                fallback_score = score
                fallback_pos = pos_world.copy()
                fallback_metrics = metrics

        if found_good:
            break

    # fallback 사용
    if best_q is None and fallback_q is not None:
        best_q = fallback_q.copy()
        best_score = fallback_score
        best_pos = fallback_pos
        best_metrics = fallback_metrics
        if verbose:
            tilt_deg = np.degrees(best_metrics.get("tilt_error", 0))
            print(
                f"[Q_NEUTRAL] using safe fallback "
                f"pos_err={best_metrics.get('pos_error', 0):.4f} "
                f"tilt={tilt_deg:.1f}deg"
            )

    if best_q is None:
        if verbose:
            print("[Q_NEUTRAL] 영역 내 validation 통과 후보 없음 — neutral 이동을 실패로 처리")
        return None

    robot.q_neutral = best_q.copy()
    # 선택된 neutral 위치를 저장 (replay_neutral에서 사용)
    robot.neutral_target_world = best_pos.copy() if best_pos is not None else tray.neutral_pos_world()
    if verbose:
        tilt_deg = np.degrees(best_metrics.get("tilt_error", 0))
        hd = best_metrics.get("head_drop", 0) * 1000
        print(
            f"[Q_NEUTRAL] selected_pos={np.round(best_pos, 4).tolist()} "
            f"pos_err={best_metrics.get('pos_error', 0)*1000:.1f}mm "
            f"tilt={tilt_deg:.1f}deg head_drop={hd:.1f}mm "
            f"score={best_score:.4f}"
        )
    return best_q


def _neutral_config_matches(cfg: SystemConfig, saved_cfg: Dict[str, object]) -> bool:
    """저장 당시 neutral config가 현재 cfg와 같은지 비교 (stale 판별)."""
    def _close(a, b) -> bool:
        try:
            aa = np.array(a, dtype=float).ravel()
            bb = np.array(b, dtype=float).ravel()
            return aa.shape == bb.shape and bool(np.allclose(aa, bb, atol=1e-9))
        except (TypeError, ValueError):
            return a == b

    current = _neutral_config_snapshot(cfg)
    for key, val in current.items():
        if key not in saved_cfg or not _close(saved_cfg[key], val):
            return False
    return True


def _neutral_connector_valid(cfg: SystemConfig, robot: RobotModel, connector: NeutralConnector) -> bool:
    """저장된 q_neutral이 현재 모델에서 구조적으로 유효한지 확인.

    compute_q_neutral은 strict validation 실패 시 safe fallback을 답으로 채택하므로,
    여기서 엄격한 위치 허용오차를 요구하면 매 run마다 재빌드가 반복될 수 있다.
    따라서 (1) 차원 일치 (2) joint limit 안전 (3) 자기충돌 없음만 검증한다.
    """
    q = np.array(connector.q_neutral, dtype=float)
    if q.shape[0] != robot.model.nq:
        return False
    # neutral은 primary(0.5%)나 fallback(0%) 어느 쪽으로든 저장될 수 있다.
    # validation은 항상 0% 마진(하드 리밋 이내)만 요구한다 — 그 이상은 IK score로 선택됨.
    # is_joint_limit_safe(1.5%)나 0.5%로 체크하면 fallback 저장분이 항상 실패해 재계산된다.
    if not robot.is_joint_limit_safe_with_margin(q, 0.0):
        return False
    d = mujoco.MjData(robot.model)
    robot.set_q(d, q)
    return int(d.ncon) <= cfg.contact_allowed


def load_or_build_q_neutral(
    cfg: SystemConfig,
    tray: TrayGeometry,
    robot: RobotModel,
    ik: IKSolver,
    force_rebuild: bool = False,
    verbose: bool = True,
) -> Optional[np.ndarray]:
    """q_neutral을 LUT 캐시(neutral.json)에서 로드하거나, 없으면 계산 후 저장한다.

    mouth connector(get_mouth_connector)와 동일한 캐싱 정책:
      - force_rebuild=False (run 모드): 캐시가 있고 config가 일치하며 q가 유효하면
        IK 없이 즉시 로드. stale/invalid면 자동 재계산.
      - force_rebuild=True (build 모드): 캐시를 무시하고 항상 계산 후 저장.
      - cfg.use_neutral_lut=False: 캐시 무시, 항상 온라인 재계산(디버그용).

    어느 경로든 robot.q_neutral / robot.neutral_target_world를 채운다.
    """
    db = NeutralDatabase(cfg)

    if cfg.use_neutral_lut and not force_rebuild:
        try:
            connector, saved_cfg = db.load_payload()
        except FileNotFoundError:
            pass
        else:
            stale = not _neutral_config_matches(cfg, saved_cfg)
            invalid = not _neutral_connector_valid(cfg, robot, connector)
            if stale or invalid:
                reason = "config가 현재 cfg와 불일치" if stale else "q_neutral validation 실패"
                if verbose:
                    print(f"[Q_NEUTRAL REBUILD] 캐시 {reason} → 재계산합니다.")
            else:
                robot.q_neutral = np.array(connector.q_neutral, dtype=float)
                robot.neutral_target_world = np.array(connector.neutral_pos, dtype=float)
                if verbose:
                    print(
                        "[Q_NEUTRAL LOAD] "
                        f"pos={np.round(robot.neutral_target_world, 4).tolist()} "
                        f"pos_err={connector.pos_error*1000:.1f}mm "
                        f"tilt={np.degrees(connector.tilt_error):.1f}deg "
                        f"head_drop={connector.head_drop*1000:.1f}mm (cache)"
                    )
                return robot.q_neutral
    elif not cfg.use_neutral_lut and verbose:
        print("[Q_NEUTRAL] use_neutral_lut=False → 캐시 무시하고 온라인 재계산합니다.")

    q_neutral = compute_q_neutral(cfg, tray, robot, ik, verbose=verbose)
    if q_neutral is not None:
        # FK 기반 metrics를 재계산해 캐시에 저장한다.
        d = mujoco.MjData(robot.model)
        robot.set_q(d, q_neutral)
        tip = robot.tip_pos(d)
        target = np.array(
            robot.neutral_target_world
            if robot.neutral_target_world is not None
            else tray.neutral_pos_world(),
            dtype=float,
        )
        n_tgt = np.array(cfg.world_up, dtype=float)
        f_tgt = np.array(cfg.scoop_drag_direction_world, dtype=float)
        tilt, _f, _du, _df = robot.orientation_errors(d, n_tgt, f_tgt)
        connector = NeutralConnector(
            q_neutral=q_neutral.tolist(),
            neutral_pos=tuple(target.tolist()),
            pos_error=float(np.linalg.norm(tip - target)),
            tilt_error=float(tilt),
            head_drop=float(robot.spoon_head_drop(d)),
        )
        db.save(connector)
    return q_neutral


# =============================================================================
# 10. CLI
# =============================================================================

def _build_and_save_mouth_connector(cfg: SystemConfig, robot: RobotModel, ik: IKSolver) -> bool:
    """q_neutral을 seed로 mouth connector를 계산해 mouth_connector.json에 저장한다.

    build_lut와 build_mouth_lut가 공유한다. q_neutral이 없으면 건너뛴다.
    """
    if robot.q_neutral is None:
        print("[MOUTH CONNECTOR SKIP] q_neutral이 없어 connector를 빌드하지 않습니다.")
        return False
    runner_for_mouth = SequenceRunner(cfg, robot)
    ok_connector, connector = runner_for_mouth.build_mouth_connector(
        seed_q=robot.q_neutral,
        ik=ik,
        rng=np.random.default_rng(cfg.random_seed),
    )
    if ok_connector and connector is not None:
        MouthConnectorDatabase(cfg).save(connector)
        return True
    print("[MOUTH CONNECTOR FAIL] connector 빌드 실패 — 저장하지 않습니다.")
    return False


def build_mouth_lut(cfg: SystemConfig):
    """scoop LUT 재빌드 없이 mouth connector LUT만 생성/갱신한다."""
    tray = TrayGeometry(cfg)
    robot = RobotModel(cfg)
    ik = IKSolver(cfg, robot)
    load_or_build_q_neutral(cfg, tray, robot, ik, force_rebuild=True)
    ok = _build_and_save_mouth_connector(cfg, robot, ik)
    print("[BUILD MOUTH LUT]", "완료" if ok else "실패")


def build_lut(cfg: SystemConfig, regions_filter: Optional[List[int]] = None):
    """neutral/mouth connector를 준비한 뒤 지정 region들의 scoop primitive LUT를 생성해 저장한다."""
    tray = TrayGeometry(cfg)
    robot = RobotModel(cfg)
    ik = IKSolver(cfg, robot)
    # Build and cache the neutral pose used by replay and mouth transfer.
    load_or_build_q_neutral(cfg, tray, robot, ik, force_rebuild=True)
    _build_and_save_mouth_connector(cfg, robot, ik)
    builder = ScoopPrimitiveBuilder(cfg, tray, robot, ik)
    regions = tray.default_regions()
    if regions_filter is not None:
        wanted = {int(r) for r in regions_filter}
        regions = [r for r in regions if r.region_id in wanted]

    all_primitives: List[ScoopPrimitive] = []
    for r in regions:
        print("==========================================")
        print(f"[BUILD REGION] {r.region_id}: {r.name}")
        print("polygon:", r.polygon_xy)
        ps = builder.build_for_region(r)
        print(f"[REGION DONE] region={r.region_id}, primitives={len(ps)}")
        all_primitives.extend(ps)

    db = PrimitiveDatabase(cfg)
    db.save(all_primitives, regions)
    print("[TOTAL PRIMITIVES]", len(all_primitives))


def select_primitive(cfg: SystemConfig, region: int, food_xy: Tuple[float, float], top_k: int = 5) -> List[ScoopPrimitive]:
    """저장된 LUT에서 지정 region과 food_xy에 맞는 상위 primitive 후보를 출력하고 반환한다."""
    db = PrimitiveDatabase(cfg)
    selected = db.select(region, food_xy, top_k=top_k)
    if not selected:
        print(f"[NO CANDIDATE] region={region}, food_xy={food_xy}")
        return []
    print("==========================================")
    print("[SELECTED PRIMITIVES]")
    for i, p in enumerate(selected, start=1):
        dist = np.linalg.norm(np.array(p.food_xy) - np.array(food_xy))
        print(f"rank={i} id={p.primitive_id} region={p.region_id} dist={dist:.4f} score={p.score:.4f} food_xy={p.food_xy} drag={p.drag_length}")
    return selected


def replay_selected(
    cfg: SystemConfig,
    region: int,
    food_xy: Tuple[float, float],
    viewer: bool,
    seed: Optional[int] = None,
    start_from_random: bool = True,
):
    """지정된 food_xy에 대한 top-1 primitive를 전체 sequence로 실행.

    제어 코드 계획서 + 사용자 요구사항:
        Random Initial -> Neutral -> Pre -> Engage -> Drag(-X) -> Lift
        -> Neutral -> Mouth -> Neutral
    """
    selected = select_primitive(cfg, region, food_xy, top_k=1)
    if not selected:
        return
    robot = RobotModel(cfg)
    # Load or build the neutral pose before replay.
    tray = TrayGeometry(cfg)
    ik_for_neutral = IKSolver(cfg, robot)
    load_or_build_q_neutral(cfg, tray, robot, ik_for_neutral)
    runner = SequenceRunner(cfg, robot)
    runner.trail_positions.clear()

    primitive = selected[0]
    print("\n[RUN SCOOP SEQUENCE]")
    print("0. Random Initial Position")
    print("1. Random -> Neutral")
    print("2. Neutral -> Pre-scoop")
    print("3. Pre-scoop -> Engage")
    print("4. Engage -> Drag start")
    print("5. Drag start -> Drag end (-X scoop)")
    print("6. Drag end -> Lift")
    print("7. Lift -> Neutral")
    print("8. Neutral -> Mouth")
    print("9. Mouth -> Neutral")

    if viewer:
        with mujoco.viewer.launch_passive(robot.model, robot.data) as v:
            completed = _execute_primitives_continuously(
                cfg, robot, runner, [primitive], v=v,
                start_from_random=start_from_random, seed=seed,
                use_full_sequence=True,
            )
    else:
        completed = _execute_primitives_continuously(
            cfg, robot, runner, [primitive], v=None,
            start_from_random=start_from_random, seed=seed,
            use_full_sequence=True,
        )
    print("[DONE] replay 완료" if completed else "[ABORTED] replay 중단")


def _best_primitives_for_region(cfg: SystemConfig, region: int) -> List[ScoopPrimitive]:
    """LUT에서 region별 food_xy마다 score가 가장 낮은 primitive만 추린다."""
    db = PrimitiveDatabase(cfg)
    all_primitives = [p for p in db.load() if p.region_id == region]
    best_per_food: Dict[Tuple[float, float], ScoopPrimitive] = {}
    for p in all_primitives:
        key = tuple(p.food_xy)
        if key not in best_per_food or p.score < best_per_food[key].score:
            best_per_food[key] = p
    return list(best_per_food.values())


def _execute_primitives_continuously(
    cfg: SystemConfig,
    robot: RobotModel,
    runner: SequenceRunner,
    selected: List[ScoopPrimitive],
    v=None,
    neutral_between: bool = True,           # legacy 옵션 (use_full_sequence=True면 자동 포함)
    start_from_random: bool = True,         # 사용자 요구: Random 초기 위치에서 시작
    seed: Optional[int] = None,
    use_full_sequence: bool = True,         # 사용자 요구: Mouth 단계 포함한 전체 sequence
    random_pose_hold_frames: int = 60,      # 랜덤 초기 자세를 viewer에 잠깐 보여주는 시간
):
    """여러 primitive를 하나의 MuJoCo data / viewer에서 연속 실행한다.

    제어 코드 계획서 (260515 계획서 7번) 기준 전체 sequence:
        Random Initial Point
            -> Neutral
            -> [ Pre-scoop -> Engage -> Drag start -> Drag end (-X) -> Lift
                 -> Neutral -> Mouth -> Neutral ] x N
    """
    # ---- 1. Random Initial Position에서 시작 ----
    completed_all = False

    if start_from_random:
        rng = np.random.default_rng(seed)
        q_random = runner.sample_neutral_reachable_initial_q(rng)
        robot.set_q(robot.data, q_random)
        if robot.model.nu > 0:
            robot.data.ctrl[:robot.model.nu] = q_random[:robot.model.nu]
        print(f"\n[STEP 0] Random Initial Position 설정 (q[:4]={np.round(q_random[:4], 3).tolist()})")
        if v is not None:
            # 잠깐 랜덤 초기 자세를 보여주고 trail에 시작점 기록
            try:
                runner.trail_positions.append(robot.tip_pos(robot.data).copy())
            except Exception:
                pass
            for _ in range(max(1, random_pose_hold_frames)):
                runner._render_trail(v)
                v.sync()
                time.sleep(robot.model.opt.timestep)
        # 2. Random -> Neutral
        print("[STEP 1] Random Initial -> Neutral")
        neutral_result = runner.replay_neutral(v=v, frames=max(960, 4 * cfg.frames_per_segment))
        if not neutral_result.ok:
            print("[ABORT] 초기 Neutral 도달 실패. scoop sequence를 시작하지 않습니다.")
            return False
    else:
        robot.set_q(robot.data, robot.q_center.copy())
        if robot.model.nu > 0:
            robot.data.ctrl[:robot.model.nu] = robot.q_center[:robot.model.nu]

    # ---- 3. 각 primitive: Neutral -> Pre -> Engage -> Drag -> Lift -> Neutral -> Mouth -> Neutral ----
    ik = IKSolver(cfg, robot) if use_full_sequence else None
    for i, primitive in enumerate(selected, 1):
        print(
            f"\n[ACTION {i}/{len(selected)}] {primitive.primitive_id}  "
            f"region={primitive.region_id}  food={primitive.food_xy}  score={primitive.score:.4f}"
        )
        if use_full_sequence:
            # 한 primitive 안에서 Mouth/Neutral 복귀까지 포함됨
            result = runner.replay_full_sequence(primitive, v=v, ik=ik, neutral_after_lift=True)
            if not result.ok:
                print(f"[ABORT] action {i} 실패. 다음 scoop으로 진행하지 않습니다.")
                return False
        else:
            result = runner.replay_continuous(primitive, v=v)
            if not result.ok:
                print(f"[ABORT] action {i} 실패. 다음 scoop으로 진행하지 않습니다.")
                return False
            if neutral_between and i < len(selected):
                print("[NEUTRAL >>] 다음 동작 전 neutral 복귀...")
                neutral_result = runner.replay_neutral(v=v, frames=max(960, 4 * cfg.frames_per_segment))
                if not neutral_result.ok:
                    print("[ABORT] 다음 동작 전 Neutral 도달 실패. 다음 scoop으로 진행하지 않습니다.")
                    return False

    completed_all = True
    return completed_all


def run_lut_sequence(
    cfg: SystemConfig,
    region: Optional[int] = None,
    n_actions: Optional[int] = 1,
    viewer: bool = False,
    seed: Optional[int] = None,
    shuffle: bool = False,
):
    """build_lut 결과 CSV에서 실행할 primitive를 자동으로 가져와 연속 실행한다.

    사용자가 food_xy를 직접 넣지 않아도 된다. CSV summary는 실행 후보의
    primitive_id와 score를 고르는 index로 사용하고, 실제 q_pre~q_lift는
    같은 LUT의 JSON에서 가져온다.
    """
    db = PrimitiveDatabase(cfg)
    selected = db.auto_select_from_lut(
        region_id=region,
        n_actions=n_actions,
        seed=seed,
        shuffle=shuffle,
        unique_food_xy=True,
    )
    if not selected:
        target = "전체 LUT" if region is None else f"region {region}"
        print(f"[ERROR] {target}에서 실행할 primitive가 없습니다. 먼저 --mode build_lut를 실행하세요.")
        return

    print(f"\n{'='*70}")
    print("[RUN LUT SEQUENCE] CSV summary 기반 자동 선택")
    print(f"  csv : {db.csv_path}")
    print(f"  json: {db.json_path}")
    print(f"  region filter: {region if region is not None else 'ALL'}")
    print(f"  actions: {len(selected)}")
    print(f"{'='*70}")
    for i, p in enumerate(selected, 1):
        print(f"  action {i:2d}: id={p.primitive_id} region={p.region_id} food_xy={p.food_xy} drag={p.drag_length:.3f} score={p.score:.4f}")
    print(f"{'='*70}\n")

    robot = RobotModel(cfg)
    # Load or build the neutral pose before replay.
    tray_for_neutral = TrayGeometry(cfg)
    ik_for_neutral = IKSolver(cfg, robot)
    load_or_build_q_neutral(cfg, tray_for_neutral, robot, ik_for_neutral)
    runner = SequenceRunner(cfg, robot)
    runner.trail_positions.clear()

    if viewer:
        with mujoco.viewer.launch_passive(robot.model, robot.data) as v:
            completed = _execute_primitives_continuously(
                cfg, robot, runner, selected, v=v,
                start_from_random=True, seed=seed,
                use_full_sequence=True,
            )
    else:
        completed = _execute_primitives_continuously(
            cfg, robot, runner, selected, v=None,
            start_from_random=True, seed=seed,
            use_full_sequence=True,
        )

    print("\n" + "="*70)
    print("[DONE] CSV LUT 기반 sequence 실행 완료" if completed else "[ABORTED] CSV LUT 기반 sequence 중단")
    print("="*70)


def run_region(
    cfg: SystemConfig,
    region: int,
    n_actions: Optional[int] = None,
    viewer: bool = False,
    seed: Optional[int] = None,
):
    """지정 region 안의 여러 scoop primitive를 한 시뮬레이션에서 연속 실행한다."""
    pool = _best_primitives_for_region(cfg, region)
    if not pool:
        print(f"[ERROR] region {region}에 저장된 primitive가 없습니다. 먼저 --mode build_lut를 실행하세요.")
        return

    rng = np.random.default_rng(seed)
    rng.shuffle(pool)
    if n_actions is None:
        n_actions = len(pool)
    selected = [pool[i % len(pool)] for i in range(n_actions)]

    print(f"\n{'='*62}")
    print(f"[RUN REGION {region}] 저장 food 위치 수={len(pool)}, 수행 동작={n_actions}")
    print(f"{'='*62}")
    for i, p in enumerate(selected, 1):
        print(f"  action {i:2d}: {p.primitive_id} food_xy={p.food_xy} drag={p.drag_length:.3f} m score={p.score:.4f}")
    print(f"{'='*62}\n")

    robot = RobotModel(cfg)
    # Load or build the neutral pose before replay.
    tray_for_neutral = TrayGeometry(cfg)
    ik_for_neutral = IKSolver(cfg, robot)
    load_or_build_q_neutral(cfg, tray_for_neutral, robot, ik_for_neutral)
    runner = SequenceRunner(cfg, robot)
    runner.trail_positions.clear()

    if viewer:
        with mujoco.viewer.launch_passive(robot.model, robot.data) as v:
            completed = _execute_primitives_continuously(
                cfg, robot, runner, selected, v=v,
                start_from_random=True, seed=seed,
                use_full_sequence=True,
            )
    else:
        completed = _execute_primitives_continuously(
            cfg, robot, runner, selected, v=None,
            start_from_random=True, seed=seed,
            use_full_sequence=True,
        )

    print("\n" + "="*62)
    print(f"[DONE] region {region} {n_actions}회 scoop 완료" if completed else f"[ABORTED] region {region} sequence 중단")
    print("="*62)


def run_all_regions(
    cfg: SystemConfig,
    regions: List[int],
    n_actions_per_region: int = 1,
    viewer: bool = False,
    seed: Optional[int] = None,
):
    """Region 1~5의 대표 primitive를 한 viewer/data에서 순서대로 연속 실행한다."""
    rng = np.random.default_rng(seed)
    selected: List[ScoopPrimitive] = []

    for region in regions:
        pool = _best_primitives_for_region(cfg, region)
        if not pool:
            print(f"[WARN] region {region} primitive 없음. 건너뜀.")
            continue
        pool = sorted(pool, key=lambda p: p.score)
        if n_actions_per_region <= 1:
            chosen = [pool[0]]
        else:
            rng.shuffle(pool)
            pool = sorted(pool, key=lambda p: p.score)
            chosen = [pool[i % len(pool)] for i in range(n_actions_per_region)]
        selected.extend(chosen)

    if not selected:
        print("[ERROR] 실행할 primitive가 없습니다. 먼저 --mode build_lut를 실행하세요.")
        return

    print(f"\n{'='*70}")
    print(f"[RUN ALL REGIONS] regions={regions}, 총 동작 수={len(selected)}")
    print(f"{'='*70}")
    for i, p in enumerate(selected, 1):
        print(f"  action {i:2d}: region={p.region_id} id={p.primitive_id} food_xy={p.food_xy} drag={p.drag_length:.3f} score={p.score:.4f}")
    print(f"{'='*70}\n")

    robot = RobotModel(cfg)
    # Load or build the neutral pose before replay.
    tray_for_neutral = TrayGeometry(cfg)
    ik_for_neutral = IKSolver(cfg, robot)
    load_or_build_q_neutral(cfg, tray_for_neutral, robot, ik_for_neutral)
    runner = SequenceRunner(cfg, robot)
    runner.trail_positions.clear()

    if viewer:
        with mujoco.viewer.launch_passive(robot.model, robot.data) as v:
            completed = _execute_primitives_continuously(
                cfg, robot, runner, selected, v=v,
                start_from_random=True, seed=seed,
                use_full_sequence=True,
            )
    else:
        completed = _execute_primitives_continuously(
            cfg, robot, runner, selected, v=None,
            start_from_random=True, seed=seed,
            use_full_sequence=True,
        )

    print("\n" + "="*70)
    print(f"[DONE] regions {regions} 연속 scoop 완료" if completed else f"[ABORTED] regions {regions} sequence 중단")
    print("="*70)


def test_run(cfg: SystemConfig):
    """Run a lightweight validation without opening the viewer.

    - q_neutral 계산 결과 출력
    - Mouth Z 후보군 multi-start IK 결과 출력
    - 첫 번째 region의 첫 번째 primitive에 대해 mouth IK + boundary 통과 여부 확인
    """
    tray = TrayGeometry(cfg)
    robot = RobotModel(cfg)
    ik = IKSolver(cfg, robot)

    print("=" * 70)
    print("[TEST_RUN] v10 - mouth connector LUT + FK head_drop IK/StepResult 검증 (viewer 없이)")
    print("=" * 70)

    # 1) q_neutral 계산
    q_neutral = compute_q_neutral(cfg, tray, robot, ik)
    if q_neutral is not None:
        d = mujoco.MjData(robot.model)
        robot.set_q(d, q_neutral)
        tip = robot.tip_pos(d)
        n_target = np.array(cfg.world_up, dtype=float)
        f_target = np.array(cfg.scoop_drag_direction_world, dtype=float)
        tilt, _fwd, _du, _df = robot.orientation_errors(d, n_target, f_target)
        neutral_err = float(np.linalg.norm(tip - tray.neutral_pos_world()))
        neutral_result = StepResult(
            label="TEST_Q_NEUTRAL",
            ok=neutral_err <= cfg.neutral_position_tol and robot.is_joint_limit_safe(q_neutral),
            reason="validated" if neutral_err <= cfg.neutral_position_tol else "neutral_position_error",
            target_pos=tuple(tray.neutral_pos_world().tolist()),
            actual_pos=tuple(tip.tolist()),
            pos_error=neutral_err,
            tilt_error=tilt,
            contact=int(d.ncon),
        )
        print(f"[q_neutral] joints={np.round(q_neutral, 4).tolist()}")
        print(f"[q_neutral] tip_world={tip.tolist()}, target_world={tray.neutral_pos_world().tolist()}")
        print(f"[q_neutral] joint_limit_safe (1.5%)={robot.is_joint_limit_safe(q_neutral)}")
        print(neutral_result.summary())
    else:
        print("[q_neutral] 계산 실패")
        print(StepResult(label="TEST_Q_NEUTRAL", ok=False, reason="neutral_ik_failed").summary())

    # 2) Mouth Z 후보 multi-start IK
    runner = SequenceRunner(cfg, robot)
    seed = q_neutral if q_neutral is not None else robot.q_center.copy()
    ok, q_mouth, metrics, pos_best = runner.solve_mouth_q_multi(seed_q=seed, ik=ik, rng=np.random.default_rng(0))
    if ok and q_mouth is not None:
        d = mujoco.MjData(robot.model)
        robot.set_q(d, q_mouth)
        tip = robot.tip_pos(d)
        mouth_err = float(np.linalg.norm(tip - np.array(pos_best, dtype=float)))
        mouth_result = StepResult(
            label="TEST_MOUTH_IK",
            ok=mouth_err <= cfg.mouth_position_tol and bool(metrics.get("joint_limit_ok", 0.0)),
            reason="validated" if mouth_err <= cfg.mouth_position_tol else "mouth_position_error",
            target_pos=tuple(pos_best),
            actual_pos=tuple(tip.tolist()),
            pos_error=mouth_err,
            tilt_error=float(metrics.get("tilt_error", 0.0)),
            contact=int(metrics.get("contact", 0)),
        )
        print(f"[MOUTH IK OK] chosen_pos={pos_best}, tip_world={tip.tolist()}")
        print(f"[MOUTH IK OK] metrics={ {k: float(v) for k, v in metrics.items()} }")
        print(mouth_result.summary())
        for i in range(robot.model.nq):
            lo, hi = robot.q_lower[i], robot.q_upper[i]
            d_to_lo = q_mouth[i] - lo
            d_to_hi = hi - q_mouth[i]
            ratio_lo = d_to_lo / max(hi - lo, 1e-6)
            ratio_hi = d_to_hi / max(hi - lo, 1e-6)
            tag = ""
            if ratio_lo <= cfg.joint_limit_margin_ratio:
                tag = "  <- near LO (strict 1.5% 위반)"
            elif ratio_hi <= cfg.joint_limit_margin_ratio:
                tag = "  <- near HI (strict 1.5% 위반)"
            print(
                f"  joint {i}: q={q_mouth[i]:+.4f}, "
                f"range=[{lo:+.4f}, {hi:+.4f}], "
                f"margin_lo_ratio={ratio_lo:.3%}, margin_hi_ratio={ratio_hi:.3%}{tag}"
            )
    else:
        print(f"[MOUTH IK FAIL] best metrics={metrics}, pos_best={pos_best}")
        print(StepResult(label="TEST_MOUTH_IK", ok=False, reason="mouth_ik_failed").summary())

    # 3) Scoop boundary 통과 여부 (LUT 첫 region에서 첫 primitive)
    try:
        db = PrimitiveDatabase(cfg)
        all_primitives = db.load()
    except FileNotFoundError:
        print("[TEST_RUN] LUT 미생성 — boundary check 생략 (build_lut 후 재실행)")
        return
    if not all_primitives:
        print("[TEST_RUN] LUT empty — boundary check skipped")
        return

    region_ids = sorted({p.region_id for p in all_primitives})
    print(f"[BOUNDARY CHECK] regions={region_ids}")
    for rid in region_ids:
        prims = [p for p in all_primitives if p.region_id == rid]
        if not prims:
            continue
        p0 = min(prims, key=lambda p: p.score)
        ds_world = np.array(p0.drag_start_pos)
        de_world = np.array(p0.drag_end_pos)
        ds_tray = tray.world_to_tray(ds_world)
        de_tray = tray.world_to_tray(de_world)
        in_tray = (
            cfg.spoon_outer_margin <= ds_tray[0] <= cfg.tray_x_length - cfg.spoon_outer_margin and
            cfg.spoon_outer_margin <= ds_tray[1] <= cfg.tray_y_length - cfg.spoon_outer_margin and
            cfg.spoon_outer_margin <= de_tray[0] <= cfg.tray_x_length - cfg.spoon_outer_margin and
            cfg.spoon_outer_margin <= de_tray[1] <= cfg.tray_y_length - cfg.spoon_outer_margin
        )
        print(
            f"  region {rid} primitive={p0.primitive_id}: "
            f"drag_start_tray={np.round(ds_tray, 4).tolist()}, "
            f"drag_end_tray={np.round(de_tray, 4).tolist()}, "
            f"within_tray={in_tray}"
        )

    print("=" * 70)
    print("[TEST_RUN] 완료")
    print("=" * 70)


def parse_args():
    """CLI 옵션을 정의하고 명령행 인자를 파싱한다."""
    parser = argparse.ArgumentParser(
        description="식사보조로봇 scoop primitive LUT 생성 / 선택 / 재생 (v10 mouth connector LUT + FK head_drop posture)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["build_lut", "build_mouth_lut", "select", "replay", "run_sequence", "run_lut", "run_region", "run_all_regions", "test_run"],
        required=True,
        help=(
            "build_lut    : scoop + mouth connector LUT 저장. 예: --regions 5 로 단일 region 먼저 확인\n"
            "build_mouth_lut : scoop LUT 재빌드 없이 mouth connector LUT만 생성/갱신\n"
            "select     : food_xy 지정 -> 후보 primitive 출력\n"
            "replay       : food_xy 지정 -> top-1 primitive를 StepResult 검증과 함께 전체 sequence 재생\n"
            "run_sequence : food_xy가 없으면 CSV LUT에서 자동 선택, 있으면 지정 food_xy 실행\n"
            "run_lut      : CSV summary에서 primitive를 자동 선택해 실행\n"
            "run_region   : 한 region 안에서 여러 primitive를 연속 실행. 먼저 확인 예: --region 5 --n_actions 1 --viewer\n"
            "run_all_regions : region 1~5 대표 primitive 연속 실행\n"
            "test_run     : Neutral/Mouth connector IK와 StepResult 출력 검증"
        ),
    )
    parser.add_argument("--region",    type=int,   default=None,
                        help="대상 region_id (1~5). run_sequence/run_lut에서 생략하면 전체 CSV 중 score 최저 후보 자동 선택")
    parser.add_argument("--food_xy",   type=float, nargs=2, default=None,
                        help="[select/replay 선택사항] tray frame 음식 위치 x y. 생략하면 run_sequence는 CSV LUT에서 자동 선택")
    parser.add_argument("--n_actions", type=int,   default=None,
                        help="[run_region] 수행할 scoop 횟수 / [run_all_regions] region당 동작 수")
    parser.add_argument("--regions", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                        help="[build_lut/run_all_regions] 대상 region 목록. 기본: 1 2 3 4 5")
    parser.add_argument("--food_samples", type=int, default=None,
                        help="[build_lut 전용] region 내부 food sample 축 개수 override. 단일 region 확인 예: 3, 빠른 디버그 예: 1")
    parser.add_argument("--ik_iters", type=int, default=None,
                        help="[build_lut/test 전용] IK 최대 반복 수 override. 빠른 디버그 예: 250")
    parser.add_argument("--multi_start", type=int, default=None,
                        help="[build_lut/test 전용] multi-start random seed 수 override. 단일 region 확인 예: 12, 빠른 디버그 예: 1")
    parser.add_argument("--seed",      type=int,   default=None,
                        help="랜덤 시드 (재현용, 미지정 시 비결정적)")
    parser.add_argument("--shuffle",   action="store_true",
                        help="[run_lut/run_sequence 자동 선택] CSV 후보를 score 정렬 대신 shuffle해서 실행")
    parser.add_argument("--viewer",    action="store_true",
                        help="MuJoCo 뷰어 창 표시")
    parser.add_argument("--xml",       type=str,   default="robot_model_v5_scene.xml",
                        help="MuJoCo XML 모델 파일명")
    return parser.parse_args()


def main():
    # Windows 콘솔(cp949)에서 '—' 같은 비-cp949 문자 print 시 UnicodeEncodeError로
    # 죽는 것을 방지한다. 인코딩(cp949)은 유지하되 못 쓰는 문자만 '?'로 대체.
    """CLI mode에 따라 LUT 생성, 선택, 재생, 테스트 실행 흐름을 분기한다."""
    import sys as _sys
    for _stream in (_sys.stdout, _sys.stderr):
        try:
            _stream.reconfigure(errors="replace")
        except Exception:
            pass
    args = parse_args()
    cfg = SystemConfig(xml_name=args.xml)
    if args.food_samples is not None:
        cfg.food_samples_per_region_axis = int(args.food_samples)
    if args.ik_iters is not None:
        cfg.ik_iters = int(args.ik_iters)
    if args.multi_start is not None:
        cfg.multi_start_trials = int(args.multi_start)
        cfg.mouth_multi_start_seeds = int(args.multi_start)
    if args.mode == "build_lut":
        build_lut(cfg, regions_filter=args.regions)
    elif args.mode == "build_mouth_lut":
        build_mouth_lut(cfg)
    elif args.mode == "select":
        if args.food_xy is None:
            print("[ERROR] select 모드는 --food_xy x y가 필요합니다.")
            return
        if args.region is None:
            print("[ERROR] select 모드는 --region이 필요합니다.")
            return
        select_primitive(cfg, args.region, tuple(args.food_xy), top_k=5)
    elif args.mode == "replay":
        if args.food_xy is None:
            run_lut_sequence(cfg, region=args.region, n_actions=(args.n_actions if args.n_actions is not None else 1), viewer=args.viewer, seed=args.seed, shuffle=args.shuffle)
        else:
            if args.region is None:
                print("[ERROR] replay 모드에서 --food_xy를 직접 주는 경우 --region도 필요합니다.")
                return
            replay_selected(cfg, args.region, tuple(args.food_xy), viewer=args.viewer, seed=args.seed, start_from_random=True)
    elif args.mode == "run_sequence":
        if args.food_xy is None:
            run_lut_sequence(cfg, region=args.region, n_actions=(args.n_actions if args.n_actions is not None else 1), viewer=args.viewer, seed=args.seed, shuffle=args.shuffle)
        else:
            replay_selected(cfg, args.region, tuple(args.food_xy), viewer=args.viewer, seed=args.seed, start_from_random=True)
    elif args.mode == "run_lut":
        run_lut_sequence(cfg, region=args.region, n_actions=(args.n_actions if args.n_actions is not None else 1), viewer=args.viewer, seed=args.seed, shuffle=args.shuffle)
    elif args.mode == "run_region":
        if args.region is None:
            print("[ERROR] run_region 모드는 --region이 필요합니다.")
            return
        run_region(
            cfg,
            region=args.region,
            n_actions=args.n_actions,
            viewer=args.viewer,
            seed=args.seed,
        )
    elif args.mode == "run_all_regions":
        run_all_regions(
            cfg,
            regions=args.regions,
            n_actions_per_region=(args.n_actions if args.n_actions is not None else 1),
            viewer=args.viewer,
            seed=args.seed,
        )
    elif args.mode == "test_run":
        test_run(cfg)


if __name__ == "__main__":
    main()
