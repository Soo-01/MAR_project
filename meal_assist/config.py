# -*- coding: utf-8 -*-
"""Central configuration for the meal-assist robot pipeline.

The values here define the geometry, IK weights, validation thresholds, LUT
paths, and replay parameters used by the modular `meal_assist` package.
Comments are intentionally descriptive rather than historical: version history
lives in docs, while this file should explain what each parameter currently
means.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass
class SystemConfig:
    """Runtime and planning parameters shared across the package."""

    # File paths.
    base_dir: Path = Path(__file__).resolve().parent.parent
    xml_name: str = "robot_model_v5_scene.xml"
    out_dir_name: str = "scoop_lut_output_v11"

    # Tray frame and scene geometry.
    # Tray coordinates use +X toward the far side, -X toward the user, and Y
    # across the tray. Values are expressed in meters.
    tray_x_length: float = 0.21
    tray_y_length: float = 0.28
    base_x_offset: float = 0.075
    base_y_offset: float = 0.13
    base_z_offset: float = 0.0
    tray_surface_z: float = 0.04

    # Neutral pose search region. Neutral is a safe carry/return pose near the
    # robot base with level spoon posture and joint-limit margin.
    neutral_radius: float = 0.08
    neutral_z_min: float = 0.22
    neutral_z_max: float = 0.28
    neutral_grid_xy: int = 7
    neutral_grid_z: int = 3
    neutral_center_tray: Tuple[float, float] = (0.075, -0.02)
    neutral_pos_tray: Tuple[float, float, float] = (0.105, 0.046, 0.32)
    neutral_pause_frames: int = 10
    neutral_position_tol: float = 0.025
    neutral_max_extra_frames: int = 1200
    neutral_normal_weight: float = 0.20
    neutral_head_drop_min: float = 0.0
    neutral_head_drop_weight: float = 0.0
    neutral_max_tilt: float = 0.10
    neutral_tilt_score_weight: float = 1.0
    neutral_joint_margin_ratio: float = 0.015
    neutral_margin_score_weight: float = 3.0
    neutral_good_margin: float = 0.10

    # Mouth target search and delivery constraints.
    mouth_grid_xyz: int = 20
    mouth_radius_xy: float = 0.025
    mouth_radius_z: float = 0.04
    default_mouth_pos_world: Tuple[float, float, float] = (0.0, 0.26, 0.45)
    mouth_forward_world: Tuple[float, float, float] = (-1.0, 0.0, 0.0)
    mouth_normal_weight: float = 0.12
    mouth_forward_weight: float = 0.03
    mouth_approach_frames: int = 960
    mouth_pause_frames: int = 30
    mouth_max_pos_error: float = 0.020
    mouth_max_tilt_error: float = 0.04
    mouth_max_forward_error: float = 0.40
    mouth_joint_limit_margin_ratio: float = 0.015
    mouth_x: float = 0.0
    mouth_y_range: Tuple[float, float] = (0.10, 0.25)
    mouth_y_step: float = 0.02
    mouth_candidate_z_range: Tuple[float, float] = (0.30, 0.60)
    mouth_candidate_z_step: float = 0.05
    mouth_multi_start_seeds: int = 8
    mouth_pre_approach_distance: float = 0.050
    mouth_position_tol: float = 0.010
    mouth_max_extra_frames: int = 1800
    use_mouth_lut: bool = True
    use_neutral_lut: bool = True

    # Runtime monitors used during replay.
    monitor_contacts_during_replay: bool = True
    monitor_tilt_during_replay: bool = True
    abort_on_scoop_level_tilt: bool = False
    tilt_warn_threshold: float = 0.10

    # Scoop primitive generation.
    scoop_drag_direction_world: Tuple[float, float, float] = (-1.0, 0.0, 0.0)
    drag_lengths: Tuple[float, ...] = (0.035, 0.050, 0.070)
    pre_scoop_height: float = 0.075
    engage_height: float = 0.018
    lift_height: float = 0.110
    scoop_start_offsets_x: Tuple[float, ...] = (0.020, 0.035, 0.050)
    scoop_y_offsets: Tuple[float, ...] = (-0.015, 0.0, 0.015)
    food_samples_per_region_axis: int = 5

    # IK solver parameters.
    ik_iters: int = 800
    ik_step_size: float = 0.05
    ik_dq_clip: float = 0.05
    posture_gain: float = 0.012
    normal_weight: float = 0.18
    forward_weight: float = 0.03
    multi_start_trials: int = 12
    random_seed: int = 13

    # Phase-aware wrist posture preferences. Joint 6 is a soft branch
    # preference; FK-based head_drop is the actual head-down metric.
    joint6_index: int = 5
    joint6_soft_enabled: bool = True
    joint6_soft_gain: float = 0.12
    joint6_pref_pre: float = -0.70
    joint6_pref_engage: float = -0.70
    joint6_pref_drag_start: float = -0.55
    joint6_pref_drag_end: float = -0.25
    joint6_pref_lift: float = -0.05
    joint6_pitch_hard_enabled: bool = False
    joint6_head_down_max: float = -0.25
    joint6_pitch_soft_weight: float = 1.5

    # Spoon geometry and head-down constraints.
    # head_drop = z_link7 - z_spoon_head. Positive means the spoon head is below
    # the wrist/link7 origin.
    spoon_head_local: Tuple[float, float, float] = (0.014529, 0.104125, 0.036995)
    head_drop_enabled: bool = True
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
    head_drop_min_lift: float = 0.0
    head_drop_weight_lift: float = 0.0
    head_drop_hard_min_lift: float = -1.0
    head_drop_ik_margin: float = 0.005
    head_drop_score_weight: float = 24.0
    normal_weight_pre: float = 0.00
    normal_weight_engage: float = 0.02
    normal_weight_drag: float = 0.06
    normal_weight_lift: float = 0.45

    # Debug logging controls.
    debug_reject_log: bool = True
    debug_print_every_try: int = 5
    debug_print_first_n_failures: int = 12

    # Spoon local axes expressed in the link7 frame.
    spoon_normal_local: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    spoon_forward_local: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    world_up: Tuple[float, float, float] = (0.0, 0.0, -1.0)

    # Validation thresholds.
    max_pos_error: float = 0.018
    max_tilt_error: float = 0.10
    max_forward_error: float = 0.40
    min_forward_dot: float = 0.85
    min_sigma: float = 0.004
    max_condition: float = 900.0
    joint_limit_margin_ratio: float = 0.015
    contact_allowed: int = 0

    # Tray/region boundary guards for scoop primitive generation.
    scoop_boundary_max_overshoot: float = 0.005
    spoon_outer_margin: float = 0.005

    # Replay and tracking parameters.
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
        """Absolute path to the MuJoCo XML model."""
        return self.base_dir / self.xml_name

    @property
    def out_dir(self) -> Path:
        """Directory for generated LUTs, summaries, and connector caches."""
        return self.base_dir / self.out_dir_name
