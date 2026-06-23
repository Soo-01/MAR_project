# -*- coding: utf-8 -*-
"""Build feasible scoop primitives for each tray food region."""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import SystemConfig
from .datatypes import FoodRegion, PoseTarget, ScoopPrimitive
from .geometry import TrayGeometry
from .ik import IKSolver
from .mathutils import sample_points_in_polygon, smoothstep5, wrap_angle
from .robot import RobotModel, mujoco

class ScoopPrimitiveBuilder:
    """Generate scoop keyframes, solve IK, preview them, and score candidates."""

    def __init__(self, cfg: SystemConfig, tray: TrayGeometry, robot: RobotModel, ik: IKSolver):
        self.cfg = cfg
        self.tray = tray
        self.robot = robot
        self.ik = ik
        self.rng = np.random.default_rng(cfg.random_seed)
        self.target_normal = np.array(cfg.world_up, dtype=float)
        self.target_forward = np.array(cfg.scoop_drag_direction_world, dtype=float)

    def make_pose_targets(self, food_xy_tray: np.ndarray, drag_len: float, start_offset_x: float, y_offset: float) -> Dict[str, PoseTarget]:
        """Create phase targets for one drag primitive.

        The spoon starts on the +X side of the food and drags in the -X
        direction. Scoop phases prefer a head-down posture; lift switches to a
        level transport posture.
        """
        x_food, y_food = float(food_xy_tray[0]), float(food_xy_tray[1] + y_offset)
        z_surface = self.cfg.tray_surface_z

        # Start on the +X side of the food and drag toward -X.
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
        """Return deterministic and random seeds for multi-start IK."""
        seeds = [np.zeros(self.robot.model.nq), self.robot.q_center.copy()]
        for _ in range(self.cfg.multi_start_trials):
            seeds.append(self.robot.sample_random_q(self.rng))
        return seeds

    def classify_ik_failure(self, metrics: Dict[str, float]) -> str:
        """Classify the most likely reason an IK candidate failed validation."""
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
        """Solve the five scoop keyframes and return the best valid sequence."""
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

            # Score combines joint motion, task error, posture error, and
            # singularity margin. Lower is better.
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
        """Preview smooth joint interpolation between keyframes before saving."""
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
        """Enumerate drag candidates in one food region and keep valid primitives."""
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

                        # (a) Keep drag endpoints inside the tray with margin.
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

                        # (b) Limit Y-direction overshoot across region bounds.
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

                        # (c) Limit -X overshoot past the region barrier.
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
