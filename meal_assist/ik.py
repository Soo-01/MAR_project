# -*- coding: utf-8 -*-
"""Constrained damped-least-squares inverse kinematics."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .config import SystemConfig
from .datatypes import PoseTarget
from .mathutils import normalize, wrap_angle
from .robot import RobotModel, mujoco


class IKSolver:
    """Solve phase-specific IK targets with soft and hard constraints.

    The task update is a damped least-squares step:

        dq_task = alpha * J^T (J J^T + lambda^2 I)^(-1) e

    where ``e`` stacks spoon-tip position error, spoon-normal error, spoon-
    forward error, and optionally a head-drop constraint. A null-space posture
    term pulls the solution toward a reference posture without directly
    competing with the task-space objective.
    """

    def __init__(self, cfg: SystemConfig, robot: RobotModel):
        self.cfg = cfg
        self.robot = robot

    def damped_pinv(self, J: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Return the adaptive DLS pseudo-inverse and singularity metrics."""
        S = np.linalg.svd(J, compute_uv=False)
        sigma_min = float(np.min(S))
        sigma_max = float(np.max(S))
        condition = sigma_max / (sigma_min + 1e-12)

        threshold = 0.03
        if sigma_min >= threshold:
            damping = 0.001
        else:
            r = 1.0 - sigma_min / threshold
            damping = 0.001 + r * r * (0.08 - 0.001)

        m = J.shape[0]
        J_dls = J.T @ np.linalg.inv(J @ J.T + damping * damping * np.eye(m))
        return J_dls, sigma_min, condition

    def solve_pose(
        self,
        target: PoseTarget,
        seed_q: np.ndarray,
        posture_ref: Optional[np.ndarray] = None,
    ) -> Tuple[bool, np.ndarray, Dict[str, float]]:
        """Solve IK for one named phase target.

        The iteration uses the no-contact IK model for speed, then validates
        the final pose on the full MuJoCo model with collision, singularity,
        joint-limit, posture, and task-space checks.
        """
        ik_m = self.robot.ik_model
        d = mujoco.MjData(ik_m)
        self.robot.set_q(d, seed_q.copy(), ik_m)
        if posture_ref is None:
            posture_ref = self.robot.q_center.copy()

        target_pos = np.array(target.pos, dtype=float)
        target_normal = np.array(target.normal, dtype=float)
        target_forward = np.array(target.forward, dtype=float)

        last = {}
        for _ in range(self.cfg.ik_iters):
            cur_pos = self.robot.tip_pos(d)
            pos_err = target_pos - cur_pos

            n_cur = self.robot.current_body_axis_world(
                d,
                np.array(self.cfg.spoon_normal_local, dtype=float),
            )
            f_cur = self.robot.current_body_axis_world(
                d,
                np.array(self.cfg.spoon_forward_local, dtype=float),
            )
            f_cur_xy = f_cur.copy()
            f_cur_xy[2] = 0.0
            f_cur_xy = normalize(f_cur_xy)
            f_tgt = target_forward.copy()
            f_tgt[2] = 0.0
            f_tgt = normalize(f_tgt)

            normal_err = np.cross(n_cur, normalize(target_normal))
            forward_err = np.cross(f_cur_xy, f_tgt)

            Jp, Jr = self.robot.jacobians(d, ik_m)
            J = np.vstack(
                [
                    Jp,
                    target.normal_weight * Jr,
                    target.forward_weight * Jr,
                ]
            )
            e = np.hstack(
                [
                    pos_err,
                    target.normal_weight * normal_err,
                    target.forward_weight * forward_err,
                ]
            )

            head_drop = self.robot.spoon_head_drop(d)
            head_drop_error = 0.0
            if (
                self.cfg.head_drop_enabled
                and target.head_drop_min is not None
                and target.head_drop_weight > 0.0
            ):
                margin = (
                    self.cfg.head_drop_ik_margin
                    if target.head_drop_hard_min is not None
                    else 0.0
                )
                effective_min = float(target.head_drop_min) + margin
                head_drop_error = max(0.0, effective_min - head_drop)
                if head_drop_error > 0.0:
                    J_link7 = self.robot.link7_origin_jacobian(d, ik_m)
                    J_head = self.robot.point_jacobian_world(
                        d,
                        np.array(self.cfg.spoon_head_local, dtype=float),
                        ik_m,
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

            tilt, fwd, dot_up, dot_forward = self.robot.orientation_errors(
                d,
                target_normal,
                target_forward,
            )
            head_drop = self.robot.spoon_head_drop(d)
            head_drop_error = 0.0
            if target.head_drop_min is not None:
                head_drop_error = max(0.0, float(target.head_drop_min) - head_drop)
            head_drop_hard_error = 0.0
            if target.head_drop_hard_min is not None:
                head_drop_hard_error = max(
                    0.0,
                    float(target.head_drop_hard_min) - head_drop,
                )
            joint6 = (
                float(d.qpos[self.cfg.joint6_index])
                if 0 <= self.cfg.joint6_index < ik_m.nq
                else 0.0
            )
            joint6_pref = (
                float(target.joint6_pref)
                if target.joint6_pref is not None
                else float("nan")
            )
            joint6_error = (
                abs(wrap_angle(joint6_pref - joint6))
                if target.joint6_pref is not None
                else 0.0
            )
            joint6_hard_error = (
                max(0.0, joint6 - float(target.joint6_max))
                if target.joint6_hard and target.joint6_max is not None
                else 0.0
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
                "head_drop_min": float(target.head_drop_min)
                if target.head_drop_min is not None
                else float("nan"),
                "head_drop_error": head_drop_error,
                "head_drop_hard_min": float(target.head_drop_hard_min)
                if target.head_drop_hard_min is not None
                else float("nan"),
                "head_drop_hard_error": head_drop_hard_error,
                "joint6": joint6,
                "joint6_pref": joint6_pref,
                "joint6_error": joint6_error,
                "joint6_max": float(target.joint6_max)
                if target.joint6_max is not None
                else float("nan"),
                "joint6_hard": float(target.joint6_hard),
                "joint6_hard_error": joint6_hard_error,
                "sigma_min": sigma_val,
                "condition": cond_val,
                "contact": 0,
            }

            if (
                last["pos_error"] < self.cfg.max_pos_error
                and (
                    (not target.normal_hard)
                    or last["tilt_error"] < self.cfg.max_tilt_error
                )
                and (
                    target.forward_weight <= 0.0
                    or not target.forward_hard
                    or (
                        last["forward_error"] < self.cfg.max_forward_error
                        and last["dot_forward"] >= self.cfg.min_forward_dot
                    )
                )
                and ((not target.joint6_hard) or last["joint6_hard_error"] <= 0.0)
                and (
                    target.head_drop_hard_min is None
                    or last["head_drop_hard_error"] <= 0.0
                )
            ):
                break

        q = d.qpos[:ik_m.nq].copy()
        ok, val_metrics = self.validate_q_for_target(q, target)
        last["contact"] = val_metrics.get("contact", 0)
        return ok, q, last

    def validate_q_for_target(
        self,
        q: np.ndarray,
        target: PoseTarget,
    ) -> Tuple[bool, Dict[str, float]]:
        """Validate a solved scoop-phase target on the full MuJoCo model."""
        d = mujoco.MjData(self.robot.model)
        self.robot.set_q(d, q)
        target_pos = np.array(target.pos, dtype=float)
        target_normal = np.array(target.normal, dtype=float)
        target_forward = np.array(target.forward, dtype=float)
        pos_error = float(np.linalg.norm(self.robot.tip_pos(d) - target_pos))
        tilt, fwd, dot_up, dot_forward = self.robot.orientation_errors(
            d,
            target_normal,
            target_forward,
        )
        head_drop = self.robot.spoon_head_drop(d)
        head_drop_error = 0.0
        if target.head_drop_min is not None:
            head_drop_error = max(0.0, float(target.head_drop_min) - head_drop)
        head_drop_hard_error = 0.0
        if target.head_drop_hard_min is not None:
            head_drop_hard_error = max(0.0, float(target.head_drop_hard_min) - head_drop)
        joint6 = (
            float(q[self.cfg.joint6_index])
            if 0 <= self.cfg.joint6_index < len(q)
            else 0.0
        )
        joint6_pref = (
            float(target.joint6_pref)
            if target.joint6_pref is not None
            else float("nan")
        )
        joint6_error = (
            abs(wrap_angle(joint6_pref - joint6))
            if target.joint6_pref is not None
            else 0.0
        )
        joint6_hard_error = (
            max(0.0, joint6 - float(target.joint6_max))
            if target.joint6_hard and target.joint6_max is not None
            else 0.0
        )
        sigma, condition = self.robot.singularity_metrics(d)
        contact = int(d.ncon)

        normal_ok = (not target.normal_hard) or (tilt <= self.cfg.max_tilt_error)
        forward_ok = (
            target.forward_weight <= 0.0
            or not target.forward_hard
            or (fwd <= self.cfg.max_forward_error and dot_forward >= self.cfg.min_forward_dot)
        )
        joint6_hard_ok = (not target.joint6_hard) or (joint6_hard_error <= 0.0)
        head_drop_ok = target.head_drop_hard_min is None or head_drop_hard_error <= 0.0
        ok = (
            pos_error <= self.cfg.max_pos_error
            and normal_ok
            and forward_ok
            and joint6_hard_ok
            and head_drop_ok
            and sigma >= self.cfg.min_sigma
            and condition <= self.cfg.max_condition
            and contact <= self.cfg.contact_allowed
            and self.robot.is_joint_limit_safe(q)
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
            "head_drop_min": float(target.head_drop_min)
            if target.head_drop_min is not None
            else float("nan"),
            "head_drop_error": head_drop_error,
            "head_drop_hard_min": float(target.head_drop_hard_min)
            if target.head_drop_hard_min is not None
            else float("nan"),
            "head_drop_hard_error": head_drop_hard_error,
            "joint6": joint6,
            "joint6_pref": joint6_pref,
            "joint6_error": joint6_error,
            "joint6_max": float(target.joint6_max)
            if target.joint6_max is not None
            else float("nan"),
            "joint6_hard": float(target.joint6_hard),
            "joint6_hard_error": joint6_hard_error,
            "sigma_min": sigma,
            "condition": condition,
            "contact": float(contact),
        }
        return ok, metrics

    def validate_q_for_mouth(
        self,
        q: np.ndarray,
        target: PoseTarget,
    ) -> Tuple[bool, Dict[str, float]]:
        """Validate a mouth-delivery target with mouth-specific tolerances.

        Mouth delivery accepts looser orientation constraints than scoop because
        the main requirement is that the spoon tip reaches the mouth target
        while avoiding collision, singularity, and joint-limit saturation.
        """
        d = mujoco.MjData(self.robot.model)
        self.robot.set_q(d, q)
        target_pos = np.array(target.pos, dtype=float)
        target_normal = np.array(target.normal, dtype=float)
        target_forward = np.array(target.forward, dtype=float)
        pos_error = float(np.linalg.norm(self.robot.tip_pos(d) - target_pos))
        tilt, fwd, dot_up, dot_forward = self.robot.orientation_errors(
            d,
            target_normal,
            target_forward,
        )
        sigma, condition = self.robot.singularity_metrics(d)
        contact = int(d.ncon)

        normal_ok = (target.normal_weight <= 0.0) or (
            tilt <= self.cfg.mouth_max_tilt_error
        )
        forward_ok = (
            target.forward_weight <= 0.0
            or (
                fwd <= self.cfg.mouth_max_forward_error
                and dot_forward >= self.cfg.min_forward_dot
            )
        )
        joint_ok = self.robot.is_joint_limit_safe_with_margin(
            q,
            self.cfg.mouth_joint_limit_margin_ratio,
        )
        min_margin_ratio = self.robot.min_joint_limit_margin_ratio(q)
        ok = (
            pos_error <= self.cfg.mouth_max_pos_error
            and normal_ok
            and forward_ok
            and sigma >= self.cfg.min_sigma
            and condition <= self.cfg.max_condition
            and contact <= self.cfg.contact_allowed
            and joint_ok
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
