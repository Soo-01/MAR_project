# -*- coding: utf-8 -*-
"""MuJoCo robot model wrapper.

`RobotModel` centralizes FK, Jacobian, joint-limit, spoon geometry, and
singularity utilities used by IK, primitive generation, replay, and trajectory
optimization experiments.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import mujoco
    import mujoco.viewer
except Exception as exc:  # pragma: no cover
    mujoco = None
    print("[WARN] Failed to import MuJoCo. Install it with `pip install mujoco`.", exc)

from .config import SystemConfig
from .mathutils import normalize


class RobotModel:
    """Convenience wrapper around the MuJoCo model and data objects."""

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

        # IK uses a no-contact copy so iterative FK/Jacobian evaluation is not
        # perturbed by contact forces during solve_pose().
        self.ik_model = mujoco.MjModel.from_xml_path(str(cfg.xml_path))
        for i in range(self.ik_model.ngeom):
            self.ik_model.geom_contype[i] = 0
            self.ik_model.geom_conaffinity[i] = 0

        # Filled later by neutral cache/build logic.
        self.q_neutral: Optional[np.ndarray] = None
        self.neutral_target_world: Optional[np.ndarray] = None

    def _joint_bounds(self):
        """Read hinge-joint bounds from the MuJoCo model."""
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
        """Clamp qpos to MuJoCo hinge-joint limits."""
        if model is None:
            model = self.model
        for j in range(model.njnt):
            qadr = int(model.jnt_qposadr[j])
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and model.jnt_limited[j]:
                qmin, qmax = model.jnt_range[j]
                d.qpos[qadr] = np.clip(d.qpos[qadr], qmin, qmax)

    def set_q(self, d, q: np.ndarray, model=None):
        """Set joint position, zero velocity, clamp limits, and run FK."""
        if model is None:
            model = self.model
        d.qpos[:model.nq] = q[:model.nq]
        d.qvel[:] = 0.0
        self.enforce_joint_limits(d, model)
        mujoco.mj_forward(model, d)

    def sample_random_q(self, rng: np.random.Generator) -> np.ndarray:
        """Sample a random joint configuration inside the configured margin."""
        q = np.zeros(self.model.nq)
        for i in range(self.model.nq):
            lo, hi = self.q_lower[i], self.q_upper[i]
            margin = self.cfg.joint_limit_margin_ratio * max(hi - lo, 1e-6)
            q[i] = rng.uniform(lo + margin, hi - margin)
        return q

    def is_joint_limit_safe(self, q: np.ndarray) -> bool:
        """Return True when every joint stays inside the default limit margin."""
        for i in range(self.model.nq):
            lo, hi = self.q_lower[i], self.q_upper[i]
            margin = self.cfg.joint_limit_margin_ratio * max(hi - lo, 1e-6)
            if q[i] <= lo + margin or q[i] >= hi - margin:
                return False
        return True

    def is_joint_limit_safe_with_margin(self, q: np.ndarray, margin_ratio: float) -> bool:
        """Return True when every joint stays inside a custom limit margin."""
        for i in range(self.model.nq):
            lo, hi = self.q_lower[i], self.q_upper[i]
            margin = margin_ratio * max(hi - lo, 1e-6)
            if q[i] < lo + margin or q[i] > hi - margin:
                return False
        return True

    def min_joint_limit_margin_ratio(self, q: np.ndarray) -> float:
        """Return the smallest normalized distance to any joint limit."""
        min_margin = float("inf")
        for i in range(self.model.nq):
            lo, hi = self.q_lower[i], self.q_upper[i]
            span = max(hi - lo, 1e-6)
            ratio = min((q[i] - lo) / span, (hi - q[i]) / span)
            min_margin = min(min_margin, float(ratio))
        return min_margin

    def current_body_axis_world(self, d, local_axis: np.ndarray) -> np.ndarray:
        """Transform a link7-local axis into world coordinates."""
        R = d.body(self.ee_id).xmat.reshape(3, 3)
        return normalize(R @ normalize(local_axis))

    def local_point_world(self, d, local_point: np.ndarray) -> np.ndarray:
        """Transform a link7-local point into world coordinates."""
        R = d.body(self.ee_id).xmat.reshape(3, 3)
        p = d.body(self.ee_id).xpos.copy()
        return p + R @ np.array(local_point, dtype=float)

    def link7_origin_world(self, d) -> np.ndarray:
        """Return the world position of the link7 body origin."""
        return d.body(self.ee_id).xpos.copy()

    def spoon_head_world(self, d) -> np.ndarray:
        """Return the world position of the configured spoon head point."""
        return self.local_point_world(d, np.array(self.cfg.spoon_head_local, dtype=float))

    def spoon_head_drop(self, d) -> float:
        """Return h(q) = z_link7 - z_spoon_head.

        Positive values mean the spoon head is below the wrist/link7 origin,
        which is the project's FK-based head-down metric.
        """
        return float(self.link7_origin_world(d)[2] - self.spoon_head_world(d)[2])

    def spoon_pitch_deg(self, d) -> float:
        """Convert head_drop to an approximate spoon pitch angle in degrees."""
        L = float(np.linalg.norm(np.array(self.cfg.spoon_head_local, dtype=float)))
        ratio = self.spoon_head_drop(d) / max(L, 1e-9)
        return float(np.degrees(np.arcsin(np.clip(ratio, -1.0, 1.0))))

    def point_jacobian_world(self, d, local_point: np.ndarray, model=None) -> np.ndarray:
        """Return translational Jacobian for a link7-local point."""
        if model is None:
            model = self.model
        p_world = self.local_point_world(d, np.array(local_point, dtype=float))
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jac(model, d, jacp, jacr, p_world, self.ee_id)
        return jacp[:, :model.nq]

    def link7_origin_jacobian(self, d, model=None) -> np.ndarray:
        """Return translational Jacobian for the link7 origin."""
        if model is None:
            model = self.model
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, d, jacp, jacr, self.ee_id)
        return jacp[:, :model.nq]

    def tip_pos(self, d) -> np.ndarray:
        """Return the world position of the `spoon_tip` site."""
        return d.site(self.spoon_tip_id).xpos.copy()

    def jacobians(self, d, model=None):
        """Return task Jacobian blocks used by IK and singularity checks.

        Jp is the translational Jacobian of the spoon tip. Jr is the rotational
        Jacobian of link7, which carries the spoon orientation.
        """
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
        """Return sigma_min and condition number of the weighted task Jacobian."""
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

    def orientation_errors(
        self,
        d,
        target_normal: np.ndarray,
        target_forward: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        """Return spoon normal and forward-direction alignment errors.

        tilt_error = ||n_current x n_target||.
        forward_error = ||f_current_xy x f_target_xy||.
        """
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
