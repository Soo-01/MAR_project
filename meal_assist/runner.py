# -*- coding: utf-8 -*-
"""Replay and low-level tracking utilities for saved scoop primitives."""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import SystemConfig
from .database import MouthConnectorDatabase
from .datatypes import MouthConnector, PoseTarget, ScoopPrimitive, StepResult
from .ik import IKSolver
from .mathutils import normalize, smoothstep5
from .robot import RobotModel, mujoco

class SequenceRunner:
    """Execute planned joint trajectories with MuJoCo position control."""

    def __init__(self, cfg: SystemConfig, robot: RobotModel):
        self.cfg = cfg
        self.robot = robot
        # Spoon-tip trail buffer used by viewer replays.
        self.trail_positions: List[np.ndarray] = []
        # Current mouth target used for dynamic viewer markers.
        self.current_mouth_pos: Optional[np.ndarray] = None

    def primitive_q_list(self, p: ScoopPrimitive) -> List[np.ndarray]:
        return [
            np.array(p.q_pre, dtype=float),
            np.array(p.q_engage, dtype=float),
            np.array(p.q_drag_start, dtype=float),
            np.array(p.q_drag_end, dtype=float),
            np.array(p.q_lift, dtype=float),
        ]

    def sample_neutral_reachable_initial_q(self, rng: np.random.Generator) -> np.ndarray:
        """Sample an initial random posture that can move back to neutral.

        The sample is drawn near ``q_neutral`` instead of across the full joint
        range, so the first controlled move is likely to converge without tray
        collision or excessive joint motion.
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
        """Advance one simulation step with low-pass filtered position control."""
        if self.robot.model.nu > 0:
            d.ctrl[:self.robot.model.nu] = (
                (1.0 - self.cfg.ctrl_filter_alpha) * d.ctrl[:self.robot.model.nu]
                + self.cfg.ctrl_filter_alpha * q_target[:self.robot.model.nu]
            )
        else:
            d.qpos[:self.robot.model.nq] = q_target[:self.robot.model.nq]
        mujoco.mj_step(self.robot.model, d)

    def _step_to_with_level(self, d, q_target: np.ndarray, level_weight: float = 3.0):
        """Advance one carry step with online spoon-level correction.

        The correction projects a rotational Jacobian update into the spoon-tip
        position Jacobian null space, reducing roll/pitch drift while mostly
        preserving the commanded tip path.
        """
        alpha = self.cfg.ctrl_filter_alpha
        nu = self.robot.model.nu
        nq = self.robot.model.nq

        # Apply the base low-pass position command.
        if nu > 0:
            d.ctrl[:nu] = (1.0 - alpha) * d.ctrl[:nu] + alpha * q_target[:nu]
        else:
            d.qpos[:nq] = q_target[:nq]
            mujoco.mj_step(self.robot.model, d)
            return

        # Compute current spoon-normal error against world-up.
        n_tgt = np.array(self.cfg.world_up, dtype=float)
        n_cur = self.robot.current_body_axis_world(
            d, np.array(self.cfg.spoon_normal_local, dtype=float)
        )
        normal_err = np.cross(n_cur, n_tgt)
        tilt = float(np.linalg.norm(normal_err))

        if tilt > 0.008:
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

            # Apply level correction in the translational null space.
            dq_level = N @ (Jr_pinv @ (level_weight * normal_err))
            dq_level = np.clip(dq_level, -0.03, 0.03)

            # Add the correction to actuator commands and clamp to joint limits.
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
                # Runtime monitor: contacts and level-tilt diagnostics.
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
        """Move from the current posture to ``q_target`` with smoothstep timing.

        This avoids instantaneous joint jumps before replaying a primitive.
        ``keep_level=True`` applies the null-space level correction during
        transport moves.
        """
        d = self.robot.data
        if frames is None:
            frames = max(960, 4 * self.cfg.frames_per_segment)
        if q_tolerance is None:
            q_tolerance = self.cfg.move_q_tolerance
        q_start = d.qpos[:self.robot.model.nq].copy()
        q_target = np.array(q_target, dtype=float)
        trail_stride = 4
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
        """Build the mouth-delivery pose target."""
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

    # Sample candidate mouth positions and keep the safest IK solution.
    def solve_mouth_q_multi(
        self,
        seed_q: np.ndarray,
        ik: Optional[IKSolver] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[bool, Optional[np.ndarray], Dict[str, float], Optional[Tuple[float, float, float]]]:
        """Search mouth-position candidates with multi-start IK.

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
        """Return True when the cached mouth config matches the current config."""
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
        """Check whether cached mouth delivery still validates on this model."""
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
        """Fallback single-target IK from lift to mouth delivery.

        The cached mouth connector is the normal runtime path; this method is
        used for standalone validation.
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
        """Execute one full scoop-to-mouth sequence.

        Sequence: current/random -> pre-scoop -> engage -> drag -> lift ->
        neutral -> mouth pre-approach -> mouth delivery -> hold -> neutral.
        """
        q_list = self.primitive_q_list(primitive)

        # A. Current posture, usually neutral, to pre-scoop.
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

        # F. Lift to neutral carry.
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

        # G. Neutral to mouth via cached connector.
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

        # I. Mouth back to neutral.
        print("[STEP I] MOUTH_TO_NEUTRAL")
        final_neutral_result = self.replay_neutral(
            v=v, frames=max(960, 4 * self.cfg.frames_per_segment), keep_level=False
        )
        print("[FULL SEQUENCE DONE]", primitive.primitive_id)
        return final_neutral_result

    def replay_continuous(self, primitive: ScoopPrimitive, v=None, approach_frames: Optional[int] = None) -> StepResult:
        """Connect smoothly to ``q_pre`` and then replay the primitive."""
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
        """Replay one saved primitive.

        Args:
            viewer: Open a new viewer when True.
            v: Reuse an existing viewer instance when provided.
        """
        q_list = self.primitive_q_list(primitive)
        d = self.robot.data
        self.robot.set_q(d, q_list[0])
        if self.robot.model.nu > 0:
            d.ctrl[:self.robot.model.nu] = q_list[0][:self.robot.model.nu]

        if v is not None:
            # Reuse a viewer supplied by run_region/run_all_regions.
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
        """Return smoothly to the cached neutral configuration.

        If the final spoon-tip position is outside ``neutral_position_tol``,
        the controller holds the neutral command for extra settling frames.
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
            # Track head-drop and tilt during replay.
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

        # Validate the final spoon-tip position against the neutral target.
        tip_now = self.robot.tip_pos(d)
        err = float(np.linalg.norm(tip_now - target_world))
        extra_used = 0
        if err <= self.cfg.neutral_position_tol:
            print(
                f"[NEUTRAL DONE] tip={np.round(tip_now, 4).tolist()}, "
                f"target={np.round(target_world, 4).tolist()}, err={err*1000:.1f} mm"
            )
        else:
            # Hold the command for extra settling if the actuator lags.
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
        """Return the neutral target in world coordinates.

        Prefer the position selected by ``compute_q_neutral``; otherwise fall
        back to the configured tray-frame neutral point.
        """
        if hasattr(self.robot, 'neutral_target_world') and self.robot.neutral_target_world is not None:
            return self.robot.neutral_target_world.copy()
        p_tray = np.array(self.cfg.neutral_pos_tray, dtype=float)
        origin = np.array([self.cfg.base_x_offset, self.cfg.base_y_offset, self.cfg.base_z_offset], dtype=float)
        return origin + p_tray
