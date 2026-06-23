# -*- coding: utf-8 -*-
"""Neutral-pose search and cache management."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .config import SystemConfig
from .database import NeutralDatabase, _neutral_config_snapshot
from .datatypes import NeutralConnector, PoseTarget
from .geometry import TrayGeometry
from .ik import IKSolver
from .robot import RobotModel, mujoco


def compute_q_neutral(
    cfg: SystemConfig,
    tray: TrayGeometry,
    robot: RobotModel,
    ik: IKSolver,
    verbose: bool = True,
) -> Optional[np.ndarray]:
    """Search for a comfortable neutral joint configuration.

    Candidate neutral positions are sampled from ``tray.neutral_points_world``.
    For each candidate, multi-start IK tries to place the spoon tip there while
    keeping the spoon approximately level. The selected configuration minimizes

        score = position_error + tilt_weight * tilt_error
                - margin_weight * joint_limit_margin

    subject to position tolerance, collision, singularity, and joint-limit
    checks. A looser fallback is accepted when no primary candidate is found.
    """
    candidate_positions = tray.neutral_points_world()
    if verbose:
        print(f"[Q_NEUTRAL] neutral candidate positions: {len(candidate_positions)}")

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
            normal_hard=False,
            forward_weight=0.0,
            forward_hard=False,
            head_drop_min=cfg.neutral_head_drop_min
            if cfg.neutral_head_drop_weight > 0
            else None,
            head_drop_weight=cfg.neutral_head_drop_weight,
        )

        for q_seed in seeds_base:
            _, q_sol, metrics = ik.solve_pose(target, q_seed, posture_ref=q_seed)
            pos_err = metrics.get("pos_error", float("inf"))
            tilt = metrics.get("tilt_error", float("inf"))
            contact = int(metrics.get("contact", 0))
            sigma = metrics.get("sigma_min", 0.0)
            condition = metrics.get("condition", float("inf"))
            marg = robot.min_joint_limit_margin_ratio(q_sol)
            head_drop = metrics.get("head_drop", 0.0)

            score = (
                pos_err
                + cfg.neutral_tilt_score_weight * tilt
                - cfg.neutral_margin_score_weight * max(0.0, marg)
            )

            safe_neutral = robot.is_joint_limit_safe_with_margin(
                q_sol,
                cfg.neutral_joint_margin_ratio,
            )

            primary_ok = (
                pos_err <= cfg.neutral_position_tol
                and safe_neutral
                and tilt <= cfg.neutral_max_tilt
                and head_drop >= 0.0
                and contact <= cfg.contact_allowed
                and sigma >= cfg.min_sigma
                and condition <= cfg.max_condition
            )
            if primary_ok and score < best_score:
                best_q = q_sol.copy()
                best_score = score
                best_pos = pos_world.copy()
                best_metrics = metrics
                if marg >= cfg.neutral_good_margin:
                    found_good = True

            fallback_ok = (
                pos_err <= 0.050
                and robot.is_joint_limit_safe_with_margin(q_sol, 0.0)
                and sigma >= cfg.min_sigma
                and condition <= cfg.max_condition
            )
            if fallback_ok and score < fallback_score:
                fallback_q = q_sol.copy()
                fallback_score = score
                fallback_pos = pos_world.copy()
                fallback_metrics = metrics

        if found_good:
            break

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
            print("[Q_NEUTRAL] no valid neutral candidate found")
        return None

    robot.q_neutral = best_q.copy()
    robot.neutral_target_world = (
        best_pos.copy() if best_pos is not None else tray.neutral_pos_world()
    )
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
    """Return True when the cached neutral config matches the current config."""

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


def _neutral_connector_valid(
    cfg: SystemConfig,
    robot: RobotModel,
    connector: NeutralConnector,
) -> bool:
    """Check whether a cached neutral connector is still usable.

    Cache validation is intentionally lighter than full IK search: it checks
    vector size, hard joint limits, and collision. Strict position/tilt scoring
    is handled when the neutral pose is originally built.
    """
    q = np.array(connector.q_neutral, dtype=float)
    if q.shape[0] != robot.model.nq:
        return False
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
    """Load ``q_neutral`` from cache or rebuild it with IK.

    Runtime mode reuses ``neutral.json`` when its config snapshot still matches
    and the stored joint vector is valid for the current model. Build mode, or
    ``use_neutral_lut=False``, forces a fresh search.
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
                reason = "config mismatch" if stale else "cached q failed validation"
                if verbose:
                    print(f"[Q_NEUTRAL REBUILD] {reason}; rebuilding cache.")
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
        print("[Q_NEUTRAL] use_neutral_lut=False; rebuilding online.")

    q_neutral = compute_q_neutral(cfg, tray, robot, ik, verbose=verbose)
    if q_neutral is not None:
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
