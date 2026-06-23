# -*- coding: utf-8 -*-
"""Common helpers for the drag trajectory optimization tutorial.

This file collects the project-specific modeling functions so each step file
can focus only on the new optimization idea introduced in that step.

The terminology follows the current project code:

    q_k in R^6
        Joint configuration at trajectory node k.

    Q = [q_0, ..., q_N]
        Whole joint trajectory. This is the trajectory optimizer's decision
        variable once we start modifying the path.

    p_tip(q)
        Spoon-tip world position from MuJoCo forward kinematics. This is the
        contact/task point we want to drag from drag_start_pos to drag_end_pos.

    h(q) = z_link7(q) - z_spoon_head(q)
        Project head-down metric. If h(q) is positive, the spoon head is below
        the link7/wrist origin. This is the same idea used in
        eating_scoop_system_v11.py.

    sigma_min(J(q))
        Numerical singularity metric. The project uses MuJoCo Jacobians and
        SVD, not symbolic determinant expansion.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from meal_assist.config import SystemConfig
from meal_assist.database import PrimitiveDatabase
from meal_assist.datatypes import ScoopPrimitive
from meal_assist.mathutils import smoothstep5
from meal_assist.robot import RobotModel, mujoco


DEFAULT_N_NODES = 21
DEFAULT_OUT_DIR = Path("tutorial_drag_opt_output")


def load_best_primitive(cfg: SystemConfig) -> ScoopPrimitive:
    """Load one existing primitive from the LUT.

    The primitive gives the current DLS IK keyframes:

        q_0 = q_drag_start
        q_N = q_drag_end

    and the task-space drag endpoints:

        p_ref,0 = drag_start_pos
        p_ref,N = drag_end_pos.

    For learning trajectory optimization, we use this as a small reproducible
    test case instead of solving the full eating pipeline.
    """
    primitives = PrimitiveDatabase(cfg).load()
    return min(primitives, key=lambda p: float(p.score))


def drag_keyframes(primitive: ScoopPrimitive) -> Tuple[np.ndarray, np.ndarray]:
    """Return the drag_start and drag_end joint keyframes.

    Mathematical meaning:

        q_0 = q_drag_start
        q_N = q_drag_end

    In the tutorial optimizers, these endpoints are usually fixed so we can
    focus on improving only the interior trajectory nodes.
    """
    return (
        np.array(primitive.q_drag_start, dtype=float),
        np.array(primitive.q_drag_end, dtype=float),
    )


def drag_task_endpoints(primitive: ScoopPrimitive) -> Tuple[np.ndarray, np.ndarray]:
    """Return the desired spoon-tip drag endpoints in world coordinates.

    Mathematical meaning:

        p_ref,0 = drag_start_pos
        p_ref,N = drag_end_pos

    These are task-space points. They are not joint configurations.
    """
    return (
        np.array(primitive.drag_start_pos, dtype=float),
        np.array(primitive.drag_end_pos, dtype=float),
    )


def smooth_q_path(q0: np.ndarray, q1: np.ndarray, n_nodes: int) -> np.ndarray:
    """Create the baseline smoothstep joint trajectory.

    Baseline interpolation:

        q_k = (1 - s_k) q_0 + s_k q_N

    where

        t_k = k / (N - 1)
        s_k = smoothstep5(t_k) = 6t_k^5 - 15t_k^4 + 10t_k^3.

    This gives smooth endpoint velocity, but it does not optimize task-space
    error, head-down, singularity, collision, or joint margins along the path.
    """
    path = []
    for k in range(n_nodes):
        s = smoothstep5(k / max(n_nodes - 1, 1))
        path.append((1.0 - s) * q0 + s * q1)
    return np.asarray(path, dtype=float)


def straight_tip_path(p0: np.ndarray, p1: np.ndarray, n_nodes: int) -> np.ndarray:
    """Create the desired straight-line spoon-tip path.

    Reference path:

        p_ref,k = (1 - t_k) p_ref,0 + t_k p_ref,N

    where t_k = k / (N - 1).

    The optimizer changes q_k. MuJoCo FK then maps q_k to p_tip(q_k), and the
    cost compares p_tip(q_k) to this reference.
    """
    path = []
    for k in range(n_nodes):
        t = k / max(n_nodes - 1, 1)
        path.append((1.0 - t) * p0 + t * p1)
    return np.asarray(path, dtype=float)


def baseline_drag_path(primitive: ScoopPrimitive, n_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    """Build both baseline q trajectory and target tip trajectory.

    Returns:

        q_path
            Existing smoothstep joint interpolation Q_base.

        target_tip_path
            Desired straight task-space path p_ref,k.
    """
    q0, q1 = drag_keyframes(primitive)
    p0, p1 = drag_task_endpoints(primitive)
    return smooth_q_path(q0, q1, n_nodes), straight_tip_path(p0, p1, n_nodes)


def tip_tracking_cost(robot: RobotModel, q_path: np.ndarray, target_tip_path: np.ndarray) -> float:
    """Compute spoon-tip task tracking cost.

    Forward kinematics:

        p_tip,k = p_tip(q_k)

    Error:

        e_p,k = p_tip(q_k) - p_ref,k

    Cost:

        J_tip(Q) = sum_k ||e_p,k||^2

    This is the most basic task-space objective for drag.
    """
    d = mujoco.MjData(robot.model)
    total = 0.0
    for k, q in enumerate(q_path):
        robot.set_q(d, q)
        err = robot.tip_pos(d) - target_tip_path[k]
        total += float(np.dot(err, err))
    return total


def acceleration_cost(q_path: np.ndarray) -> float:
    """Compute discrete joint acceleration smoothness.

    Discrete second difference:

        ddq_k = q_{k+1} - 2 q_k + q_{k-1}

    Cost:

        J_acc(Q) = sum_k ||ddq_k||^2

    If dt is constant, ddq_k is proportional to qddot_k. This term discourages
    sharp bends in joint space, making the path easier for PD/position control.
    """
    ddq = q_path[2:] - 2.0 * q_path[1:-1] + q_path[:-2]
    return float(np.sum(ddq * ddq))


def head_min_path(cfg: SystemConfig, n_nodes: int) -> np.ndarray:
    """Return desired minimum head-down value during drag.

    Current project metric:

        h(q) = z_link7(q) - z_spoon_head(q)

    One-sided desired condition:

        h(q_k) >= h_min,k

    During drag we taper h_min,k from the stronger drag-start value to the
    weaker drag-end value:

        h_min,k = lerp(head_drop_hard_min_drag_start,
                       head_drop_hard_min_drag_end,
                       t_k)
    """
    return np.linspace(
        float(cfg.head_drop_hard_min_drag_start),
        float(cfg.head_drop_hard_min_drag_end),
        n_nodes,
    )


def head_singularity_terms(
    robot: RobotModel,
    cfg: SystemConfig,
    q_path: np.ndarray,
) -> dict[str, float]:
    """Compute head-down and singularity penalties.

    Head-down penalty:

        e_head,k = max(0, h_min,k - h(q_k))
        J_head   = sum_k e_head,k^2

    Singularity penalty:

        sigma_min,k = smallest singular value of J_task(q_k)
        e_sing,k    = max(0, sigma_threshold - sigma_min,k)
        J_sing      = sum_k e_sing,k^2

    The underlying Jacobian and sigma_min are computed by RobotModel using
    MuJoCo Jacobians:

        J_task = [Jp_spoon_tip ; 0.15 Jr_link7]

    This is numerical singularity checking, not symbolic analysis.
    """
    d = mujoco.MjData(robot.model)
    h_min = head_min_path(cfg, len(q_path))
    j_head = 0.0
    j_sing = 0.0
    min_head = float("inf")
    min_sigma = float("inf")

    for k, q in enumerate(q_path):
        robot.set_q(d, q)
        head_drop = robot.spoon_head_drop(d)
        sigma, _condition = robot.singularity_metrics(d)
        min_head = min(min_head, float(head_drop))
        min_sigma = min(min_sigma, float(sigma))

        head_err = max(0.0, h_min[k] - head_drop)
        sing_err = max(0.0, cfg.min_sigma - sigma)
        j_head += head_err * head_err
        j_sing += sing_err * sing_err

    return {
        "head": float(j_head),
        "sing": float(j_sing),
        "min_head": float(min_head),
        "min_sigma": float(min_sigma),
    }


def tip_positions(robot: RobotModel, q_path: np.ndarray) -> np.ndarray:
    """Map a joint trajectory to visible spoon-tip workspace points.

    The optimizer modifies 6D joint states q_k. A MuJoCo viewer cannot directly
    display 6D joint space, so we visualize:

        p_tip(q_k) in R^3.

    These points are what the tutorial viewer draws.
    """
    d = mujoco.MjData(robot.model)
    pts = []
    for q in q_path:
        robot.set_q(d, q)
        pts.append(robot.tip_pos(d).copy())
    return np.asarray(pts, dtype=float)


def show_tip_path_viewer(
    robot: RobotModel,
    baseline: np.ndarray,
    optimized: np.ndarray,
    title: str,
) -> None:
    """Show baseline and optimized spoon-tip paths in MuJoCo viewer.

    Blue dots:

        p_tip(q_k^baseline)

    Green dots:

        p_tip(q_k^optimized)

    This is only a visualization of the workspace path. The actual optimized
    variables are still the joint configurations q_k.
    """
    baseline_pts = tip_positions(robot, baseline)
    optimized_pts = tip_positions(robot, optimized)
    d = robot.data
    mat = np.eye(3, dtype=np.float64).reshape(-1)
    blue = np.array([0.1, 0.35, 1.0, 0.9], dtype=np.float32)
    green = np.array([0.0, 0.95, 0.25, 0.95], dtype=np.float32)

    with mujoco.viewer.launch_passive(robot.model, d) as viewer:
        print(f"[VIEWER] {title}")
        print("[VIEWER] blue=baseline smoothstep, green=optimized")
        print("[VIEWER] close the MuJoCo viewer window to finish")
        while viewer.is_running():
            scn = viewer.user_scn
            scn.ngeom = 0
            for pts, rgba, radius in [(baseline_pts, blue, 0.006), (optimized_pts, green, 0.005)]:
                for p in pts:
                    if scn.ngeom >= scn.maxgeom:
                        break
                    mujoco.mjv_initGeom(
                        scn.geoms[scn.ngeom],
                        mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([radius, 0.0, 0.0], dtype=np.float64),
                        np.array(p, dtype=np.float64),
                        mat,
                        rgba,
                    )
                    scn.ngeom += 1
            viewer.sync()
