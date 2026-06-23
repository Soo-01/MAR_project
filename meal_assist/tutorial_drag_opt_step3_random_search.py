# -*- coding: utf-8 -*-
"""Step 3: optimize interior trajectory nodes by random search.

This is the smallest possible trajectory optimizer.

Decision variable:

    Q = [q_0, q_1, ..., q_N]

Endpoint constraint:

    q_0 = q_drag_start
    q_N = q_drag_end

Only the interior nodes are modified:

    q_1, ..., q_{N-1}

Cost:

    J(Q) = w_tip J_tip(Q) + w_acc J_acc(Q)

Accept rule:

    if J(Q_candidate) < J(Q_best):
        Q_best <- Q_candidate
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from meal_assist.config import SystemConfig
from meal_assist.robot import RobotModel
from tutorial_drag_opt_common import (
    DEFAULT_N_NODES,
    DEFAULT_OUT_DIR,
    acceleration_cost,
    baseline_drag_path,
    drag_keyframes,
    load_best_primitive,
    show_tip_path_viewer,
    tip_tracking_cost,
)


N_TRIALS = 500
NOISE_STD = 0.035
W_TIP = 1.0
W_ACC = 0.1


def total_cost(robot: RobotModel, q_path: np.ndarray, target_tip_path: np.ndarray) -> float:
    """Compute the Step 3 objective.

    The cost is:

        J(Q) = w_tip J_tip(Q) + w_acc J_acc(Q)

    with:

        J_tip = sum_k ||p_tip(q_k) - p_ref,k||^2

        J_acc = sum_k ||q_{k+1} - 2q_k + q_{k-1}||^2

    This cost does not yet know about head-down, singularity, collision, or
    joint limit margin. It is intentionally minimal.
    """
    return (
        W_TIP * tip_tracking_cost(robot, q_path, target_tip_path)
        + W_ACC * acceleration_cost(q_path)
    )


def random_search(
    robot: RobotModel,
    baseline_path: np.ndarray,
    target_tip_path: np.ndarray,
    q_start: np.ndarray,
    q_goal: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Run the simplest random-search trajectory optimizer.

    Candidate generation:

        Q_candidate = Q_best + Delta

    where Delta is Gaussian noise:

        Delta_k ~ N(0, sigma^2 I)

    but only for interior nodes:

        k = 1, ..., N-1.

    Endpoint projection:

        q_0 <- q_start
        q_N <- q_goal

    Accept rule:

        if J(Q_candidate) < J(Q_best), keep it.

    This is not a powerful optimizer, but it makes the core mechanics of
    trajectory optimization visible.
    """
    rng = np.random.default_rng(13)
    best_path = baseline_path.copy()
    best_cost = total_cost(robot, best_path, target_tip_path)
    initial_cost = best_cost

    for trial in range(N_TRIALS):
        candidate = best_path.copy()
        candidate[1:-1] += rng.normal(0.0, NOISE_STD, size=candidate[1:-1].shape)
        candidate[0] = q_start
        candidate[-1] = q_goal

        candidate_cost = total_cost(robot, candidate, target_tip_path)
        if candidate_cost < best_cost:
            best_cost = candidate_cost
            best_path = candidate
            print(f"  accepted trial={trial:04d}, cost={best_cost:.8f}")

    return best_path, initial_cost, best_cost


def main() -> None:
    """Run Step 3 and optionally show baseline vs optimized path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()

    cfg = SystemConfig()
    robot = RobotModel(cfg)
    primitive = load_best_primitive(cfg)
    q_start, q_goal = drag_keyframes(primitive)
    baseline_path, target_tip_path = baseline_drag_path(primitive, DEFAULT_N_NODES)

    best_path, initial_cost, best_cost = random_search(
        robot,
        baseline_path,
        target_tip_path,
        q_start,
        q_goal,
    )

    DEFAULT_OUT_DIR.mkdir(exist_ok=True)
    out_path = DEFAULT_OUT_DIR / "step3_random_search_result.json"
    out_path.write_text(
        json.dumps(
            {
                "primitive_id": primitive.primitive_id,
                "initial_cost": initial_cost,
                "best_cost": best_cost,
                "q_path": best_path.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("[STEP 3] Random-search trajectory optimization")
    print(f"primitive_id = {primitive.primitive_id}")
    print(f"initial cost = {initial_cost:.8f}")
    print(f"best cost    = {best_cost:.8f}")
    print(f"improvement  = {(initial_cost - best_cost) / max(initial_cost, 1e-12) * 100:.2f}%")
    print(f"saved        = {out_path}")

    if args.viewer:
        show_tip_path_viewer(robot, baseline_path, best_path, "Step 3 random search")


if __name__ == "__main__":
    main()
