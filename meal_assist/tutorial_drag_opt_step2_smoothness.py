# -*- coding: utf-8 -*-
"""Step 2: add joint-space smoothness to the cost.

Step 1:

    J_tip(Q) = sum_k ||p_tip(q_k) - p_ref,k||^2

Step 2 adds:

    J_acc(Q) = sum_k ||q_{k+1} - 2q_k + q_{k-1}||^2

Total:

    J(Q) = w_tip J_tip(Q) + w_acc J_acc(Q)

This still does not optimize. It only shows that a trajectory cost can combine
task-space tracking and joint-space smoothness.
"""
from __future__ import annotations

from meal_assist.config import SystemConfig
from meal_assist.robot import RobotModel
from tutorial_drag_opt_common import (
    DEFAULT_N_NODES,
    acceleration_cost,
    baseline_drag_path,
    load_best_primitive,
    tip_tracking_cost,
)


W_TIP = 1.0
W_ACC = 0.1


def total_cost(robot: RobotModel, q_path, target_tip_path):
    """Compute Step 2 weighted cost terms.

    Tip tracking:

        J_tip = sum_k ||p_tip(q_k) - p_ref,k||^2

    Acceleration smoothness:

        ddq_k = q_{k+1} - 2q_k + q_{k-1}
        J_acc = sum_k ||ddq_k||^2

    Weighted total:

        J = w_tip J_tip + w_acc J_acc

    This shows a key trajectory optimization idea: "good" is not one thing.
    We define it by adding cost terms with weights.
    """
    j_tip = tip_tracking_cost(robot, q_path, target_tip_path)
    j_acc = acceleration_cost(q_path)
    return W_TIP * j_tip + W_ACC * j_acc, j_tip, j_acc


def main() -> None:
    """Run Step 2 and print the cost decomposition."""
    cfg = SystemConfig()
    robot = RobotModel(cfg)
    primitive = load_best_primitive(cfg)
    q_path, target_tip_path = baseline_drag_path(primitive, DEFAULT_N_NODES)

    total, j_tip, j_acc = total_cost(robot, q_path, target_tip_path)

    print("[STEP 2] Cost = tip tracking + acceleration smoothness")
    print(f"primitive_id = {primitive.primitive_id}")
    print(f"J_tip        = {j_tip:.8f}")
    print(f"J_acc        = {j_acc:.8f}")
    print(f"J_total      = {total:.8f}")


if __name__ == "__main__":
    main()
