# -*- coding: utf-8 -*-
"""Step 4: add head-down and singularity penalties.

Step 3 objective:

    J = w_tip J_tip + w_acc J_acc

Step 4 objective:

    J = w_tip  J_tip
      + w_acc  J_acc
      + w_head J_head
      + w_sing J_sing

New terms:

    J_head = sum_k max(0, h_min,k - h(q_k))^2

    J_sing = sum_k max(0, sigma_threshold - sigma_min(J(q_k)))^2

where:

    h(q) = z_link7(q) - z_spoon_head(q)

and sigma_min is computed numerically from the MuJoCo task Jacobian.
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
    animate_two_viewers,
    baseline_drag_path,
    drag_keyframes,
    head_singularity_terms,
    load_best_primitive,
    tip_tracking_cost,
)


N_TRIALS = 800
NOISE_STD = 0.025
W_TIP = 1.0
W_ACC = 0.1
W_HEAD = 10.0
W_SING = 10.0


def cost_terms(
    robot: RobotModel,
    cfg: SystemConfig,
    q_path: np.ndarray,
    target_tip_path: np.ndarray,
) -> dict[str, float]:
    """Compute Step 4 cost terms.

    Tip tracking:

        J_tip = sum_k ||p_tip(q_k) - p_ref,k||^2

    Smoothness:

        J_acc = sum_k ||q_{k+1} - 2q_k + q_{k-1}||^2

    Head-down:

        e_head,k = max(0, h_min,k - h(q_k))
        J_head   = sum_k e_head,k^2

    Singularity:

        e_sing,k = max(0, sigma_threshold - sigma_min(J(q_k)))
        J_sing   = sum_k e_sing,k^2

    Total:

        J = w_tip J_tip + w_acc J_acc + w_head J_head + w_sing J_sing
    """
    j_tip = tip_tracking_cost(robot, q_path, target_tip_path)
    j_acc = acceleration_cost(q_path)
    hs = head_singularity_terms(robot, cfg, q_path)
    total = W_TIP * j_tip + W_ACC * j_acc + W_HEAD * hs["head"] + W_SING * hs["sing"]
    return {
        "total": float(total),
        "tip": float(j_tip),
        "acc": float(j_acc),
        "head": float(hs["head"]),
        "sing": float(hs["sing"]),
        "min_head": float(hs["min_head"]),
        "min_sigma": float(hs["min_sigma"]),
    }


def random_search(
    robot: RobotModel,
    cfg: SystemConfig,
    baseline_path: np.ndarray,
    target_tip_path: np.ndarray,
    q_start: np.ndarray,
    q_goal: np.ndarray,
) -> tuple[np.ndarray, dict[str, float], dict[str, float]]:
    """Optimize interior trajectory nodes with the Step 4 cost.

    The candidate generation is still the same simple rule:

        Q_candidate = Q_best + Delta

    with Gaussian noise only on q_1..q_{N-1}. The difference from Step 3 is
    not the optimizer; it is the definition of "better":

        Step 3: better = lower tip/smoothness cost
        Step 4: better = lower tip/smoothness/head/singularity cost
    """
    rng = np.random.default_rng(13)
    best_path = baseline_path.copy()
    best_terms = cost_terms(robot, cfg, best_path, target_tip_path)
    initial_terms = dict(best_terms)

    for trial in range(N_TRIALS):
        candidate = best_path.copy()
        candidate[1:-1] += rng.normal(0.0, NOISE_STD, size=candidate[1:-1].shape)
        candidate[0] = q_start
        candidate[-1] = q_goal

        terms = cost_terms(robot, cfg, candidate, target_tip_path)
        if terms["total"] < best_terms["total"]:
            best_terms = terms
            best_path = candidate
            print(
                f"  accepted trial={trial:04d}, cost={best_terms['total']:.8f}, "
                f"min_head={best_terms['min_head']*1000:.1f}mm, "
                f"min_sigma={best_terms['min_sigma']:.5f}"
            )

    return best_path, initial_terms, best_terms


def main() -> None:
    """Run Step 4 and optionally show baseline vs optimized path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()

    cfg = SystemConfig()
    robot = RobotModel(cfg)
    primitive = load_best_primitive(cfg)
    q_start, q_goal = drag_keyframes(primitive)
    baseline_path, target_tip_path = baseline_drag_path(primitive, DEFAULT_N_NODES)

    best_path, initial_terms, best_terms = random_search(
        robot,
        cfg,
        baseline_path,
        target_tip_path,
        q_start,
        q_goal,
    )

    DEFAULT_OUT_DIR.mkdir(exist_ok=True)
    out_path = DEFAULT_OUT_DIR / "step4_head_singularity_result.json"
    out_path.write_text(
        json.dumps(
            {
                "primitive_id": primitive.primitive_id,
                "initial_terms": initial_terms,
                "best_terms": best_terms,
                "q_path": best_path.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("[STEP 4] Random search + head_drop + singularity")
    print(f"primitive_id = {primitive.primitive_id}")
    print(f"initial      = {initial_terms}")
    print(f"best         = {best_terms}")
    print(f"saved        = {out_path}")

    if args.viewer:
        animate_two_viewers(robot, baseline_path, best_path, "Step 4 head/singularity")


if __name__ == "__main__":
    main()
