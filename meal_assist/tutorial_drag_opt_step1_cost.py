# -*- coding: utf-8 -*-
"""Step 1: compute only the spoon-tip tracking cost.

This step answers the most basic question:

    "Given a joint trajectory Q, how do we measure whether the spoon tip
    follows the desired drag path?"

Cost:

    J_tip(Q) = sum_k ||p_tip(q_k) - p_ref,k||^2

No optimization happens in this file. We only evaluate the existing
smoothstep trajectory from q_drag_start to q_drag_end.
"""
from __future__ import annotations

import numpy as np

from meal_assist.config import SystemConfig
from meal_assist.robot import RobotModel
from tutorial_drag_opt_common import (
    DEFAULT_N_NODES,
    baseline_drag_path,
    load_best_primitive,
    tip_tracking_cost,
)


def main() -> None:
    """Run Step 1.

    Procedure:

        1. Load one primitive from the LUT.
        2. Build baseline Q_base using smoothstep interpolation.
        3. Build desired straight task-space path p_ref,k.
        4. Compute:

               J_tip = sum_k ||p_tip(q_k) - p_ref,k||^2

    The RMS error printed at the end is:

        sqrt(J_tip / N)

    converted to millimeters.
    """
    cfg = SystemConfig()
    robot = RobotModel(cfg)
    primitive = load_best_primitive(cfg)
    q_path, target_tip_path = baseline_drag_path(primitive, DEFAULT_N_NODES)

    cost = tip_tracking_cost(robot, q_path, target_tip_path)

    print("[STEP 1] Tip tracking cost only")
    print(f"primitive_id = {primitive.primitive_id}")
    print(f"N_NODES      = {DEFAULT_N_NODES}")
    print(f"J_tip        = {cost:.8f}")
    print(f"RMS tip err  = {np.sqrt(cost / DEFAULT_N_NODES) * 1000:.2f} mm")


if __name__ == "__main__":
    main()
