# -*- coding: utf-8 -*-
"""Step 6: drag + lift 구간 최적화 및 hard tilt constraint 적용.

구간 구성:
    drag: q_drag_start → q_drag_end   (tilt: 제약 없음)
    lift: q_drag_end   → q_lift       (tilt: hard inequality constraint)

Phase별 cost:
    공통: J_tip (task 추종) + J_acc (smoothness)
    drag: + J_head (head-down soft) + J_sing (singularity soft)
    lift: + hard tilt constraint  ||n(q_k) × n_ref|| ≤ EPS_TILT

Hard tilt constraint (SLSQP inequality):
    g_k = EPS_TILT - ||n(q_k) × n_ref|| >= 0   (lift 구간 전 노드)

시각화: animate_two_viewers 로 baseline vs optimized 동기 재생
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.optimize import minimize

from meal_assist.config import SystemConfig
from meal_assist.robot import RobotModel
from tutorial_drag_opt_common import (
    DEFAULT_OUT_DIR,
    N_REF,
    acceleration_cost,
    animate_two_viewers,
    baseline_drag_path,
    baseline_lift_path,
    head_singularity_terms,
    joint_bounds,
    lift_tilt_violation,
    load_best_primitive,
    measure_tilt,
    pack_endpoint_path,
    ramp_tilt_constraints,
    tip_tracking_cost,
    unpack_endpoint_path,
)

# ── 설정 ─────────────────────────────────────────────────────────────────────
N_DRAG = 15          # drag 구간 노드 수
N_LIFT = 15          # lift 구간 노드 수 (q_drag_end 포함, q_lift 포함)

W_TIP   = 1.0
W_ACC   = 0.1
W_HEAD  = 10.0
W_SING  = 10.0

EPS_TILT = np.deg2rad(3.0)   # q_lift 실제 tilt + 이 여유가 EPS_TILT (ramp 끝점)
W_ACC_LIFT = 1.0                    # lift는 drag보다 smoothness 더 강하게 (bouncing 방지)


# ── drag cost ─────────────────────────────────────────────────────────────────

def drag_total_cost(x, robot, cfg, q_init, target_tip_path):
    q_path = unpack_endpoint_path(x, q_init)
    j_tip = tip_tracking_cost(robot, q_path, target_tip_path)
    j_acc = acceleration_cost(q_path)
    hs    = head_singularity_terms(robot, cfg, q_path)
    return W_TIP * j_tip + W_ACC * j_acc + W_HEAD * hs["head"] + W_SING * hs["sing"]


def drag_cost_terms(robot, cfg, q_path, target_tip_path):
    j_tip = tip_tracking_cost(robot, q_path, target_tip_path)
    j_acc = acceleration_cost(q_path)
    hs    = head_singularity_terms(robot, cfg, q_path)
    total = W_TIP * j_tip + W_ACC * j_acc + W_HEAD * hs["head"] + W_SING * hs["sing"]
    return {"total": float(total), "tip": float(j_tip), "acc": float(j_acc),
            "head": float(hs["head"]), "sing": float(hs["sing"]),
            "min_head": float(hs["min_head"]), "min_sigma": float(hs["min_sigma"])}


# ── lift cost + hard tilt constraint ─────────────────────────────────────────

def lift_total_cost(x, robot, q_init, target_tip_path):
    q_path = unpack_endpoint_path(x, q_init)
    j_tip = tip_tracking_cost(robot, q_path, target_tip_path)
    j_acc = acceleration_cost(q_path)
    return W_TIP * j_tip + W_ACC_LIFT * j_acc   # lift는 smoothness 더 강하게


def lift_tilt_constraints(x, robot, q_init, n_ref, eps_tilt, tilt_at_start: float):
    """lift 구간 ramp hard tilt constraint.

    SLSQP inequality: g(x) >= 0

    허용 tilt를 k에 따라 선형으로 감소:
        alpha_k = k / (N-1)               (0=drag_end, 1=q_lift)
        eps_k   = tilt_at_start*(1-alpha) + eps_tilt*alpha

    즉, drag_end 직후는 실제 drag tilt를 허용하다가
    q_lift로 가면서 점차 eps_tilt까지 조여드는 ramp.

    → q_path[0] (q_drag_end) 제외: 고정 끝점이라 constraint 불필요
    """
    q_path = unpack_endpoint_path(x, q_init)
    return ramp_tilt_constraints(robot, q_path, n_ref, eps_tilt, tilt_at_start)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--n-drag", type=int, default=N_DRAG)
    parser.add_argument("--n-lift", type=int, default=N_LIFT)
    args = parser.parse_args()

    cfg   = SystemConfig()
    robot = RobotModel(cfg)
    prim  = load_best_primitive(cfg)

    tilt_at_drag_end = measure_tilt(robot, np.array(prim.q_drag_end), N_REF)
    tilt_at_lift     = measure_tilt(robot, np.array(prim.q_lift),     N_REF)

    # EPS_TILT: ramp의 끝점 (q_lift 도착 시 허용 tilt)
    #   = tilt_at_lift + margin
    # ramp는 tilt_at_drag_end → EPS_TILT 로 감소해야 의미 있음
    EPS_TILT = tilt_at_lift   # ramp 끝점: q_lift 실제 tilt 그대로 사용
    print(f"[DIAG] tilt_at_drag_end={np.degrees(tilt_at_drag_end):.1f}°  "
          f"tilt_at_lift={np.degrees(tilt_at_lift):.1f}°")
    print(f"[DIAG] EPS_TILT = {np.degrees(EPS_TILT):.1f}°")
    print(f"[DIAG] ramp 방향: {np.degrees(tilt_at_drag_end):.1f}° → {np.degrees(EPS_TILT):.1f}° (감소)")

    # ── drag 기준 궤적 ───────────────────────────────────────────────────────
    q_drag_init, target_drag_tip = baseline_drag_path(prim, args.n_drag)
    drag_base_terms = drag_cost_terms(robot, cfg, q_drag_init, target_drag_tip)
    print(f"\n[DRAG] baseline cost = {drag_base_terms['total']:.6f}")

    # ── drag 최적화 ──────────────────────────────────────────────────────────
    x0_drag = pack_endpoint_path(q_drag_init)
    res_drag = minimize(
        drag_total_cost,
        x0_drag,
        args=(robot, cfg, q_drag_init, target_drag_tip),
        method="SLSQP",
        bounds=joint_bounds(robot, args.n_drag - 2),
        options={"maxiter": 100, "ftol": 1e-6, "disp": True},
    )
    q_drag_opt = unpack_endpoint_path(res_drag.x, q_drag_init)
    drag_opt_terms = drag_cost_terms(robot, cfg, q_drag_opt, target_drag_tip)
    print(f"[DRAG] optimized cost = {drag_opt_terms['total']:.6f}  "
          f"(iter={res_drag.nit}, success={res_drag.success})")
    print(f"[DRAG] min_head: {drag_base_terms['min_head']*1000:.1f}mm → "
          f"{drag_opt_terms['min_head']*1000:.1f}mm")

    # ── lift 기준 궤적 ───────────────────────────────────────────────────────
    q_lift_init, target_lift_tip = baseline_lift_path(prim, args.n_lift)

    lift_base_viol = lift_tilt_violation(robot, q_lift_init, N_REF, EPS_TILT, tilt_at_drag_end)
    n_checked = lift_base_viol['n_checked']
    print(f"\n[LIFT] baseline  max_tilt={lift_base_viol['max_tilt_deg']:.1f}°  "
          f"final_tilt={lift_base_viol['final_tilt_deg']:.1f}°  "
          f"n_violated(ramp)={lift_base_viol['n_violated']}/{n_checked}  "
          f"satisfied={lift_base_viol['constraint_satisfied']}")
    print(f"       ramp: {np.degrees(tilt_at_drag_end):.1f}° → {np.degrees(EPS_TILT):.1f}°")

    # ── lift 최적화 (ramp hard tilt constraint) ──────────────────────────────
    x0_lift = pack_endpoint_path(q_lift_init)
    constraints = [{
        "type": "ineq",
        "fun":  lift_tilt_constraints,
        "args": (robot, q_lift_init, N_REF, EPS_TILT, tilt_at_drag_end),
    }]
    res_lift = minimize(
        lift_total_cost,
        x0_lift,
        args=(robot, q_lift_init, target_lift_tip),
        method="SLSQP",
        bounds=joint_bounds(robot, args.n_lift - 2),
        constraints=constraints,
        options={"maxiter": 150, "ftol": 1e-6, "disp": True},
    )
    q_lift_opt = unpack_endpoint_path(res_lift.x, q_lift_init)
    lift_opt_viol = lift_tilt_violation(robot, q_lift_opt, N_REF, EPS_TILT, tilt_at_drag_end)
    print(f"[LIFT] optimized max_tilt={lift_opt_viol['max_tilt_deg']:.1f}°  "
          f"final_tilt={lift_opt_viol['final_tilt_deg']:.1f}°  "
          f"n_violated(ramp)={lift_opt_viol['n_violated']}/{n_checked}  "
          f"satisfied={lift_opt_viol['constraint_satisfied']}  "
          f"(iter={res_lift.nit}, success={res_lift.success})")

    # ── 전체 궤적 연결 (drag + lift, q_drag_end 중복 제거) ──────────────────
    baseline_full = np.vstack([q_drag_init, q_lift_init[1:]])
    optimized_full = np.vstack([q_drag_opt,  q_lift_opt[1:]])

    # ── 저장 ─────────────────────────────────────────────────────────────────
    DEFAULT_OUT_DIR.mkdir(exist_ok=True)
    out_path = DEFAULT_OUT_DIR / "step6_drag_lift_result.json"
    out_path.write_text(
        json.dumps({
            "primitive_id": prim.primitive_id,
            "n_drag": args.n_drag, "n_lift": args.n_lift,
            "n_ref": N_REF.tolist(),
            "eps_tilt_deg": float(np.degrees(EPS_TILT)),
            "drag": {
                "baseline": drag_base_terms,
                "optimized": drag_opt_terms,
                "success": bool(res_drag.success),
            },
            "lift": {
                "baseline_violation": lift_base_viol,
                "optimized_violation": lift_opt_viol,
                "success": bool(res_lift.success),
            },
            "q_drag_opt": q_drag_opt.tolist(),
            "q_lift_opt": q_lift_opt.tolist(),
        }, indent=2),
        encoding="utf-8",
    )
    print(f"\n  saved → {out_path}")

    # ── 시각화 ───────────────────────────────────────────────────────────────
    if args.viewer:
        print("\n[VIEWER] drag + lift 전체 궤적 비교")
        print(f"  파란 창 = baseline  |  초록 창 = optimized")
        animate_two_viewers(robot, baseline_full, optimized_full,
                            "Step 6: drag + lift")


if __name__ == "__main__":
    main()
