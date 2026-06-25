# -*- coding: utf-8 -*-
"""Step 7: drag + lift 전체를 하나의 optimizer로 동시에 최적화.

Step 6 문제:
    drag와 lift를 따로 최적화 → q_drag_end에서 속도 불연속 (bounce)

Step 7 해결:
    단일 결정변수 x로 drag+lift 전체 내부 노드를 동시에 최적화.
    J_acc = Σ ||ddq_k||² 를 전체 궤적에 계산하므로
    접합부(q_drag_end) 가속도도 자동으로 penalty에 포함 → 연속성 확보.

전체 궤적 구조:
    q0 ──[x1..xm]── q_DE ──[y1..yn]── qL
    (고정)           (고정 waypoint)    (고정)

    고정점: q0 (idx 0), q_DE (idx N_DRAG-1), qL (idx -1)
    가변점: drag interior (N_DRAG-2개) + lift interior (N_LIFT-2개)
    결정변수 차원: (N_DRAG + N_LIFT - 4) * 6

Cost (전체):
    J = w_tip  * (J_tip_drag + J_tip_lift)
      + w_acc  * J_acc        ← 전체 궤적, 접합부 포함
      + w_head * J_head        ← drag 구간만
      + w_sing * J_sing        ← drag 구간만

Constraint (lift 구간):
    ramp hard tilt:
        alpha_k = k / (N_LIFT-1)   (k=1: q_DE 직후, k=N_LIFT-1: qL)
        eps_k   = tilt_DE * (1-alpha_k) + EPS_TILT * alpha_k
        ||n(q_k) × n_ref|| ≤ eps_k   for k=1..N_LIFT-1
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
    pack_fixed_nodes,
    ramp_tilt_constraints,
    tip_tracking_cost,
    unpack_fixed_nodes,
)

# ── 설정 ─────────────────────────────────────────────────────────────────────
N_DRAG = 15
N_LIFT = 15

W_TIP  = 1.0
W_ACC  = 20.0   # head/sing(10.0)보다 강하게 → 접합부 포함 smoothness 우선
W_HEAD = 10.0
W_SING = 10.0


# ── pack / unpack (고정점 3개: q0, q_DE, qL) ─────────────────────────────────

def pack(q_full: np.ndarray, N_DRAG: int) -> np.ndarray:
    """전체 궤적 → optimizer 벡터.

    고정점(idx 0, N_DRAG-1, -1)을 제외한 내부 노드만 추출.

        drag_inner: q_full[1 : N_DRAG-1]       (N_DRAG-2 nodes)
        lift_inner: q_full[N_DRAG : -1]         (N_LIFT-2 nodes)
    """
    fixed = {0, N_DRAG - 1, len(q_full) - 1}
    return pack_fixed_nodes(q_full, fixed)


def unpack(x: np.ndarray, q_full_init: np.ndarray, N_DRAG: int) -> np.ndarray:
    """optimizer 벡터 → 전체 궤적 (고정점 복원).

        [q0 | drag_inner | q_DE | lift_inner | qL]
    """
    fixed = {0, N_DRAG - 1, len(q_full_init) - 1}
    return unpack_fixed_nodes(x, q_full_init, fixed)


# ── cost ──────────────────────────────────────────────────────────────────────

def total_cost(x, robot, cfg, q_full_init, N_DRAG,
               target_drag_tip, target_lift_tip):
    """통합 cost: drag + lift 전체에 걸쳐 계산.

    접합부(q_drag_end)는 J_acc 계산 시 자동으로 포함:
        ddq[N_DRAG-1] = q_full[N_DRAG] - 2*q_DE + q_full[N_DRAG-2]
                        (첫 lift 내부)    (고정)    (마지막 drag 내부)
    """
    q_full = unpack(x, q_full_init, N_DRAG)
    q_drag = q_full[:N_DRAG]          # drag 구간 (q0 ~ q_DE)
    q_lift = q_full[N_DRAG - 1:]      # lift 구간 (q_DE ~ qL, q_DE 공유)

    j_tip  = (tip_tracking_cost(robot, q_drag, target_drag_tip)
              + tip_tracking_cost(robot, q_lift, target_lift_tip))
    j_acc  = acceleration_cost(q_full)     # ← 전체에 한 번만
    hs     = head_singularity_terms(robot, cfg, q_drag)

    return W_TIP * j_tip + W_ACC * j_acc + W_HEAD * hs["head"] + W_SING * hs["sing"]


def cost_breakdown(robot, cfg, q_full, N_DRAG, target_drag_tip, target_lift_tip):
    q_drag = q_full[:N_DRAG]
    q_lift = q_full[N_DRAG - 1:]
    j_tip_drag = tip_tracking_cost(robot, q_drag, target_drag_tip)
    j_tip_lift = tip_tracking_cost(robot, q_lift, target_lift_tip)
    j_acc      = acceleration_cost(q_full)
    hs         = head_singularity_terms(robot, cfg, q_drag)
    total = (W_TIP * (j_tip_drag + j_tip_lift)
             + W_ACC * j_acc
             + W_HEAD * hs["head"]
             + W_SING * hs["sing"])
    return {
        "total": float(total),
        "tip_drag": float(j_tip_drag), "tip_lift": float(j_tip_lift),
        "acc": float(j_acc),
        "head": float(hs["head"]), "sing": float(hs["sing"]),
        "min_head": float(hs["min_head"]), "min_sigma": float(hs["min_sigma"]),
    }


# ── junction 연속성 진단 ──────────────────────────────────────────────────────

def junction_acceleration(q_full, N_DRAG):
    """접합부(q_drag_end) 가속도 벡터 크기."""
    ddq = q_full[N_DRAG] - 2 * q_full[N_DRAG - 1] + q_full[N_DRAG - 2]
    return float(np.linalg.norm(ddq))


# ── tilt constraint (lift 구간 ramp) ─────────────────────────────────────────

def tilt_constraints(x, robot, q_full_init, N_DRAG, N_LIFT,
                     n_ref, eps_tilt, tilt_at_drag_end):
    """lift 구간 ramp hard tilt constraint.

    q_full에서 lift 구간은 index N_DRAG-1 (q_DE) ~ N_DRAG+N_LIFT-2 (qL).
    q_DE는 고정 끝점이므로 constraint에서 제외 (k=0).
    k=1 (첫 lift interior) ~ k=N_LIFT-1 (qL) 에 ramp 적용.
    """
    q_full = unpack(x, q_full_init, N_DRAG)
    q_lift = q_full[N_DRAG - 1:N_DRAG + N_LIFT - 1]
    return ramp_tilt_constraints(robot, q_lift, n_ref, eps_tilt, tilt_at_drag_end)


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

    nd, nl = args.n_drag, args.n_lift

    # ── tilt 진단 및 EPS_TILT 자동 설정 ─────────────────────────────────────
    tilt_at_drag_end = measure_tilt(robot, np.array(prim.q_drag_end), N_REF)
    tilt_at_lift     = measure_tilt(robot, np.array(prim.q_lift),     N_REF)
    EPS_TILT         = tilt_at_lift   # ramp 끝점: q_lift 실제 tilt
    print(f"[DIAG] tilt_at_drag_end={np.degrees(tilt_at_drag_end):.1f}°  "
          f"tilt_at_lift={np.degrees(tilt_at_lift):.1f}°  "
          f"EPS_TILT={np.degrees(EPS_TILT):.1f}°")
    print(f"[DIAG] ramp: {np.degrees(tilt_at_drag_end):.1f}° → {np.degrees(EPS_TILT):.1f}°")

    # ── baseline 궤적 (drag + lift 연결) ─────────────────────────────────────
    q_drag_init, target_drag_tip = baseline_drag_path(prim, nd)
    q_lift_init, target_lift_tip = baseline_lift_path(prim, nl)
    # q_drag_end 중복 제거하여 전체 연결
    q_full_init = np.vstack([q_drag_init, q_lift_init[1:]])  # (nd+nl-1, 6)

    base_terms = cost_breakdown(robot, cfg, q_full_init, nd,
                                target_drag_tip, target_lift_tip)
    base_junc  = junction_acceleration(q_full_init, nd)
    print(f"\n[BASELINE] total={base_terms['total']:.6f}  "
          f"acc={base_terms['acc']:.6f}  junction_ddq={base_junc:.4f}")

    # ── 통합 최적화 ──────────────────────────────────────────────────────────
    x0     = pack(q_full_init, nd)
    bounds = joint_bounds(robot, (nd - 2) + (nl - 2))
    constraints = [{
        "type": "ineq",
        "fun":  tilt_constraints,
        "args": (robot, q_full_init, nd, nl, N_REF, EPS_TILT, tilt_at_drag_end),
    }]

    print(f"\n  결정변수 차원: {len(x0)}  ({nd-2} drag + {nl-2} lift 내부 노드 × 6)")
    print("  Optimizing (joint drag+lift) ...")

    result = minimize(
        total_cost,
        x0,
        args=(robot, cfg, q_full_init, nd, target_drag_tip, target_lift_tip),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 200, "ftol": 1e-6, "disp": True},
    )

    q_full_opt = unpack(result.x, q_full_init, nd)
    opt_terms  = cost_breakdown(robot, cfg, q_full_opt, nd,
                                target_drag_tip, target_lift_tip)
    opt_junc   = junction_acceleration(q_full_opt, nd)

    # lift tilt 검증
    q_lift_opt = q_full_opt[nd - 1:]
    tilt_viol  = lift_tilt_violation(robot, q_lift_opt, N_REF, EPS_TILT, tilt_at_drag_end)

    print(f"\n[OPTIMIZED] total={opt_terms['total']:.6f}  "
          f"acc={opt_terms['acc']:.6f}  junction_ddq={opt_junc:.4f}")
    print(f"  tip(drag)={opt_terms['tip_drag']:.4f}  "
          f"tip(lift)={opt_terms['tip_lift']:.4f}")
    print(f"  head={opt_terms['head']:.4f}  sing={opt_terms['sing']:.4f}")
    print(f"  success={result.success}  iter={result.nit}")

    print(f"\n[JUNCTION] baseline ddq={base_junc:.4f}  →  optimized ddq={opt_junc:.4f}")
    improvement = (base_junc - opt_junc) / max(base_junc, 1e-12) * 100
    print(f"  접합부 가속도 개선: {improvement:.1f}%")

    print(f"\n[LIFT TILT] max={tilt_viol['max_tilt_deg']:.1f}°  "
          f"n_violated={tilt_viol['n_violated']}/{tilt_viol['n_checked']}  "
          f"satisfied={tilt_viol['constraint_satisfied']}")

    # ── 저장 ─────────────────────────────────────────────────────────────────
    DEFAULT_OUT_DIR.mkdir(exist_ok=True)
    out_path = DEFAULT_OUT_DIR / "step7_joint_result.json"
    out_path.write_text(
        json.dumps({
            "n_drag": nd, "n_lift": nl,
            "n_vars": len(x0),
            "baseline": {**base_terms, "junction_ddq": base_junc},
            "optimized": {**opt_terms, "junction_ddq": opt_junc},
            "tilt": tilt_viol,
            "optimizer": {
                "success": bool(result.success),
                "message": str(result.message),
                "nit": int(result.nit),
                "nfev": int(result.nfev),
            },
            "q_full_opt": q_full_opt.tolist(),
        }, indent=2),
        encoding="utf-8",
    )
    print(f"\n  saved → {out_path}")

    if args.viewer:
        print("\n[VIEWER] 파란 창=baseline / 초록 창=optimized")
        animate_two_viewers(robot, q_full_init, q_full_opt,
                            "Step 7: joint drag+lift")


if __name__ == "__main__":
    main()
