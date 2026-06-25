# -*- coding: utf-8 -*-
"""Step 5: scipy SLSQP gradient-based optimizer for drag trajectory.

Step 4 used random search (800 trials, Gaussian noise).
Step 5 replaces it with scipy.optimize.minimize (SLSQP).

SLSQP (Sequential Least-Squares Programming):
  - 매 스텝마다 cost를 2차로, 제약을 1차로 근사해 작은 QP를 풀어 방향 결정
  - quasi-Newton으로 Hessian 근사 → random search보다 훨씬 빠름
  - bounds와 constraints를 직접 처리

Cost:
    J = w_tip  * J_tip
      + w_acc  * J_acc
      + w_head * J_head
      + w_sing * J_sing
      + w_tilt * J_tilt   ← step 5에서 추가 (drag 구간 soft)

New term:
    J_tilt = sum_k ||n(q_k) x n_ref||^2   (sin^2(theta) 합산)

Constraints:
    bounds: joint limits (hard)
    (lift~mouth 구간 tilt hard constraint → step 6)
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.optimize import minimize

from meal_assist.config import SystemConfig
from meal_assist.robot import RobotModel, mujoco
from tutorial_drag_opt_common import (
    DEFAULT_N_NODES,
    DEFAULT_OUT_DIR,
    N_REF,
    acceleration_cost,
    animate_two_viewers,
    baseline_drag_path,
    drag_keyframes,
    head_singularity_terms,
    joint_bounds,
    load_best_primitive,
    pack_endpoint_path,
    tip_tracking_cost,
    unpack_endpoint_path,
)

# ── Weights ───────────────────────────────────────────────────────────────────
W_TIP  = 1.0
W_ACC  = 1.0
W_HEAD = 10.0
W_SING = 10.0
W_TILT = 0.0    # drag 구간은 tilt 비활성 (head-down 자세가 목적이라 수평 강제 X)
                # lift~mouth 구간 tilt hard constraint는 step6에서 추가

# ── pack / unpack ─────────────────────────────────────────────────────────────

def pack(q_path: np.ndarray) -> np.ndarray:
    """전체 trajectory (N+1, 6) → optimizer 벡터 x ((N-1)*6,).

    양 끝(q_0, q_N)은 고정이므로 내부 노드만 포함.
    """
    return pack_endpoint_path(q_path)


def unpack(x: np.ndarray, q_path: np.ndarray) -> np.ndarray:
    """optimizer 벡터 x → 전체 trajectory (N+1, 6).

    고정 끝점:
        q_0 = q_path[0]  (drag_start)
        q_N = q_path[-1] (drag_end)
    """
    return unpack_endpoint_path(x, q_path)


# ── tilt cost ─────────────────────────────────────────────────────────────────

def tilt_cost(robot: RobotModel, q_path: np.ndarray) -> float:
    """숟가락 수평 유지 cost.

    J_tilt = sum_k ||n(q_k) x n_ref||^2

    n(q_k): world frame에서 spoon normal 방향 (link7 local x축을 FK로 변환)
    n_ref : [0, 0, 1] (world up)
    ||cross||^2 = sin^2(theta) → 수평일 때 0, 기울어질수록 증가
    """
    d = mujoco.MjData(robot.model)
    total = 0.0
    for q in q_path:
        robot.set_q(d, q)
        n_cur = robot.current_body_axis_world(
            d, np.array(robot.cfg.spoon_normal_local, dtype=float)
        )
        tilt_vec = np.cross(n_cur, N_REF)
        total += float(np.dot(tilt_vec, tilt_vec))
    return total / max(len(q_path), 1)   # 노드 수로 나눠 N에 무관한 값으로 정규화


# ── cost function (optimizer에 넘기는 형태) ────────────────────────────────────

def total_cost(
    x: np.ndarray,
    robot: RobotModel,
    cfg: SystemConfig,
    q_init: np.ndarray,
    target_tip_path: np.ndarray,
) -> float:
    """optimizer가 최소화할 scalar cost.

    J = w_tip J_tip + w_acc J_acc + w_head J_head + w_sing J_sing + w_tilt J_tilt
    """
    q_path = unpack(x, q_init)
    j_tip  = tip_tracking_cost(robot, q_path, target_tip_path)
    j_acc  = acceleration_cost(q_path)
    hs     = head_singularity_terms(robot, cfg, q_path)
    j_tilt = tilt_cost(robot, q_path)
    return (
        W_TIP  * j_tip
        + W_ACC  * j_acc
        + W_HEAD * hs["head"]
        + W_SING * hs["sing"]
        + W_TILT * j_tilt
    )


def cost_terms(
    robot: RobotModel,
    cfg: SystemConfig,
    q_path: np.ndarray,
    target_tip_path: np.ndarray,
) -> dict[str, float]:
    """항목별 cost 분해 (로깅·비교용)."""
    j_tip  = tip_tracking_cost(robot, q_path, target_tip_path)
    j_acc  = acceleration_cost(q_path)
    hs     = head_singularity_terms(robot, cfg, q_path)
    j_tilt = tilt_cost(robot, q_path)
    total  = (
        W_TIP  * j_tip
        + W_ACC  * j_acc
        + W_HEAD * hs["head"]
        + W_SING * hs["sing"]
        + W_TILT * j_tilt
    )
    return {
        "total":     float(total),
        "tip":       float(j_tip),
        "acc":       float(j_acc),
        "head":      float(hs["head"]),
        "sing":      float(hs["sing"]),
        "tilt":      float(j_tilt),
        "min_head":  float(hs["min_head"]),
        "min_sigma": float(hs["min_sigma"]),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer", action="store_true", help="MuJoCo viewer로 결과 시각화")
    parser.add_argument("--nodes", type=int, default=DEFAULT_N_NODES, help="trajectory 노드 수")
    args = parser.parse_args()

    cfg   = SystemConfig()
    robot = RobotModel(cfg)
    prim  = load_best_primitive(cfg)
    q_start, q_goal = drag_keyframes(prim)
    q_init, target_tip_path = baseline_drag_path(prim, args.nodes)

    # ── baseline 평가 ─────────────────────────────────────────────────────────
    initial_terms = cost_terms(robot, cfg, q_init, target_tip_path)
    print("[STEP 5] SLSQP drag trajectory optimization")
    print(f"  primitive_id  = {prim.primitive_id}")
    print(f"  N_NODES       = {args.nodes}")
    print(f"  baseline cost = {initial_terms['total']:.6f}")
    print(f"  [baseline]    tip={initial_terms['tip']:.4f}  acc={initial_terms['acc']:.4f}  "
          f"head={initial_terms['head']:.4f}  sing={initial_terms['sing']:.4f}  "
          f"tilt={initial_terms['tilt']:.4f}")
    print(f"  [baseline]    min_head={initial_terms['min_head']*1000:.1f}mm  "
          f"min_sigma={initial_terms['min_sigma']:.5f}")

    # ── SLSQP 최적화 ──────────────────────────────────────────────────────────
    x0     = pack(q_init)
    bounds = joint_bounds(robot, args.nodes - 2)

    print("\n  Optimizing with SLSQP ...")
    result = minimize(
        total_cost,
        x0,
        args=(robot, cfg, q_init, target_tip_path),
        method="SLSQP",
        bounds=bounds,
        options={"maxiter": 100, "ftol": 1e-6, "disp": True},
    )

    q_opt      = unpack(result.x, q_init)
    best_terms = cost_terms(robot, cfg, q_opt, target_tip_path)

    # ── 결과 출력 ─────────────────────────────────────────────────────────────
    print(f"\n  success    = {result.success}  ({result.message})")
    print(f"  iterations = {result.nit}   func_evals = {result.nfev}")
    print(f"  best cost  = {best_terms['total']:.6f}  "
          f"(baseline {initial_terms['total']:.6f}  →  {best_terms['total']:.6f})")
    print(f"  [optimized] tip={best_terms['tip']:.4f}  acc={best_terms['acc']:.4f}  "
          f"head={best_terms['head']:.4f}  sing={best_terms['sing']:.4f}  "
          f"tilt={best_terms['tilt']:.4f}")
    print(f"  min_head  : {initial_terms['min_head']*1000:.1f}mm → {best_terms['min_head']*1000:.1f}mm")
    print(f"  min_sigma : {initial_terms['min_sigma']:.5f} → {best_terms['min_sigma']:.5f}")

    # ── 저장 ──────────────────────────────────────────────────────────────────
    DEFAULT_OUT_DIR.mkdir(exist_ok=True)
    out_path = DEFAULT_OUT_DIR / "step5_slsqp_result.json"
    out_path.write_text(
        json.dumps(
            {
                "primitive_id":  prim.primitive_id,
                "n_nodes":       args.nodes,
                "weights":       {"tip": W_TIP, "acc": W_ACC, "head": W_HEAD,
                                  "sing": W_SING, "tilt": W_TILT},
                "initial_terms": initial_terms,
                "best_terms":    best_terms,
                "optimizer":     {
                    "success": bool(result.success),
                    "message": str(result.message),
                    "nit":     int(result.nit),
                    "nfev":    int(result.nfev),
                },
                "q_path": q_opt.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  saved → {out_path}")

    if args.viewer:
        animate_two_viewers(robot, q_init, q_opt, "Step 5 SLSQP")


if __name__ == "__main__":
    main()
