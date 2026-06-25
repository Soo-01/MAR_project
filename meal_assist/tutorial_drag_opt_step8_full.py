# -*- coding: utf-8 -*-
"""Step 8: 전체 동작 (pre → engage → drag → lift) 통합 최적화.

전체 궤적:
    q_pre → q_engage → q_drag_start → q_drag_end → q_lift

고정 waypoint (5개):
    idx 0           : q_pre
    idx IDX_ENGAGE  : q_engage
    idx IDX_DSTART  : q_drag_start
    idx IDX_DEND    : q_drag_end
    idx IDX_LIFT    : q_lift

결정변수: 5개 waypoint를 제외한 모든 내부 노드

세그먼트별 cost/constraint:
    seg0  pre→engage      : J_tip + J_acc
    seg1  engage→drag_start: J_tip + J_acc
    seg2  drag             : J_tip + J_acc + J_head + J_sing
    seg3  lift             : J_tip + J_acc + tilt ramp hard constraint

J_acc는 전체 궤적에 한 번만 계산 → 모든 waypoint 접합부 자동 포함.
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
    head_singularity_terms,
    joint_bounds,
    lift_tilt_violation,
    load_best_primitive,
    measure_tilt,
    pack_fixed_nodes,
    ramp_tilt_constraints,
    smooth_q_path,
    straight_tip_path,
    tip_tracking_cost,
    unpack_fixed_nodes,
)

# ── 노드 수 ───────────────────────────────────────────────────────────────────
N_SEG  = 10    # pre→engage / engage→drag_start 각각의 노드 수
N_DRAG = 15    # drag_start→drag_end
N_LIFT = 15    # drag_end→lift

# ── Weights ───────────────────────────────────────────────────────────────────
W_TIP  = 1.0
W_ACC  = 30.0
W_HEAD = 10.0
W_SING = 10.0


# ── 전체 궤적 인덱스 계산 ────────────────────────────────────────────────────

def compute_indices(n_seg, n_drag, n_lift):
    """각 waypoint의 q_full 내 인덱스와 세그먼트 범위 반환.

    전체 길이 = n_seg + (n_seg-1) + (n_drag-1) + (n_lift-1)
              = 2*n_seg + n_drag + n_lift - 3
    """
    i_pre    = 0
    i_engage = n_seg - 1
    i_dstart = 2 * n_seg - 2
    i_dend   = 2 * n_seg + n_drag - 3
    i_lift   = 2 * n_seg + n_drag + n_lift - 4
    total    = i_lift + 1
    fixed    = {i_pre, i_engage, i_dstart, i_dend, i_lift}
    return {
        "pre":    i_pre,
        "engage": i_engage,
        "dstart": i_dstart,
        "dend":   i_dend,
        "lift":   i_lift,
        "total":  total,
        "fixed":  fixed,
        # 세그먼트 슬라이스
        "seg0": (i_pre,    i_engage + 1),
        "seg1": (i_engage, i_dstart + 1),
        "seg2": (i_dstart, i_dend   + 1),
        "seg3": (i_dend,   i_lift   + 1),
    }


# ── baseline 전체 궤적 생성 ──────────────────────────────────────────────────

def build_baseline(prim, n_seg, n_drag, n_lift):
    """5개 keyframe을 smoothstep 보간으로 연결한 baseline 궤적."""
    segs = [
        (np.array(prim.q_pre,        dtype=float), np.array(prim.q_engage,     dtype=float), n_seg),
        (np.array(prim.q_engage,     dtype=float), np.array(prim.q_drag_start, dtype=float), n_seg),
        (np.array(prim.q_drag_start, dtype=float), np.array(prim.q_drag_end,   dtype=float), n_drag),
        (np.array(prim.q_drag_end,   dtype=float), np.array(prim.q_lift,       dtype=float), n_lift),
    ]
    parts = []
    for i, (q0, q1, n) in enumerate(segs):
        seg = smooth_q_path(q0, q1, n)
        parts.append(seg if i == 0 else seg[1:])   # 중복 끝점 제거
    return np.vstack(parts)


def build_tip_targets(prim, n_seg, n_drag, n_lift):
    """세그먼트별 직선 tip 목표 경로."""
    def p(attr): return np.array(getattr(prim, attr), dtype=float)
    return {
        "seg0": straight_tip_path(p("pre_scoop_pos"),  p("engage_pos"),    n_seg),
        "seg1": straight_tip_path(p("engage_pos"),     p("drag_start_pos"), n_seg),
        "seg2": straight_tip_path(p("drag_start_pos"), p("drag_end_pos"),   n_drag),
        "seg3": straight_tip_path(p("drag_end_pos"),   p("lift_pos"),       n_lift),
    }


# ── pack / unpack (5개 고정점) ───────────────────────────────────────────────

def pack(q_full: np.ndarray, fixed: set) -> np.ndarray:
    """고정점 제외한 내부 노드만 1-D 벡터로."""
    return pack_fixed_nodes(q_full, fixed)


def unpack(x: np.ndarray, q_full_init: np.ndarray, fixed: set) -> np.ndarray:
    """optimizer 벡터 → 전체 궤적 (고정점 복원)."""
    return unpack_fixed_nodes(x, q_full_init, fixed)


# ── cost ─────────────────────────────────────────────────────────────────────

def total_cost(x, robot, cfg, q_full_init, idx, tips):
    q_full = unpack(x, q_full_init, idx["fixed"])

    s0 = slice(*idx["seg0"]); s1 = slice(*idx["seg1"])
    s2 = slice(*idx["seg2"]); s3 = slice(*idx["seg3"])

    j_tip = (tip_tracking_cost(robot, q_full[s0], tips["seg0"])
           + tip_tracking_cost(robot, q_full[s1], tips["seg1"])
           + tip_tracking_cost(robot, q_full[s2], tips["seg2"])
           + tip_tracking_cost(robot, q_full[s3], tips["seg3"]))

    j_acc = acceleration_cost(q_full)           # 전체, 모든 접합부 포함

    hs    = head_singularity_terms(robot, cfg, q_full[s2])   # drag 구간만

    return W_TIP * j_tip + W_ACC * j_acc + W_HEAD * hs["head"] + W_SING * hs["sing"]


def cost_breakdown(robot, cfg, q_full, idx, tips):
    s0 = slice(*idx["seg0"]); s1 = slice(*idx["seg1"])
    s2 = slice(*idx["seg2"]); s3 = slice(*idx["seg3"])
    hs = head_singularity_terms(robot, cfg, q_full[s2])
    terms = {
        "tip_seg0": float(tip_tracking_cost(robot, q_full[s0], tips["seg0"])),
        "tip_seg1": float(tip_tracking_cost(robot, q_full[s1], tips["seg1"])),
        "tip_drag":  float(tip_tracking_cost(robot, q_full[s2], tips["seg2"])),
        "tip_lift":  float(tip_tracking_cost(robot, q_full[s3], tips["seg3"])),
        "acc":       float(acceleration_cost(q_full)),
        "head":      float(hs["head"]),
        "sing":      float(hs["sing"]),
        "min_head":  float(hs["min_head"]),
        "min_sigma": float(hs["min_sigma"]),
    }
    terms["total"] = (W_TIP * (terms["tip_seg0"] + terms["tip_seg1"]
                               + terms["tip_drag"] + terms["tip_lift"])
                      + W_ACC * terms["acc"]
                      + W_HEAD * terms["head"]
                      + W_SING * terms["sing"])
    return terms


def junction_accelerations(q_full, idx):
    """각 waypoint 접합부 가속도 ||ddq||."""
    result = {}
    for name, i in [("engage", idx["engage"]),
                    ("dstart", idx["dstart"]),
                    ("dend",   idx["dend"])]:
        ddq = q_full[i + 1] - 2 * q_full[i] + q_full[i - 1]
        result[name] = float(np.linalg.norm(ddq))
    return result


# ── tilt constraint (lift 구간 ramp) ─────────────────────────────────────────

def tilt_constraints(x, robot, q_full_init, idx, n_lift, n_ref, eps_tilt, tilt_at_dend):
    """lift 구간 ramp hard tilt constraint.

    q_drag_end (idx["dend"]) 는 고정점이므로 k=0 제외.
    k=1..n_lift-1 에 ramp 적용.
    """
    i0 = idx["dend"]
    q_full = unpack(x, q_full_init, idx["fixed"])
    q_lift = q_full[i0:i0 + n_lift]
    return ramp_tilt_constraints(robot, q_lift, n_ref, eps_tilt, tilt_at_dend)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--n-seg",  type=int, default=N_SEG)
    parser.add_argument("--n-drag", type=int, default=N_DRAG)
    parser.add_argument("--n-lift", type=int, default=N_LIFT)
    args = parser.parse_args()

    cfg   = SystemConfig()
    robot = RobotModel(cfg)
    prim  = load_best_primitive(cfg)

    ns, nd, nl = args.n_seg, args.n_drag, args.n_lift
    idx = compute_indices(ns, nd, nl)

    # ── tilt 설정 ─────────────────────────────────────────────────────────────
    tilt_at_dend = measure_tilt(robot, np.array(prim.q_drag_end), N_REF)
    tilt_at_lift = measure_tilt(robot, np.array(prim.q_lift),     N_REF)
    EPS_TILT     = tilt_at_lift
    print(f"[DIAG] tilt_at_drag_end={np.degrees(tilt_at_dend):.1f}°  "
          f"tilt_at_lift={np.degrees(tilt_at_lift):.1f}°  "
          f"EPS_TILT={np.degrees(EPS_TILT):.1f}°")
    print(f"[DIAG] 전체 궤적 노드 수: {idx['total']}  "
          f"(seg={ns}×2 + drag={nd} + lift={nl} - 3)")
    print(f"[DIAG] 결정변수 차원: {(idx['total'] - 5) * 6}  "
          f"({idx['total'] - 5} 내부 노드 × 6)")

    # ── baseline ──────────────────────────────────────────────────────────────
    q_full_init = build_baseline(prim, ns, nd, nl)
    tips        = build_tip_targets(prim, ns, nd, nl)

    base_terms = cost_breakdown(robot, cfg, q_full_init, idx, tips)
    base_junc  = junction_accelerations(q_full_init, idx)
    print(f"\n[BASELINE] total={base_terms['total']:.4f}  acc={base_terms['acc']:.5f}")
    print(f"  junction ddq — engage:{base_junc['engage']:.4f}  "
          f"drag_start:{base_junc['dstart']:.4f}  drag_end:{base_junc['dend']:.4f}")

    # ── 최적화 ───────────────────────────────────────────────────────────────
    n_var  = idx["total"] - 5
    x0     = pack(q_full_init, idx["fixed"])
    bounds = joint_bounds(robot, n_var)
    constraints = [{
        "type": "ineq",
        "fun":  tilt_constraints,
        "args": (robot, q_full_init, idx, nl, N_REF, EPS_TILT, tilt_at_dend),
    }]

    print(f"\n  Optimizing full sequence ({idx['total']} nodes, {len(x0)} vars) ...")
    result = minimize(
        total_cost,
        x0,
        args=(robot, cfg, q_full_init, idx, tips),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 300, "ftol": 1e-6, "disp": True},
    )

    q_full_opt = unpack(result.x, q_full_init, idx["fixed"])
    opt_terms  = cost_breakdown(robot, cfg, q_full_opt, idx, tips)
    opt_junc   = junction_accelerations(q_full_opt, idx)

    q_lift_seg = q_full_opt[idx["dend"]:]
    tilt_viol  = lift_tilt_violation(robot, q_lift_seg, N_REF, EPS_TILT, tilt_at_dend)

    print(f"\n[OPTIMIZED] total={opt_terms['total']:.4f}  acc={opt_terms['acc']:.5f}  "
          f"success={result.success}  iter={result.nit}")
    print(f"  tip  seg0={opt_terms['tip_seg0']:.4f}  seg1={opt_terms['tip_seg1']:.4f}  "
          f"drag={opt_terms['tip_drag']:.4f}  lift={opt_terms['tip_lift']:.4f}")
    print(f"  head={opt_terms['head']:.4f}  sing={opt_terms['sing']:.4f}  "
          f"min_head={opt_terms['min_head']*1000:.1f}mm")
    print(f"  junction ddq — engage:{opt_junc['engage']:.4f}  "
          f"drag_start:{opt_junc['dstart']:.4f}  drag_end:{opt_junc['dend']:.4f}")
    print(f"\n[LIFT TILT] max={tilt_viol['max_tilt_deg']:.1f}°  "
          f"violated={tilt_viol['n_violated']}/{tilt_viol['n_checked']}  "
          f"satisfied={tilt_viol['constraint_satisfied']}")

    # ── 저장 ─────────────────────────────────────────────────────────────────
    DEFAULT_OUT_DIR.mkdir(exist_ok=True)
    out_path = DEFAULT_OUT_DIR / "step8_full_result.json"
    out_path.write_text(
        json.dumps({
            "n_seg": ns, "n_drag": nd, "n_lift": nl,
            "n_total": idx["total"], "n_vars": len(x0),
            "eps_tilt_deg": float(np.degrees(EPS_TILT)),
            "baseline":  {**base_terms, "junction": base_junc},
            "optimized": {**opt_terms,  "junction": opt_junc},
            "tilt": tilt_viol,
            "optimizer": {
                "success": bool(result.success),
                "message": str(result.message),
                "nit":     int(result.nit),
                "nfev":    int(result.nfev),
            },
            "q_full_opt": q_full_opt.tolist(),
        }, indent=2),
        encoding="utf-8",
    )
    print(f"\n  saved → {out_path}")

    if args.viewer:
        print("\n[VIEWER] 파란 창=baseline / 초록 창=optimized")
        animate_two_viewers(robot, q_full_init, q_full_opt,
                            "Step 8: full sequence")


if __name__ == "__main__":
    main()
