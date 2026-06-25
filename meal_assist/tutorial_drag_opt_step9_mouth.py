# -*- coding: utf-8 -*-
"""Step 9: 전체 동작 + 입 배달까지 통합 최적화.

전체 궤적 (7개 고정 waypoint, 6개 세그먼트):
    q_pre → q_engage → q_drag_start → q_drag_end → q_lift → q_mpre → q_delivery

고정 waypoint:
    idx i_pre     : q_pre          (ScoopPrimitive)
    idx i_engage  : q_engage       (ScoopPrimitive)
    idx i_dstart  : q_drag_start   (ScoopPrimitive)
    idx i_dend    : q_drag_end     (ScoopPrimitive)
    idx i_lift    : q_lift         (ScoopPrimitive)
    idx i_mpre    : q_pre_mouth    (MouthConnector.q_pre)
    idx i_deliver : q_delivery     (MouthConnector.q_delivery)

세그먼트별 cost/constraint:
    seg0  pre→engage           : J_tip + J_acc
    seg1  engage→drag_start    : J_tip + J_acc
    seg2  drag                 : J_tip + J_acc + J_head + J_sing
    seg3  lift                 : J_tip + J_acc + tilt ramp hard constraint
    seg4  carry (lift→mpre)    : J_tip + J_acc + tilt flat hard constraint
    seg5  deliver (mpre→mouth) : J_tip + J_acc + tilt flat hard constraint

tilt constraint 요약:
    seg3: ramp 방식 — q_drag_end 고 tilt에서 q_lift 목표 tilt로 점감
    seg4: flat — EPS_TILT_CARRY 이하 유지 (음식 흘리지 않기 위해)
    seg5: flat — EPS_TILT_CARRY 이하 유지

J_acc는 전체 궤적에 한 번만 계산 → 모든 waypoint 접합부 자동 포함.

MouthConnector 로드:
    MouthConnectorDatabase(cfg).load_payload() → (MouthConnector, config)
    connector.q_pre      : mouth approach joint config
    connector.q_delivery : mouth delivery joint config
    connector.pre_pos    : approach world position
    connector.mouth_pos  : delivery world position (== 입 좌표)
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scipy.optimize import minimize

from meal_assist.config import SystemConfig
from meal_assist.database import MouthConnectorDatabase
from meal_assist.datatypes import MouthConnector
from meal_assist.robot import RobotModel, mujoco
from tutorial_drag_opt_common import (
    DEFAULT_OUT_DIR,
    N_REF,
    acceleration_cost,
    animate_two_viewers,
    flat_tilt_constraints,
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


# ── 플롯 설정 ─────────────────────────────────────────────────────────────────
DT_DEFAULT = 0.05   # 노드 간 시간 간격 (초) — 실제 실행 속도에 맞게 조정

# 세그먼트 배경색 (행동 구간 구분)

SEG_COLORS = {
    "seg0": "#e8e8e8",
    "seg1": "#d0e8ff",
    "seg2": "#fff0b0",
    "seg3": "#c8f0c8",
    "seg4": "#ffe0b0",
    "seg5": "#ffd0d0",
}

SEG_LABELS = {
    "seg0": "pre→engage",
    "seg1": "eng→drag_s",
    "seg2": "drag",
    "seg3": "lift",
    "seg4": "carry",
    "seg5": "deliver",
}
JOINT_COLORS = ["#e6194b","#3cb44b","#4363d8","#f58231","#911eb4","#42d4f4"]

# ── 노드 수 ───────────────────────────────────────────────────────────────────
N_SEG     = 10   # pre→engage, engage→drag_start 각각
N_DRAG    = 15   # drag_start→drag_end
N_LIFT    = 15   # drag_end→q_lift
N_CARRY   = 12   # q_lift→q_mpre
N_DELIVER = 10   # q_mpre→q_delivery

# ── Weights ───────────────────────────────────────────────────────────────────
W_TIP  = 1.0
W_ACC  = 30.0
W_HEAD = 10.0
W_SING = 10.0

OMEGA_MAX = 1.2   # [rad/s] — per-joint angular velocity hard limit


# ── 전체 궤적 인덱스 계산 ────────────────────────────────────────────────────

def compute_indices(n_seg, n_drag, n_lift, n_carry, n_deliver):
    """7개 고정 waypoint 인덱스와 6개 세그먼트 범위 반환.

    전체 길이 = n_seg                  (seg0: pre→engage)
              + (n_seg - 1)             (seg1: engage→dstart, 끝점 중복 제거)
              + (n_drag - 1)            (seg2: dstart→dend)
              + (n_lift - 1)            (seg3: dend→lift)
              + (n_carry - 1)           (seg4: lift→mpre)
              + (n_deliver - 1)         (seg5: mpre→deliver)
              = 2*n_seg + n_drag + n_lift + n_carry + n_deliver - 5
    """
    i_pre     = 0
    i_engage  = n_seg - 1
    i_dstart  = 2 * n_seg - 2
    i_dend    = 2 * n_seg + n_drag - 3
    i_lift    = 2 * n_seg + n_drag + n_lift - 4
    i_mpre    = 2 * n_seg + n_drag + n_lift + n_carry - 5
    i_deliver = 2 * n_seg + n_drag + n_lift + n_carry + n_deliver - 6
    total     = i_deliver + 1
    fixed     = {i_pre, i_engage, i_dstart, i_dend, i_lift, i_mpre, i_deliver}

    return {
        "pre":     i_pre,
        "engage":  i_engage,
        "dstart":  i_dstart,
        "dend":    i_dend,
        "lift":    i_lift,
        "mpre":    i_mpre,
        "deliver": i_deliver,
        "total":   total,
        "fixed":   fixed,
        # 세그먼트 슬라이스 (시작 포함, 끝 포함)
        "seg0": (i_pre,    i_engage  + 1),
        "seg1": (i_engage, i_dstart  + 1),
        "seg2": (i_dstart, i_dend    + 1),
        "seg3": (i_dend,   i_lift    + 1),
        "seg4": (i_lift,   i_mpre    + 1),
        "seg5": (i_mpre,   i_deliver + 1),
    }


# ── MouthConnector 로드 ──────────────────────────────────────────────────────

def load_mouth_connector(cfg) -> MouthConnector:
    """mouth_connector.json에서 MouthConnector 로드."""
    db = MouthConnectorDatabase(cfg)
    connector, config = db.load_payload()
    print(f"[MOUTH] connector_id={connector.connector_id}")
    print(f"  mouth_pos  = {np.array(connector.mouth_pos).round(4)}")
    print(f"  pre_pos    = {np.array(connector.pre_pos).round(4)}")
    print(f"  pos_error  = {connector.pos_error:.4f}  tilt_error={connector.tilt_error:.4f}")
    return connector


# ── baseline 전체 궤적 생성 ──────────────────────────────────────────────────

def build_baseline(prim, connector, n_seg, n_drag, n_lift, n_carry, n_deliver):
    """7개 keyframe을 smoothstep 보간으로 연결한 baseline 궤적."""
    segs = [
        (np.array(prim.q_pre,        dtype=float), np.array(prim.q_engage,     dtype=float), n_seg),
        (np.array(prim.q_engage,     dtype=float), np.array(prim.q_drag_start, dtype=float), n_seg),
        (np.array(prim.q_drag_start, dtype=float), np.array(prim.q_drag_end,   dtype=float), n_drag),
        (np.array(prim.q_drag_end,   dtype=float), np.array(prim.q_lift,       dtype=float), n_lift),
        (np.array(prim.q_lift,       dtype=float), np.array(connector.q_pre,   dtype=float), n_carry),
        (np.array(connector.q_pre,   dtype=float), np.array(connector.q_delivery, dtype=float), n_deliver),
    ]
    parts = []
    for i, (q0, q1, n) in enumerate(segs):
        seg = smooth_q_path(q0, q1, n)
        parts.append(seg if i == 0 else seg[1:])   # 중복 끝점 제거
    return np.vstack(parts)


def build_tip_targets(prim, connector, n_seg, n_drag, n_lift, n_carry, n_deliver):
    """세그먼트별 직선 tip 목표 경로."""
    def p(attr): return np.array(getattr(prim, attr), dtype=float)
    def m(attr): return np.array(getattr(connector, attr), dtype=float)
    return {
        "seg0": straight_tip_path(p("pre_scoop_pos"),  p("engage_pos"),     n_seg),
        "seg1": straight_tip_path(p("engage_pos"),     p("drag_start_pos"), n_seg),
        "seg2": straight_tip_path(p("drag_start_pos"), p("drag_end_pos"),   n_drag),
        "seg3": straight_tip_path(p("drag_end_pos"),   p("lift_pos"),       n_lift),
        "seg4": straight_tip_path(p("lift_pos"),       m("pre_pos"),        n_carry),
        "seg5": straight_tip_path(m("pre_pos"),        m("mouth_pos"),      n_deliver),
    }


# ── pack / unpack (7개 고정점, set 기반) ─────────────────────────────────────

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
    s4 = slice(*idx["seg4"]); s5 = slice(*idx["seg5"])

    j_tip = (tip_tracking_cost(robot, q_full[s0], tips["seg0"])
           + tip_tracking_cost(robot, q_full[s1], tips["seg1"])
           + tip_tracking_cost(robot, q_full[s2], tips["seg2"])
           + tip_tracking_cost(robot, q_full[s3], tips["seg3"])
           + tip_tracking_cost(robot, q_full[s4], tips["seg4"])
           + tip_tracking_cost(robot, q_full[s5], tips["seg5"]))

    j_acc = acceleration_cost(q_full)           # 전체, 모든 접합부 포함

    hs = head_singularity_terms(robot, cfg, q_full[s2])   # drag 구간만

    return W_TIP * j_tip + W_ACC * j_acc + W_HEAD * hs["head"] + W_SING * hs["sing"]


def cost_breakdown(robot, cfg, q_full, idx, tips):
    s0 = slice(*idx["seg0"]); s1 = slice(*idx["seg1"])
    s2 = slice(*idx["seg2"]); s3 = slice(*idx["seg3"])
    s4 = slice(*idx["seg4"]); s5 = slice(*idx["seg5"])
    hs = head_singularity_terms(robot, cfg, q_full[s2])
    terms = {
        "tip_seg0":    float(tip_tracking_cost(robot, q_full[s0], tips["seg0"])),
        "tip_seg1":    float(tip_tracking_cost(robot, q_full[s1], tips["seg1"])),
        "tip_drag":    float(tip_tracking_cost(robot, q_full[s2], tips["seg2"])),
        "tip_lift":    float(tip_tracking_cost(robot, q_full[s3], tips["seg3"])),
        "tip_carry":   float(tip_tracking_cost(robot, q_full[s4], tips["seg4"])),
        "tip_deliver": float(tip_tracking_cost(robot, q_full[s5], tips["seg5"])),
        "acc":         float(acceleration_cost(q_full)),
        "head":        float(hs["head"]),
        "sing":        float(hs["sing"]),
        "min_head":    float(hs["min_head"]),
        "min_sigma":   float(hs["min_sigma"]),
    }
    terms["total"] = (W_TIP * (terms["tip_seg0"] + terms["tip_seg1"]
                               + terms["tip_drag"] + terms["tip_lift"]
                               + terms["tip_carry"] + terms["tip_deliver"])
                      + W_ACC * terms["acc"]
                      + W_HEAD * terms["head"]
                      + W_SING * terms["sing"])
    return terms


def junction_accelerations(q_full, idx):
    """각 waypoint 접합부 가속도 ||ddq||."""
    result = {}
    for name, i in [
        ("engage",  idx["engage"]),
        ("dstart",  idx["dstart"]),
        ("dend",    idx["dend"]),
        ("lift",    idx["lift"]),
        ("mpre",    idx["mpre"]),
    ]:
        ddq = q_full[i + 1] - 2 * q_full[i] + q_full[i - 1]
        result[name] = float(np.linalg.norm(ddq))
    return result


# ── tilt constraint (lift ramp + carry/deliver flat) ─────────────────────────

def tilt_constraints_all(x, robot, q_full_init, idx,
                          n_lift, n_carry, n_deliver,
                          n_ref, eps_tilt, tilt_at_dend):
    """전체 tilt hard constraint.

    seg3 (lift): ramp from tilt_at_dend → eps_tilt  (k=1..n_lift-1)
    seg4 (carry):   flat at eps_tilt                (k=1..n_carry-1)
    seg5 (deliver): flat at eps_tilt                (k=1..n_deliver-1)

    고정 끝점(q_lift, q_mpre, q_delivery)은 optimizer가 변경 불가이므로
    constraint 계산에서 제외하고 진단 출력에서 별도 확인.
    """
    q_full = unpack(x, q_full_init, idx["fixed"])
    g_lift = ramp_tilt_constraints(
        robot,
        q_full[idx["dend"]:idx["dend"] + n_lift],
        n_ref,
        eps_tilt,
        tilt_at_dend,
    )
    g_carry = flat_tilt_constraints(
        robot,
        q_full[idx["lift"]:idx["lift"] + n_carry],
        n_ref,
        eps_tilt,
    )
    g_deliver = flat_tilt_constraints(
        robot,
        q_full[idx["mpre"]:idx["mpre"] + n_deliver],
        n_ref,
        eps_tilt,
    )
    return np.concatenate([g_lift, g_carry, g_deliver])


def velocity_constraints(x, q_full_init, fixed, dt, omega_max):
    """Per-joint angular velocity hard constraint: |dq/dt| <= omega_max.

    Returns array of shape (N-1)*6, each entry >= 0 when satisfied.
    Fixed waypoints are included so junctions are also bounded.
    """
    q  = unpack(x, q_full_init, fixed)           # (N, 6)
    dq = np.abs(np.diff(q, axis=0)) / dt         # (N-1, 6)  [rad/s]
    return (omega_max - dq).ravel()               # >= 0 required


def auto_nodes(q_a: np.ndarray, q_b: np.ndarray,
               base: int, dt: float, omega_max: float) -> int:
    """Return minimum node count so no joint exceeds omega_max between two waypoints.

    Minimum nodes = ceil(max_joint_delta / (omega_max * dt)) + 1
    (the +1 is because N nodes span N-1 intervals).
    Always >= base so the default floor is respected.
    """
    max_delta = float(np.max(np.abs(np.array(q_b, dtype=float)
                                    - np.array(q_a, dtype=float))))
    min_nodes = int(np.ceil(max_delta / (omega_max * dt))) + 1
    return max(base, min_nodes)


def tilt_report(robot, q_full, idx, n_ref, eps_tilt, tilt_at_dend, n_lift, n_carry, n_deliver):
    """각 세그먼트의 tilt 위반 현황 반환."""
    d = mujoco.MjData(robot.model)
    n_local = np.array(robot.cfg.spoon_normal_local, dtype=float)

    def _check_seg(start_idx, n_nodes, eps_fn):
        """start_idx부터 n_nodes개 노드의 tilt 검사."""
        tilts, violations = [], 0
        for k in range(n_nodes):
            robot.set_q(d, q_full[start_idx + k])
            n_cur = robot.current_body_axis_world(d, n_local)
            tilt = float(np.linalg.norm(np.cross(n_cur, n_ref)))
            eps = eps_fn(k, n_nodes)
            tilts.append(tilt)
            if tilt > eps + 1e-6:
                violations += 1
        return {
            "max_tilt_deg": float(np.degrees(max(tilts))),
            "n_violated":   violations,
            "n_checked":    n_nodes,
        }

    def ramp_eps(k, n):
        alpha = k / (n - 1) if n > 1 else 1.0
        return tilt_at_dend * (1.0 - alpha) + eps_tilt * alpha

    def flat_eps(k, n):
        return eps_tilt

    return {
        "lift":    _check_seg(idx["dend"],  n_lift,    ramp_eps),
        "carry":   _check_seg(idx["lift"],  n_carry,   flat_eps),
        "deliver": _check_seg(idx["mpre"],  n_deliver, flat_eps),
    }


# ── 궤적 분석: 속도/가속도/토크/tilt 계산 ───────────────────────────────────

def _compute_kinematics(q_full: np.ndarray, dt: float):
    """유한 차분으로 각속도·각가속도 계산 (rad/s, rad/s²)."""
    vel = np.zeros_like(q_full)
    acc = np.zeros_like(q_full)
    vel[1:-1] = (q_full[2:] - q_full[:-2]) / (2.0 * dt)
    vel[0]    = (q_full[1]  - q_full[0])   / dt
    vel[-1]   = (q_full[-1] - q_full[-2])  / dt
    acc[1:-1] = (q_full[2:] - 2.0 * q_full[1:-1] + q_full[:-2]) / (dt ** 2)
    return vel, acc


def _compute_torques(robot, q_full: np.ndarray, vel: np.ndarray, acc: np.ndarray):
    """MuJoCo inverse dynamics로 각 관절 토크 계산 (Nm).

    data.qpos/qvel/qacc 설정 후 mj_inverse 호출 → qfrc_inverse[:6].
    """
    N  = len(q_full)
    nq = robot.model.nq
    torque = np.zeros((N, 6), dtype=float)
    d = mujoco.MjData(robot.model)
    for k in range(N):
        d.qpos[:nq] = q_full[k][:nq]
        d.qvel[:nq] = vel[k][:nq]
        d.qacc[:nq] = acc[k][:nq]
        mujoco.mj_inverse(robot.model, d)
        torque[k] = d.qfrc_inverse[:6]
    return torque


def _compute_tilt_deg(robot, q_full: np.ndarray, n_ref: np.ndarray):
    """전체 궤적의 숟가락 tilt [deg] 계산."""
    N = len(q_full)
    tilt = np.zeros(N)
    n_local = np.array(robot.cfg.spoon_normal_local, dtype=float)
    d = mujoco.MjData(robot.model)
    for k in range(N):
        robot.set_q(d, q_full[k])
        mujoco.mj_forward(robot.model, d)
        n_cur = robot.current_body_axis_world(d, n_local)
        tilt[k] = float(np.degrees(np.linalg.norm(np.cross(n_cur, n_ref))))
    return tilt


def plot_trajectory_analysis(
    robot,
    q_baseline: np.ndarray,
    q_optimized: np.ndarray,
    idx: dict,
    dt: float,
    eps_tilt: float,
    n_ref: np.ndarray,
    save_path: str | None = None,
):
    """최적화 전후 궤적을 4행 subplot으로 비교.

    행 구성:
        1) 각속도 (rad/s)
        2) 각가속도 (rad/s²)
        3) 관절 토크 (Nm)
        4) Tilt 오차 (deg) + EPS_TILT 기준선

    공통:
        - x축: 시간 [s]
        - 세그먼트 배경색으로 행동 구간 구분
        - 세그먼트 전환점에 수직 점선 + 레이블
        - 실선 = optimized, 점선 = baseline
    """
    N   = len(q_optimized)
    t   = np.arange(N) * dt

    print("[PLOT] 속도/가속도 계산 중...")
    vel_b, acc_b = _compute_kinematics(q_baseline,  dt)
    vel_o, acc_o = _compute_kinematics(q_optimized, dt)

    print("[PLOT] inverse dynamics (토크) 계산 중...")
    tor_b = _compute_torques(robot, q_baseline,  vel_b, acc_b)
    tor_o = _compute_torques(robot, q_optimized, vel_o, acc_o)

    print("[PLOT] tilt 계산 중...")
    tilt_b = _compute_tilt_deg(robot, q_baseline,  n_ref)
    tilt_o = _compute_tilt_deg(robot, q_optimized, n_ref)

    # ── 세그먼트 전환점 시간 ─────────────────────────────────────────────────
    seg_keys = ["seg0","seg1","seg2","seg3","seg4","seg5"]
    junc_idx = {
        "pre":     idx["pre"],
        "engage":  idx["engage"],
        "drag_s":  idx["dstart"],
        "drag_e":  idx["dend"],
        "lift":    idx["lift"],
        "mpre":    idx["mpre"],
        "deliver": idx["deliver"],
    }
    junc_t = {name: i * dt for name, i in junc_idx.items()}

    # ── 세그먼트 구간 (시작t, 끝t, 색, 레이블) ──────────────────────────────
    seg_spans = []
    for key in seg_keys:
        s, e = idx[key]
        seg_spans.append((s * dt, (e - 1) * dt, SEG_COLORS[key], SEG_LABELS[key]))

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    fig.suptitle("Step 9: Trajectory dynamic analysis (baseline=dashed, optimized=solid)",
                 fontsize=13, fontweight="bold")

    row_data = [
        (axes[0], "Angular velocity  [rad/s]",  vel_b,  vel_o),
        (axes[1], "Angular acceleration [rad/s²]", acc_b,  acc_o),
        (axes[2], "Torque  [Nm]",       tor_b,  tor_o),
    ]

    for ax, ylabel, data_b, data_o in row_data:
        # 세그먼트 배경 shading
        for (ts, te, color, _) in seg_spans:
            ax.axvspan(ts, te, color=color, alpha=0.45, linewidth=0)
        # 데이터 (6관절)
        for j in range(6):
            c = JOINT_COLORS[j]
            ax.plot(t, data_b[:, j], color=c, lw=0.9, ls="--", alpha=0.55)
            ax.plot(t, data_o[:, j], color=c, lw=1.2, ls="-",  label=f"J{j+1}")
        # 전환점 수직선
        for name, tv in junc_t.items():
            ax.axvline(tv, color="#444", lw=0.8, ls=":")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, axis="y", lw=0.4, alpha=0.5)
        ax.tick_params(labelsize=8)

    # ── Row 4: Tilt ──────────────────────────────────────────────────────────
    ax = axes[3]
    for (ts, te, color, label) in seg_spans:
        ax.axvspan(ts, te, color=color, alpha=0.45, linewidth=0)
        ax.text((ts + te) / 2, -0.5, label,
                ha="center", va="top", fontsize=7, color="#555",
                transform=ax.get_xaxis_transform())
    ax.plot(t, tilt_b, color="#888",    lw=1.0, ls="--", alpha=0.7, label="baseline")
    ax.plot(t, tilt_o, color="#d62728", lw=1.4, ls="-",             label="optimized")
    ax.axhline(np.degrees(eps_tilt), color="navy", lw=1.0, ls="-.",
               label=f"EPS_TILT={np.degrees(eps_tilt):.1f}°")
    for name, tv in junc_t.items():
        ax.axvline(tv, color="#444", lw=0.8, ls=":")
    ax.set_ylabel("Tilt error [deg]", fontsize=9)
    ax.set_xlabel("Time [s]", fontsize=10)
    ax.grid(True, axis="y", lw=0.4, alpha=0.5)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, loc="upper right")

    # ── x축 전환점 tick ──────────────────────────────────────────────────────
    xtick_times  = sorted(junc_t.values())
    xtick_labels = [name for name, _ in sorted(junc_t.items(), key=lambda x: x[1])]
    for ax in axes:
        ax.set_xticks(xtick_times)
        ax.set_xticklabels(xtick_labels, fontsize=7.5, rotation=20, ha="right")
        ax.set_xlim(t[0], t[-1])

    # ── 관절 범례 (Row 1에만 표시) ─────────────────────────────────────────
    handles = [mpatches.Patch(color=JOINT_COLORS[j], label=f"J{j+1}") for j in range(6)]
    handles += [
        plt.Line2D([0], [0], color="gray", ls="--", lw=1, label="baseline"),
        plt.Line2D([0], [0], color="gray", ls="-",  lw=1.3, label="optimized"),
    ]
    axes[0].legend(handles=handles, fontsize=7.5, loc="upper right",
                   ncol=4, framealpha=0.8)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[PLOT] 저장 → {save_path}")
    else:
        plt.show()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer",    action="store_true")
    parser.add_argument("--n-seg",     type=int,   default=N_SEG)
    parser.add_argument("--n-drag",    type=int,   default=N_DRAG)
    parser.add_argument("--n-lift",    type=int,   default=N_LIFT)
    parser.add_argument("--n-carry",   type=int,   default=N_CARRY)
    parser.add_argument("--n-deliver", type=int,   default=N_DELIVER)
    parser.add_argument("--plot",      action="store_true",
                        help="최적화 후 동역학 분석 그래프 표시")
    parser.add_argument("--plot-save", type=str,   default=None,
                        help="그래프 저장 경로 (예: step9_analysis.png)")
    parser.add_argument("--dt",        type=float, default=DT_DEFAULT,
                        help=f"노드 간 시간 간격 [s] (기본 {DT_DEFAULT})")
    args = parser.parse_args()

    cfg   = SystemConfig()
    robot = RobotModel(cfg)
    prim  = load_best_primitive(cfg)

    # ── MouthConnector 로드 ───────────────────────────────────────────────────
    connector = load_mouth_connector(cfg)
    q_mpre    = np.array(connector.q_pre,      dtype=float)
    q_deliver = np.array(connector.q_delivery,  dtype=float)

    # ── 노드 수 자동 산정 ─────────────────────────────────────────────────────
    # 각 세그먼트의 waypoint 간 최대 관절 이동량을 기반으로,
    # omega_max * dt 한 스텝에 이동 가능한 양을 넘지 않도록 노드 수 자동 확장.
    q_pre_w    = np.array(prim.q_pre,        dtype=float)
    q_engage_w = np.array(prim.q_engage,     dtype=float)
    q_dstart_w = np.array(prim.q_drag_start, dtype=float)
    q_dend_w   = np.array(prim.q_drag_end,   dtype=float)
    q_lift_w   = np.array(prim.q_lift,       dtype=float)

    # seg0/seg1은 동일한 n_seg 파라미터를 공유하므로 두 구간 중 더 큰 값 채택
    ns_raw = max(
        auto_nodes(q_pre_w,    q_engage_w, args.n_seg,     args.dt, OMEGA_MAX),
        auto_nodes(q_engage_w, q_dstart_w, args.n_seg,     args.dt, OMEGA_MAX),
    )
    nd_raw = auto_nodes(q_dstart_w, q_dend_w,   args.n_drag,    args.dt, OMEGA_MAX)
    nl_raw = auto_nodes(q_dend_w,   q_lift_w,   args.n_lift,    args.dt, OMEGA_MAX)
    nc_raw = auto_nodes(q_lift_w,   q_mpre,     args.n_carry,   args.dt, OMEGA_MAX)
    nv_raw = auto_nodes(q_mpre,     q_deliver,  args.n_deliver, args.dt, OMEGA_MAX)

    def _node_log(name, base, final):
        flag = f"  (expanded from {base})" if final > base else ""
        print(f"  {name:12s}: {final} nodes{flag}")

    print(f"\n[NODES]  omega_max={OMEGA_MAX} rad/s  dt={args.dt} s")
    _node_log("seg0/1 (ns)", args.n_seg,     ns_raw)
    _node_log("drag   (nd)", args.n_drag,    nd_raw)
    _node_log("lift   (nl)", args.n_lift,    nl_raw)
    _node_log("carry  (nc)", args.n_carry,   nc_raw)
    _node_log("deliver(nv)", args.n_deliver, nv_raw)

    ns, nd, nl, nc, nv = ns_raw, nd_raw, nl_raw, nc_raw, nv_raw

    idx = compute_indices(ns, nd, nl, nc, nv)

    # ── tilt 설정 ─────────────────────────────────────────────────────────────
    tilt_at_dend    = measure_tilt(robot, np.array(prim.q_drag_end), N_REF)
    tilt_at_lift    = measure_tilt(robot, np.array(prim.q_lift),     N_REF)
    tilt_at_mpre    = measure_tilt(robot, q_mpre,                    N_REF)
    tilt_at_deliver = measure_tilt(robot, q_deliver,                 N_REF)
    EPS_TILT = tilt_at_lift   # lift 끝점 실제 tilt → carry/deliver 기준

    print(f"\n[TILT DIAG]")
    print(f"  drag_end  : {np.degrees(tilt_at_dend):.1f}°")
    print(f"  lift      : {np.degrees(tilt_at_lift):.1f}°  ← EPS_TILT")
    print(f"  mpre      : {np.degrees(tilt_at_mpre):.1f}°  (fixed, 참고용)")
    print(f"  delivery  : {np.degrees(tilt_at_deliver):.1f}°  (fixed, 참고용)")
    print(f"  ramp: {np.degrees(tilt_at_dend):.1f}° → {np.degrees(EPS_TILT):.1f}°  "
          f"then flat at {np.degrees(EPS_TILT):.1f}°")

    if tilt_at_mpre > EPS_TILT + 0.05:
        print(f"  [WARN] q_mpre tilt={np.degrees(tilt_at_mpre):.1f}° > EPS_TILT — "
              f"carry 구간 마지막이 제약 불충족 상태. carry 제약은 내부 노드만 적용.")
    if tilt_at_deliver > EPS_TILT + 0.05:
        print(f"  [WARN] q_delivery tilt={np.degrees(tilt_at_deliver):.1f}° > EPS_TILT — "
              f"deliver 구간 마지막이 제약 불충족 상태.")

    print(f"\n[IDX] 전체 궤적 노드 수: {idx['total']}")
    print(f"  fixed waypoints (7개): pre={idx['pre']} engage={idx['engage']} "
          f"dstart={idx['dstart']} dend={idx['dend']} "
          f"lift={idx['lift']} mpre={idx['mpre']} deliver={idx['deliver']}")
    n_var = idx["total"] - 7
    print(f"  결정변수: {n_var} 노드 × 6 = {n_var * 6} 차원")

    # ── baseline ──────────────────────────────────────────────────────────────
    q_full_init = build_baseline(prim, connector, ns, nd, nl, nc, nv)
    tips        = build_tip_targets(prim, connector, ns, nd, nl, nc, nv)

    assert len(q_full_init) == idx["total"], (
        f"baseline 길이 불일치: {len(q_full_init)} != {idx['total']}"
    )

    base_terms = cost_breakdown(robot, cfg, q_full_init, idx, tips)
    base_junc  = junction_accelerations(q_full_init, idx)
    base_tilt  = tilt_report(robot, q_full_init, idx, N_REF, EPS_TILT,
                              tilt_at_dend, nl, nc, nv)

    print(f"\n[BASELINE] total={base_terms['total']:.4f}  acc={base_terms['acc']:.5f}")
    print(f"  tip  seg0={base_terms['tip_seg0']:.4f}  seg1={base_terms['tip_seg1']:.4f}  "
          f"drag={base_terms['tip_drag']:.4f}  lift={base_terms['tip_lift']:.4f}  "
          f"carry={base_terms['tip_carry']:.4f}  deliver={base_terms['tip_deliver']:.4f}")
    print(f"  junction ddq — engage:{base_junc['engage']:.4f}  "
          f"dstart:{base_junc['dstart']:.4f}  dend:{base_junc['dend']:.4f}  "
          f"lift:{base_junc['lift']:.4f}  mpre:{base_junc['mpre']:.4f}")
    for seg, info in base_tilt.items():
        print(f"  tilt[{seg}] max={info['max_tilt_deg']:.1f}°  "
              f"violated={info['n_violated']}/{info['n_checked']}")

    # ── 최적화 ───────────────────────────────────────────────────────────────
    x0     = pack(q_full_init, idx["fixed"])
    bounds = joint_bounds(robot, n_var)
    constraints = [
        {
            "type": "ineq",
            "fun":  tilt_constraints_all,
            "args": (robot, q_full_init, idx, nl, nc, nv, N_REF, EPS_TILT, tilt_at_dend),
        },
        {
            "type": "ineq",
            "fun":  velocity_constraints,
            "args": (q_full_init, idx["fixed"], args.dt, OMEGA_MAX),
        },
    ]

    n_tilt_constraints = (nl - 1) + (nc - 1) + (nv - 1)
    n_vel_constraints  = (idx["total"] - 1) * 6
    print(f"\n  Optimizing full sequence + mouth delivery ...")
    print(f"  {idx['total']} nodes, {len(x0)} vars")
    print(f"  tilt constraints: {n_tilt_constraints}")
    print(f"  velocity constraints: {n_vel_constraints}  (OMEGA_MAX={OMEGA_MAX} rad/s)")

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
    opt_tilt   = tilt_report(robot, q_full_opt, idx, N_REF, EPS_TILT,
                              tilt_at_dend, nl, nc, nv)

    print(f"\n[OPTIMIZED] total={opt_terms['total']:.4f}  acc={opt_terms['acc']:.5f}  "
          f"success={result.success}  iter={result.nit}")
    print(f"  tip  seg0={opt_terms['tip_seg0']:.4f}  seg1={opt_terms['tip_seg1']:.4f}  "
          f"drag={opt_terms['tip_drag']:.4f}  lift={opt_terms['tip_lift']:.4f}  "
          f"carry={opt_terms['tip_carry']:.4f}  deliver={opt_terms['tip_deliver']:.4f}")
    print(f"  head={opt_terms['head']:.4f}  sing={opt_terms['sing']:.4f}  "
          f"min_head={opt_terms['min_head']*1000:.1f}mm")
    print(f"  junction ddq — engage:{opt_junc['engage']:.4f}  "
          f"dstart:{opt_junc['dstart']:.4f}  dend:{opt_junc['dend']:.4f}  "
          f"lift:{opt_junc['lift']:.4f}  mpre:{opt_junc['mpre']:.4f}")
    for seg, info in opt_tilt.items():
        sat = info['n_violated'] == 0
        print(f"  tilt[{seg}] max={info['max_tilt_deg']:.1f}°  "
              f"violated={info['n_violated']}/{info['n_checked']}  "
              f"satisfied={sat}")

    # ── 저장 ─────────────────────────────────────────────────────────────────
    DEFAULT_OUT_DIR.mkdir(exist_ok=True)
    out_path = DEFAULT_OUT_DIR / "step9_mouth_result.json"
    out_path.write_text(
        json.dumps({
            "n_seg": ns, "n_drag": nd, "n_lift": nl,
            "n_carry": nc, "n_deliver": nv,
            "n_total": idx["total"], "n_vars": len(x0),
            "dt": args.dt, "omega_max": OMEGA_MAX,
            "eps_tilt_deg": float(np.degrees(EPS_TILT)),
            "tilt_waypoints_deg": {
                "drag_end": float(np.degrees(tilt_at_dend)),
                "lift":     float(np.degrees(tilt_at_lift)),
                "mpre":     float(np.degrees(tilt_at_mpre)),
                "delivery": float(np.degrees(tilt_at_deliver)),
            },
            "baseline":  {**base_terms, "junction": base_junc, "tilt": base_tilt},
            "optimized": {**opt_terms,  "junction": opt_junc,  "tilt": opt_tilt},
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

    # ── q_delivery까지 궤적 segment 분리 저장 ────────────────────────────────
    seg_out = DEFAULT_OUT_DIR / "step9_segments.json"
    seg_out.write_text(
        json.dumps({
            "q_pre_to_engage":   q_full_opt[slice(*idx["seg0"])].tolist(),
            "q_engage_to_dstart": q_full_opt[slice(*idx["seg1"])].tolist(),
            "q_drag":            q_full_opt[slice(*idx["seg2"])].tolist(),
            "q_lift":            q_full_opt[slice(*idx["seg3"])].tolist(),
            "q_carry":           q_full_opt[slice(*idx["seg4"])].tolist(),
            "q_deliver":         q_full_opt[slice(*idx["seg5"])].tolist(),
        }, indent=2),
        encoding="utf-8",
    )
    print(f"  segments → {seg_out}")

    if args.viewer:
        print("\n[VIEWER] 파란 창=baseline / 초록 창=optimized")
        animate_two_viewers(robot, q_full_init, q_full_opt,
                            "Step 9: full sequence + mouth delivery")

    # ── 동역학 분석 그래프 ────────────────────────────────────────────────────
    if args.plot or args.plot_save:
        # N_REF: 기준 수직 벡터 (tilt 0 기준)
        n_ref_vec = np.array(N_REF, dtype=float)

        save_path = args.plot_save
        if save_path is None and args.plot:
            save_path = None  # plt.show()
        elif save_path and not os.path.isabs(save_path):
            save_path = str(DEFAULT_OUT_DIR / save_path)

        print(f"\n[PLOT] 동역학 분석 그래프 생성 중 (dt={args.dt}s/node) ...")
        plot_trajectory_analysis(
            robot,
            q_full_init,
            q_full_opt,
            idx,
            dt=args.dt,
            eps_tilt=EPS_TILT,
            n_ref=n_ref_vec,
            save_path=save_path,
        )


if __name__ == "__main__":
    main()
