# -*- coding: utf-8 -*-
"""Trajectory analysis plots from pre-computed step9 JSON.

Loads q_full_opt from tutorial_drag_opt_output/step9_mouth_result.json
and produces 5 figures (4 individual + 1 combined) without re-running
the SLSQP optimization.

Usage
-----
  python meal_assist/step9_plot_analysis.py
  python meal_assist/step9_plot_analysis.py --dt 0.02
  python meal_assist/step9_plot_analysis.py --json path/to/step9_mouth_result.json
  python meal_assist/step9_plot_analysis.py --out-dir my_plots/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless-safe; switch to TkAgg/Qt5Agg if interactive
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from meal_assist.config import SystemConfig
from meal_assist.robot import RobotModel, mujoco

# Reference upward direction for tilt computation (world Z-up)
N_REF = np.array([0.0, 0.0, 1.0], dtype=float)

# ── Visual constants (all-English) ────────────────────────────────────────────
DT_DEFAULT = 0.05   # seconds per node

JOINT_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4"]
JOINT_NAMES  = [f"J{i+1}" for i in range(6)]

SEG_COLORS = {
    "seg0": "#e8e8e8",
    "seg1": "#d0e8ff",
    "seg2": "#fff0b0",
    "seg3": "#c8f0c8",
    "seg4": "#ffe0b0",
    "seg5": "#ffd0d0",
}
SEG_LABELS = {
    "seg0": "pre->engage",
    "seg1": "engage->drag_s",
    "seg2": "drag",
    "seg3": "lift",
    "seg4": "carry",
    "seg5": "deliver",
}
JUNC_KEYS = ["pre", "engage", "dstart", "dend", "lift", "mpre", "deliver"]
JUNC_DISPLAY = {
    "pre":     "pre",
    "engage":  "engage",
    "dstart":  "drag_s",
    "dend":    "drag_e",
    "lift":    "lift",
    "mpre":    "m_pre",
    "deliver": "deliver",
}


# ── Index helper (mirrors compute_indices in step9) ───────────────────────────

def compute_indices(n_seg, n_drag, n_lift, n_carry, n_deliver):
    i_pre     = 0
    i_engage  = n_seg - 1
    i_dstart  = 2 * n_seg - 2
    i_dend    = 2 * n_seg + n_drag - 3
    i_lift    = 2 * n_seg + n_drag + n_lift - 4
    i_mpre    = 2 * n_seg + n_drag + n_lift + n_carry - 5
    i_deliver = 2 * n_seg + n_drag + n_lift + n_carry + n_deliver - 6
    return {
        "pre":     i_pre,
        "engage":  i_engage,
        "dstart":  i_dstart,
        "dend":    i_dend,
        "lift":    i_lift,
        "mpre":    i_mpre,
        "deliver": i_deliver,
        "total":   i_deliver + 1,
        "seg0": (i_pre,    i_engage  + 1),
        "seg1": (i_engage, i_dstart  + 1),
        "seg2": (i_dstart, i_dend    + 1),
        "seg3": (i_dend,   i_lift    + 1),
        "seg4": (i_lift,   i_mpre    + 1),
        "seg5": (i_mpre,   i_deliver + 1),
    }


# ── Kinematics (finite difference) ────────────────────────────────────────────

def compute_kinematics(q: np.ndarray, dt: float):
    """Angular velocity and acceleration via central finite difference."""
    vel = np.zeros_like(q)
    acc = np.zeros_like(q)
    vel[1:-1] = (q[2:] - q[:-2]) / (2.0 * dt)
    vel[0]    = (q[1]  - q[0])   / dt
    vel[-1]   = (q[-1] - q[-2])  / dt
    acc[1:-1] = (q[2:] - 2.0 * q[1:-1] + q[:-2]) / dt ** 2
    return vel, acc


# ── Torque via MuJoCo inverse dynamics ────────────────────────────────────────

def compute_torques(robot, q: np.ndarray, vel: np.ndarray, acc: np.ndarray):
    """Joint torques [Nm] using mj_inverse at each node."""
    N   = len(q)
    nq  = robot.model.nq
    torque = np.zeros((N, 6), dtype=float)
    d = mujoco.MjData(robot.model)
    for k in range(N):
        d.qpos[:nq] = q[k][:nq]
        d.qvel[:nq] = vel[k][:nq]
        d.qacc[:nq] = acc[k][:nq]
        mujoco.mj_inverse(robot.model, d)
        torque[k] = d.qfrc_inverse[:6]
    return torque


# ── Tilt trajectory ───────────────────────────────────────────────────────────

def compute_tilt_deg(robot, q: np.ndarray, n_ref: np.ndarray):
    """Spoon tilt [deg] at each waypoint."""
    N       = len(q)
    tilt    = np.zeros(N)
    n_local = np.array(robot.cfg.spoon_normal_local, dtype=float)
    d       = mujoco.MjData(robot.model)
    for k in range(N):
        robot.set_q(d, q[k])
        mujoco.mj_forward(robot.model, d)
        n_cur   = robot.current_body_axis_world(d, n_local)
        tilt[k] = float(np.degrees(np.linalg.norm(np.cross(n_cur, n_ref))))
    return tilt


# ── Shared axis decoration ────────────────────────────────────────────────────

def _apply_time_xaxis(ax, t, junc_t, is_bottom=True):
    """Set up x-axis:
    - Bottom axis: regular time ticks [s]
    - Top secondary axis: junction labels (pre/engage/drag_s/...)

    Call this on every subplot. Pass is_bottom=True for the last (or only) row
    so the 'Time [s]' label is drawn; False for upper rows in a shared-x figure.
    """
    t_end = t[-1]
    ax.set_xlim(t[0], t_end)

    # ── primary (bottom) x-axis: uniform time ticks ─────────────────────────
    # pick a round step so we get ~6-10 ticks
    raw_step = t_end / 8.0
    for candidate in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0]:
        if raw_step <= candidate:
            step = candidate
            break
    else:
        step = round(raw_step, 0)

    primary_ticks = np.arange(0, t_end + step * 0.01, step)
    ax.set_xticks(primary_ticks)
    ax.set_xticklabels([f"{v:.1f}" for v in primary_ticks], fontsize=8)
    if is_bottom:
        ax.set_xlabel("Time [s]", fontsize=10)

    # ── secondary (top) x-axis: junction labels ──────────────────────────────
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())

    junc_t_list = [junc_t[k] for k in JUNC_KEYS]
    junc_l_list = [JUNC_DISPLAY[k] for k in JUNC_KEYS]
    ax2.set_xticks(junc_t_list)
    ax2.set_xticklabels(junc_l_list, fontsize=7.5, rotation=25, ha="left", color="#444")
    ax2.tick_params(axis="x", length=4, width=0.8, color="#888")

    # hide top axis spine to keep it clean
    ax2.spines["top"].set_linewidth(0.5)
    ax2.spines["top"].set_color("#aaa")

    return ax2


def _base_decorate(ax, seg_spans, junc_t, ylabel):
    """Segment shading + junction lines + ylabel."""
    for ts, te, color, _ in seg_spans:
        ax.axvspan(ts, te, color=color, alpha=0.40, linewidth=0)
    for name in JUNC_KEYS:
        ax.axvline(junc_t[name], color="#555", lw=0.8, ls=":")
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, axis="y", lw=0.4, alpha=0.45)
    ax.tick_params(axis="y", labelsize=8)


def _seg_legend(ax):
    """Segment color legend patches."""
    patches = [mpatches.Patch(color=SEG_COLORS[k], alpha=0.7, label=SEG_LABELS[k])
               for k in SEG_COLORS]
    ax.legend(handles=patches, fontsize=7, loc="upper right",
              ncol=3, framealpha=0.85, title="Segment")


def _joint_legend(ax, extra_handles=None):
    handles = [Line2D([0], [0], color=JOINT_COLORS[j], lw=1.4, label=JOINT_NAMES[j])
               for j in range(6)]
    if extra_handles:
        handles += extra_handles
    ax.legend(handles=handles, fontsize=8, loc="upper right",
              ncol=3, framealpha=0.85)


# ── Individual figure savers ──────────────────────────────────────────────────

def _plot_joint_metric(t, data, seg_spans, junc_t, idx,
                       title, ylabel, fname, out_dir):
    """Plot 6-joint metric — bottom axis: time [s], top axis: junction labels."""
    fig, ax = plt.subplots(figsize=(13, 4.5))
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)

    _base_decorate(ax, seg_spans, junc_t, ylabel)
    for j in range(6):
        ax.plot(t, data[:, j], color=JOINT_COLORS[j], lw=1.3, label=JOINT_NAMES[j])
    ax.legend(fontsize=8, loc="upper right", ncol=3, framealpha=0.85)
    _apply_time_xaxis(ax, t, junc_t, is_bottom=True)

    plt.tight_layout()
    path = out_dir / fname
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved -> {path}")
    plt.close(fig)


def _plot_tilt(t, tilt, eps_tilt_deg, seg_spans, junc_t, fname, out_dir):
    """Tilt error — bottom axis: time [s], top axis: junction labels."""
    fig, ax = plt.subplots(figsize=(13, 4.5))
    fig.suptitle("Spoon Tilt Error [deg]", fontsize=12, fontweight="bold", y=0.98)

    _base_decorate(ax, seg_spans, junc_t, "Tilt [deg]")
    ax.plot(t, tilt, color="#d62728", lw=1.4, label="tilt")
    ax.axhline(eps_tilt_deg, color="navy", lw=1.0, ls="-.",
               label=f"EPS_TILT = {eps_tilt_deg:.1f} deg")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.85)
    _apply_time_xaxis(ax, t, junc_t, is_bottom=True)

    plt.tight_layout()
    path = out_dir / fname
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved -> {path}")
    plt.close(fig)


def _plot_combined(t, vel, acc, torque, tilt, eps_tilt_deg,
                   seg_spans, junc_t, fname, out_dir):
    """4-row combined figure.
    Each row: bottom = time ticks, top = junction labels (only shown on row 1).
    """
    fig, axes = plt.subplots(4, 1, figsize=(15, 14), sharex=False)
    fig.suptitle("Step 9: Full Trajectory Dynamics Analysis",
                 fontsize=13, fontweight="bold")

    row_cfg = [
        (axes[0], "Angular Velocity [rad/s]",      vel,    False),
        (axes[1], "Angular Acceleration [rad/s^2]", acc,    False),
        (axes[2], "Joint Torque [Nm]",              torque, False),
        (axes[3], "Tilt [deg]",                     None,   True),
    ]

    for ax, ylabel, data, is_bot in row_cfg:
        _base_decorate(ax, seg_spans, junc_t, ylabel)
        if data is not None:
            for j in range(6):
                ax.plot(t, data[:, j], color=JOINT_COLORS[j], lw=1.2, label=JOINT_NAMES[j])
        _apply_time_xaxis(ax, t, junc_t, is_bottom=is_bot)

    # tilt row extras
    axes[3].plot(t, tilt, color="#d62728", lw=1.4, label="tilt")
    axes[3].axhline(eps_tilt_deg, color="navy", lw=1.0, ls="-.",
                    label=f"EPS_TILT={eps_tilt_deg:.1f}deg")
    axes[3].set_ylim(bottom=0)
    axes[3].legend(fontsize=8, loc="upper right")

    # joint legend on row 1
    handles = [Line2D([0], [0], color=JOINT_COLORS[j], lw=1.3, label=JOINT_NAMES[j])
               for j in range(6)]
    axes[0].legend(handles=handles, fontsize=8, loc="upper right",
                   ncol=3, framealpha=0.85)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = out_dir / fname
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved -> {path}")
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    # default JSON path
    here = Path(__file__).resolve().parent
    default_json = here.parent / "tutorial_drag_opt_output" / "step9_mouth_result.json"

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json",    type=Path, default=default_json,
                        help="Path to step9_mouth_result.json")
    parser.add_argument("--dt",      type=float, default=DT_DEFAULT,
                        help=f"Time step per node [s] (default {DT_DEFAULT})")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory for figures (default: same as JSON)")
    args = parser.parse_args()

    # ── Load JSON ─────────────────────────────────────────────────────────────
    json_path = args.json
    if not json_path.exists():
        print(f"[ERROR] JSON not found: {json_path}")
        sys.exit(1)

    print(f"[INFO] Loading {json_path}")
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    q_full_opt   = np.array(data["q_full_opt"], dtype=float)
    eps_tilt_deg = float(data["eps_tilt_deg"])
    ns = data["n_seg"];   nd = data["n_drag"]
    nl = data["n_lift"];  nc = data["n_carry"];  nv = data["n_deliver"]
    idx = compute_indices(ns, nd, nl, nc, nv)
    N   = len(q_full_opt)
    t   = np.arange(N) * args.dt

    print(f"  Trajectory: {N} nodes, dt={args.dt}s, "
          f"total={t[-1]:.2f}s, EPS_TILT={eps_tilt_deg:.1f}deg")

    # ── Robot model (needed for torque + tilt) ─────────────────────────────
    print("[INFO] Loading robot model ...")
    cfg   = SystemConfig()
    robot = RobotModel(cfg)
    n_ref = np.array(N_REF, dtype=float)

    # ── Compute kinematics ─────────────────────────────────────────────────
    print("[INFO] Computing velocity & acceleration ...")
    vel, acc = compute_kinematics(q_full_opt, args.dt)

    # ── Compute torques ────────────────────────────────────────────────────
    print("[INFO] Computing torques via inverse dynamics ...")
    torque = compute_torques(robot, q_full_opt, vel, acc)

    # ── Compute tilt ───────────────────────────────────────────────────────
    print("[INFO] Computing spoon tilt ...")
    tilt = compute_tilt_deg(robot, q_full_opt, n_ref)

    # ── Build shared geometry for plots ────────────────────────────────────
    junc_t = {k: idx[k] * args.dt for k in JUNC_KEYS}

    seg_spans = []
    for key in ["seg0", "seg1", "seg2", "seg3", "seg4", "seg5"]:
        s, e = idx[key]
        seg_spans.append((s * args.dt, (e - 1) * args.dt,
                          SEG_COLORS[key], SEG_LABELS[key]))

    # ── Output directory ───────────────────────────────────────────────────
    out_dir = args.out_dir if args.out_dir else json_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Saving figures to {out_dir}")

    # ── 1) Angular velocity ────────────────────────────────────────────────
    _plot_joint_metric(
        t, vel, seg_spans, junc_t, idx,
        title="Angular Velocity  [rad/s]",
        ylabel="Ang. Velocity [rad/s]",
        fname="step9_angular_velocity.png",
        out_dir=out_dir,
    )

    # ── 2) Angular acceleration ────────────────────────────────────────────
    _plot_joint_metric(
        t, acc, seg_spans, junc_t, idx,
        title="Angular Acceleration  [rad/s^2]",
        ylabel="Ang. Acceleration [rad/s^2]",
        fname="step9_angular_acceleration.png",
        out_dir=out_dir,
    )

    # ── 3) Joint torque ────────────────────────────────────────────────────
    _plot_joint_metric(
        t, torque, seg_spans, junc_t, idx,
        title="Joint Torque  [Nm]  (inverse dynamics)",
        ylabel="Torque [Nm]",
        fname="step9_torque.png",
        out_dir=out_dir,
    )

    # ── 4) Spoon tilt ──────────────────────────────────────────────────────
    _plot_tilt(
        t, tilt, eps_tilt_deg, seg_spans, junc_t,
        fname="step9_tilt.png",
        out_dir=out_dir,
    )

    # ── 5) Combined 4-row ──────────────────────────────────────────────────
    _plot_combined(
        t, vel, acc, torque, tilt, eps_tilt_deg,
        seg_spans, junc_t,
        fname="step9_combined.png",
        out_dir=out_dir,
    )

    print(f"\n[DONE] 5 figures saved to {out_dir}")


if __name__ == "__main__":
    main()
