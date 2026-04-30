from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


B_DIR = Path("/home/lux_liang/work/projects/github/数模校赛B题")
OUT = ROOT / "figures" / "paper"

TIME_COL = "时间(s)"
X_COL = "X坐标(m)"
Y_COL = "Y坐标(m)"

BLUE = "#356EA9"
TEAL = "#2A9D8F"
ORANGE = "#D97732"
GRAY = "#6B7280"
DARK = "#253238"
GOLD = "#D8A31A"
RED = "#B94A48"


def setup() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.5,
            "legend.fontsize": 8.2,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.75,
            "lines.linewidth": 1.35,
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(path: Path) -> None:
    plt.tight_layout(pad=0.7)
    plt.savefig(path)
    plt.close()


def style_axis(ax: plt.Axes, equal: bool = False) -> None:
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if equal:
        ax.axis("equal")


def read_raw(idx: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    xls = B_DIR / f"附件{idx}.xlsx"
    return (
        pd.read_excel(xls, sheet_name="方式1(4Hz)"),
        pd.read_excel(xls, sheet_name="方式2(5Hz)"),
    )


def selected_r5() -> pd.DataFrame:
    return pd.read_csv(B_DIR / "outputs" / "tables" / "selected_tasks_R5_multi_uncertainty.csv")


def plot_attachment1() -> None:
    raw1, raw2 = read_raw(1)
    fused = pd.read_csv(B_DIR / "outputs" / "trajectories" / "fused_attachment1_10hz.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    axes[0].plot(raw1[X_COL], raw1[Y_COL], color=BLUE, label="source 1", alpha=0.95)
    axes[0].plot(raw2[X_COL], raw2[Y_COL], color=ORANGE, label="source 2", alpha=0.78)
    axes[0].set_title("(a) before alignment")
    axes[1].plot(fused["x1_aligned"], fused["y1_aligned"], color=BLUE, label="source 1")
    axes[1].plot(
        fused["x2_aligned_corrected"],
        fused["y2_aligned_corrected"],
        color=ORANGE,
        linestyle=(0, (4, 2)),
        label="source 2 aligned",
    )
    axes[1].set_title("(b) after alignment")
    for ax in axes:
        ax.set_xlabel("X / m")
        ax.set_ylabel("Y / m")
        style_axis(ax, equal=True)
        ax.legend(frameon=False)
    save(OUT / "fig1_attachment1_alignment.pdf")


def plot_attachment2() -> None:
    raw1, raw2 = read_raw(2)
    fused = pd.read_csv(B_DIR / "outputs" / "trajectories" / "fused_attachment2_10hz.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    axes[0].plot(raw1[X_COL], raw1[Y_COL], color=BLUE, label="source 1", alpha=0.92)
    axes[0].plot(raw2[X_COL], raw2[Y_COL], color=ORANGE, label="source 2", alpha=0.68)
    axes[0].set_title("(a) raw trajectories")
    axes[1].plot(fused["x1_aligned"], fused["y1_aligned"], color=BLUE, label="source 1")
    axes[1].plot(
        fused["x2_aligned_corrected"],
        fused["y2_aligned_corrected"],
        color=TEAL,
        linestyle=(0, (4, 2)),
        label="source 2 corrected",
    )
    axes[1].plot(fused[X_COL], fused[Y_COL], color=DARK, linewidth=1.0, alpha=0.78, label="fused")
    axes[1].set_title("(b) after temporal-spatial registration")
    for ax in axes:
        ax.set_xlabel("X / m")
        ax.set_ylabel("Y / m")
        style_axis(ax, equal=True)
        ax.legend(frameon=False)
    save(OUT / "fig2_attachment2_correction.pdf")


def plot_residuals2() -> None:
    res = pd.read_csv(B_DIR / "outputs" / "tables" / "attachment2_residuals.csv")
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    hb = ax.hexbin(
        res["residual_x_after"],
        res["residual_y_after"],
        gridsize=42,
        cmap="Blues",
        mincnt=1,
        linewidths=0,
    )
    ax.scatter(
        res["residual_x_after"].sample(min(800, len(res)), random_state=7),
        res["residual_y_after"].sample(min(800, len(res)), random_state=7),
        s=4,
        color=DARK,
        alpha=0.16,
        edgecolors="none",
    )
    ax.axhline(0, color=GRAY, linewidth=0.8)
    ax.axvline(0, color=GRAY, linewidth=0.8)
    ax.set_xlabel("residual X after correction / m")
    ax.set_ylabel("residual Y after correction / m")
    ax.set_title("Residual distribution after relative-bias correction")
    cbar = fig.colorbar(hb, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("count")
    style_axis(ax)
    save(OUT / "fig3_attachment2_residuals.pdf")


def plot_attachment3_traj() -> None:
    traj = pd.read_csv(B_DIR / "outputs" / "trajectories" / "fused_attachment3_10hz.csv")
    fig, ax = plt.subplots(figsize=(6.2, 4.7))
    points = ax.scatter(
        traj[X_COL],
        traj[Y_COL],
        c=traj[TIME_COL],
        s=7,
        cmap="viridis",
        linewidths=0,
        alpha=0.88,
    )
    ax.plot(traj[X_COL], traj[Y_COL], color=DARK, linewidth=0.5, alpha=0.32)
    ax.scatter(traj[X_COL].iloc[0], traj[Y_COL].iloc[0], s=45, color=BLUE, label="start", zorder=5)
    ax.scatter(traj[X_COL].iloc[-1], traj[Y_COL].iloc[-1], s=45, color=ORANGE, label="end", zorder=5)
    ax.set_xlabel("X / m")
    ax.set_ylabel("Y / m")
    ax.set_title("Fused 10 Hz state trajectory of Attachment 3")
    cbar = fig.colorbar(points, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("time / s")
    style_axis(ax, equal=True)
    ax.legend(frameon=False, loc="best")
    save(OUT / "fig4_attachment3_fused_traj.pdf")


def plot_kinematics() -> None:
    traj = pd.read_csv(B_DIR / "outputs" / "trajectories" / "fused_attachment3_10hz.csv")
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.2), sharex=True)
    axes[0].plot(traj[TIME_COL], traj["speed"], color=BLUE)
    axes[0].fill_between(traj[TIME_COL], 0, traj["speed"], color=BLUE, alpha=0.08)
    axes[0].axhline(2.0, color=ORANGE, linestyle="--", linewidth=1.0, label="shooting limit")
    axes[0].axhline(1.5, color=TEAL, linestyle=":", linewidth=1.2, label="photo limit")
    axes[0].set_ylabel("speed / (m s$^{-1}$)")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].plot(traj[TIME_COL], traj["acceleration"], color=GRAY)
    axes[1].fill_between(traj[TIME_COL], 0, traj["acceleration"], color=GRAY, alpha=0.10)
    axes[1].axhline(1.5, color=ORANGE, linestyle="--", linewidth=1.0, label="acceleration limit")
    axes[1].set_xlabel("time / s")
    axes[1].set_ylabel("acceleration / (m s$^{-2}$)")
    axes[1].legend(frameon=False)
    for ax in axes:
        style_axis(ax)
    save(OUT / "fig5_attachment3_kinematics.pdf")


def read_targets() -> pd.DataFrame:
    frames = []
    for sheet, task in [("射击目标", "shooting"), ("拍照目标", "photo")]:
        df = pd.read_excel(B_DIR / "附件4.xlsx", sheet_name=sheet)
        df["task"] = task
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def plot_tasks_distribution() -> None:
    traj = pd.read_csv(B_DIR / "outputs" / "trajectories" / "fused_attachment3_10hz.csv")
    targets = read_targets()
    selected = selected_r5()
    fig, ax = plt.subplots(figsize=(6.8, 5.1))
    ax.plot(traj[X_COL], traj[Y_COL], color=GRAY, linewidth=0.95, alpha=0.72, label="fused trajectory")
    shoot = targets[targets["task"] == "shooting"]
    photo = targets[targets["task"] == "photo"]
    ax.scatter(shoot[X_COL], shoot[Y_COL], s=30, color=ORANGE, alpha=0.72, label="shooting targets")
    ax.scatter(photo[X_COL], photo[Y_COL], s=30, color=BLUE, alpha=0.72, label="photo targets")
    chosen = targets.merge(selected[["目标编号", "任务"]], left_on="编号", right_on="目标编号", how="inner")
    ax.scatter(
        chosen[X_COL],
        chosen[Y_COL],
        s=130,
        marker="*",
        color=GOLD,
        edgecolor=DARK,
        linewidth=0.65,
        label="R5 selected targets",
        zorder=5,
    )
    for _, row in chosen.iterrows():
        ax.text(row[X_COL] + 0.6, row[Y_COL] + 0.6, str(row["编号"]), fontsize=7.4, color=DARK)
    ax.set_xlabel("X / m")
    ax.set_ylabel("Y / m")
    ax.set_title("R5 selected tasks on the fixed trajectory")
    style_axis(ax, equal=True)
    ax.legend(frameon=False, loc="best")
    save(OUT / "fig6_task_distribution.pdf")


def plot_timeline() -> None:
    cand = pd.read_csv(B_DIR / "outputs" / "tables" / "task_candidates.csv")
    selected = selected_r5()
    fig, ax = plt.subplots(figsize=(8.2, 3.5))
    y = np.where(cand["任务"] == "射击", 0, 1)
    ax.scatter(cand["exec_time"], y, s=7, color="#A8ADB4", alpha=0.30, label="feasible candidates")
    y_sel = np.where(selected["任务"] == "射击", 0, 1)
    ax.scatter(selected["任务执行时刻(s)"], y_sel, s=72, marker="*", color=ORANGE, label="R5 selected")
    for _, row in selected.iterrows():
        yy = 0 if row["任务"] == "射击" else 1
        ax.hlines(yy, row["开始准备时刻(s)"], row["任务执行时刻(s)"], color=TEAL, linewidth=2.0)
    ax.set_yticks([0, 1], ["shooting", "photo"])
    ax.set_xlabel("time / s")
    ax.set_ylim(-0.42, 1.42)
    ax.set_title("Feasible windows and R5 selected execution times")
    style_axis(ax)
    ax.legend(frameon=False, loc="upper right")
    save(OUT / "fig7_task_timeline.pdf")


def plot_robust_model_comparison() -> None:
    df = pd.read_csv(B_DIR / "outputs" / "tables" / "robust_task_model_comparison_v3.csv")
    df = df[df["model"].isin(["R3", "R4", "R5", "R6_buffered_eps_0.00"])].copy()
    df["model"] = df["model"].replace({"R6_buffered_eps_0.00": "R6"})
    x = np.arange(len(df))
    fig, ax1 = plt.subplots(figsize=(6.4, 3.8))
    ax1.bar(x - 0.18, df["mean_scenario_margin_sum"], width=0.34, color=BLUE, alpha=0.86, label="mean margin sum")
    ax1.bar(x + 0.18, df["worst_case_margin_sum"], width=0.34, color=TEAL, alpha=0.86, label="worst margin sum")
    ax1.set_ylabel("normalized margin sum")
    ax1.set_xticks(x, df["model"])
    ax2 = ax1.twinx()
    ax2.plot(x, df["scenario_feasible_rate"], color=ORANGE, marker="o", linewidth=1.6, label="scenario feasible rate")
    ax2.set_ylabel("scenario feasible rate")
    ax2.set_ylim(0, 1.05)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="upper left")
    ax1.set_title("Robust task model comparison")
    style_axis(ax1)
    ax2.spines["top"].set_visible(False)
    save(OUT / "fig8_robust_model_comparison.pdf")


def plot_smoothing_audit() -> None:
    df = pd.read_csv(B_DIR / "outputs" / "tables" / "oversmoothing_audit.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    for poly, color in [(2, BLUE), (3, ORANGE)]:
        part = df[df["polyorder"] == poly].sort_values("window_length")
        axes[0].plot(part["window_length"], part["feasible_candidate_count"], marker="o", color=color, label=f"poly={poly}")
        axes[1].plot(part["window_length"], part["fidelity_mean"], marker="o", color=color, label=f"poly={poly}")
    axes[0].set_xlabel("Savitzky-Golay window length")
    axes[0].set_ylabel("feasible candidate count")
    axes[0].set_title("(a) candidate sensitivity")
    axes[1].set_xlabel("Savitzky-Golay window length")
    axes[1].set_ylabel("mean fidelity shift / m")
    axes[1].set_title("(b) state-estimation deviation")
    for ax in axes:
        style_axis(ax)
        ax.legend(frameon=False)
    save(OUT / "fig9_smoothing_audit.pdf")


def main() -> None:
    setup()
    plot_attachment1()
    plot_attachment2()
    plot_residuals2()
    plot_attachment3_traj()
    plot_kinematics()
    plot_tasks_distribution()
    plot_timeline()
    plot_robust_model_comparison()
    plot_smoothing_audit()
    print(f"paper figures written to {OUT}")


if __name__ == "__main__":
    main()
