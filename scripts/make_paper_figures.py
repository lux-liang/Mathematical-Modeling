from __future__ import annotations

from pathlib import Path
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
B_DIR = Path("/home/lux_liang/work/projects/github/数模校赛B题")
OUT = ROOT / "figures" / "paper"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
TIME_COL = "时间(s)"
X_COL = "X坐标(m)"
Y_COL = "Y坐标(m)"


def setup() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.35,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def read_raw(idx: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    xls = B_DIR / f"附件{idx}.xlsx"
    return (
        pd.read_excel(xls, sheet_name="方式1(4Hz)"),
        pd.read_excel(xls, sheet_name="方式2(5Hz)"),
    )


def plot_attachment1() -> None:
    raw1, raw2 = read_raw(1)
    fused = pd.read_csv(B_DIR / "outputs" / "trajectories" / "fused_attachment1_10hz.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    axes[0].plot(raw1[X_COL], raw1[Y_COL], color="#3b6ea8", label="source 1")
    axes[0].plot(raw2[X_COL], raw2[Y_COL], color="#e6842a", alpha=0.85, label="source 2")
    axes[0].set_title("(a) before time alignment")
    axes[1].plot(fused["x1_aligned"], fused["y1_aligned"], color="#3b6ea8", label="source 1")
    axes[1].plot(fused["x2_aligned_corrected"], fused["y2_aligned_corrected"], color="#e6842a", linestyle="--", label="source 2 aligned")
    axes[1].set_title("(b) after time alignment")
    for ax in axes:
        ax.set_xlabel("X / m")
        ax.set_ylabel("Y / m")
        ax.axis("equal")
        ax.grid(alpha=0.22)
        ax.legend(frameon=False)
    save(OUT / "fig1_attachment1_alignment.pdf")


def plot_attachment2() -> None:
    raw1, raw2 = read_raw(2)
    fused = pd.read_csv(B_DIR / "outputs" / "trajectories" / "fused_attachment2_10hz.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    axes[0].plot(raw1[X_COL], raw1[Y_COL], color="#3b6ea8", label="source 1")
    axes[0].plot(raw2[X_COL], raw2[Y_COL], color="#e6842a", alpha=0.75, label="source 2")
    axes[0].set_title("(a) raw trajectories")
    axes[1].plot(fused["x1_aligned"], fused["y1_aligned"], color="#3b6ea8", label="source 1")
    axes[1].plot(fused["x2_aligned_corrected"], fused["y2_aligned_corrected"], color="#2a9d8f", linestyle="--", label="source 2 corrected")
    axes[1].plot(fused[X_COL], fused[Y_COL], color="#555555", linewidth=1.1, label="fused")
    axes[1].set_title("(b) after temporal-spatial registration")
    for ax in axes:
        ax.set_xlabel("X / m")
        ax.set_ylabel("Y / m")
        ax.axis("equal")
        ax.grid(alpha=0.22)
        ax.legend(frameon=False)
    save(OUT / "fig2_attachment2_correction.pdf")


def plot_residuals2() -> None:
    res = pd.read_csv(B_DIR / "outputs" / "tables" / "attachment2_residuals.csv")
    plt.figure(figsize=(5.2, 4.4))
    plt.scatter(res["residual_x_after"], res["residual_y_after"], s=7, color="#3b6ea8", alpha=0.35, edgecolors="none")
    plt.axhline(0, color="#666666", linewidth=0.8)
    plt.axvline(0, color="#666666", linewidth=0.8)
    plt.xlabel("residual X after correction / m")
    plt.ylabel("residual Y after correction / m")
    plt.title("Residual distribution of Attachment 2")
    plt.grid(alpha=0.2)
    save(OUT / "fig3_attachment2_residuals.pdf")


def plot_attachment3_traj() -> None:
    traj = pd.read_csv(B_DIR / "outputs" / "trajectories" / "fused_attachment3_10hz.csv")
    plt.figure(figsize=(6.2, 4.8))
    plt.plot(traj[X_COL], traj[Y_COL], color="#2a9d8f", linewidth=1.4)
    plt.scatter(traj[X_COL].iloc[0], traj[Y_COL].iloc[0], s=38, color="#3b6ea8", label="start")
    plt.scatter(traj[X_COL].iloc[-1], traj[Y_COL].iloc[-1], s=38, color="#d05a28", label="end")
    plt.xlabel("X / m")
    plt.ylabel("Y / m")
    plt.title("Fused 10 Hz trajectory of Attachment 3")
    plt.axis("equal")
    plt.grid(alpha=0.22)
    plt.legend(frameon=False)
    save(OUT / "fig4_attachment3_fused_traj.pdf")


def plot_kinematics() -> None:
    traj = pd.read_csv(B_DIR / "outputs" / "trajectories" / "fused_attachment3_10hz.csv")
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.4), sharex=True)
    axes[0].plot(traj[TIME_COL], traj["speed"], color="#3b6ea8")
    axes[0].axhline(2.0, color="#d05a28", linestyle="--", linewidth=1.0, label="shooting limit")
    axes[0].axhline(1.5, color="#2a9d8f", linestyle=":", linewidth=1.2, label="photo limit")
    axes[0].set_ylabel("speed / (m s$^{-1}$)")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].plot(traj[TIME_COL], traj["acceleration"], color="#6c757d")
    axes[1].axhline(1.5, color="#d05a28", linestyle="--", linewidth=1.0, label="acceleration limit")
    axes[1].set_xlabel("time / s")
    axes[1].set_ylabel("acceleration / (m s$^{-2}$)")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(alpha=0.22)
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
    selected = pd.read_csv(B_DIR / "outputs" / "tables" / "optimized_selected_tasks.csv")
    plt.figure(figsize=(6.8, 5.2))
    plt.plot(traj[X_COL], traj[Y_COL], color="#666666", linewidth=1.0, label="fused trajectory")
    shoot = targets[targets["task"] == "shooting"]
    photo = targets[targets["task"] == "photo"]
    plt.scatter(shoot[X_COL], shoot[Y_COL], s=28, color="#d05a28", alpha=0.75, label="shooting targets")
    plt.scatter(photo[X_COL], photo[Y_COL], s=28, color="#3b6ea8", alpha=0.75, label="photo targets")
    chosen = targets.merge(selected[["目标编号", "任务"]], left_on=["编号"], right_on=["目标编号"], how="inner")
    plt.scatter(chosen[X_COL], chosen[Y_COL], s=120, marker="*", color="#f0b429", edgecolor="#333333", linewidth=0.6, label="selected")
    plt.xlabel("X / m")
    plt.ylabel("Y / m")
    plt.title("Selected tasks on the fixed trajectory")
    plt.axis("equal")
    plt.grid(alpha=0.22)
    plt.legend(frameon=False, loc="best")
    save(OUT / "fig6_task_distribution.pdf")


def plot_timeline() -> None:
    cand = pd.read_csv(B_DIR / "outputs" / "tables" / "task_candidates.csv")
    selected = pd.read_csv(B_DIR / "outputs" / "tables" / "optimized_selected_tasks.csv")
    plt.figure(figsize=(8.2, 3.8))
    y = np.where(cand["任务"] == "射击", 0, 1)
    plt.scatter(cand["exec_time"], y, s=8, color="#9aa0a6", alpha=0.35, label="feasible candidates")
    y_sel = np.where(selected["任务"] == "射击", 0, 1)
    plt.scatter(selected["任务执行时刻(s)"], y_sel, s=60, marker="*", color="#d05a28", label="selected")
    for _, row in selected.iterrows():
        yy = 0 if row["任务"] == "射击" else 1
        plt.hlines(yy, row["开始准备时刻(s)"], row["任务执行时刻(s)"], color="#2a9d8f", linewidth=2.0)
    plt.yticks([0, 1], ["shooting", "photo"])
    plt.xlabel("time / s")
    plt.ylim(-0.45, 1.45)
    plt.title("Feasible task windows on the fixed trajectory")
    plt.grid(axis="x", alpha=0.22)
    plt.legend(frameon=False, loc="upper right")
    save(OUT / "fig7_task_timeline.pdf")


def main() -> None:
    setup()
    plot_attachment1()
    plot_attachment2()
    plot_residuals2()
    plot_attachment3_traj()
    plot_kinematics()
    plot_tasks_distribution()
    plot_timeline()
    print(f"paper figures written to {OUT}")


if __name__ == "__main__":
    main()
