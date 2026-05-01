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
TABLES = B_DIR / "outputs" / "tables"
OUT = ROOT / "figures" / "paper" / "quality"

BLUE = "#2F5D8C"
TEAL = "#2A8C82"
ORANGE = "#C96E2C"
RED = "#B44646"
GOLD = "#C99A2E"
GRAY = "#6B7280"
DARK = "#1F2933"
LIGHT = "#F5F7FA"


def setup() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    font_path = "/home/lux_liang/.local/texlive/2026/texmf-dist/fonts/opentype/public/fandol/FandolHei-Regular.otf"
    if Path(font_path).exists():
        mpl.font_manager.fontManager.addfont(font_path)
        font = "FandolHei"
    else:
        font = "DejaVu Sans"
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [font, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 9.2,
            "axes.labelsize": 9.2,
            "axes.titlesize": 10.2,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "axes.linewidth": 0.72,
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout(pad=0.8)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


def style(ax: plt.Axes, grid: bool = True) -> None:
    if grid:
        ax.grid(alpha=0.18, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / name, encoding="utf-8-sig")


def plot_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 3.25))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    steps = [
        ("数据输入", "方式1: 4 Hz\n方式2: 5 Hz\n附件1--4与目标点"),
        ("时空配准", r"搜索 $\Delta$\n鲁棒中位数偏差\nBootstrap 判据"),
        ("状态估计", "Kalman Filter\nRTS 平滑\n10 Hz 融合轨迹"),
        ("偏差结构识别", "M0--M4 模型\n时间块交叉验证\n决定校正策略"),
        ("任务事件优化", "射击/拍照候选\n窗口冲突与裕度\n联合方案输出"),
    ]
    xs = np.linspace(0.04, 0.80, len(steps))
    w, h = 0.15, 0.56
    y = 0.24
    colors = [BLUE, BLUE, TEAL, GOLD, ORANGE]
    for i, ((title, body), x, color) in enumerate(zip(steps, xs, colors)):
        rect = mpl.patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=1.25,
            edgecolor=color,
            facecolor="white",
        )
        ax.add_patch(rect)
        ax.text(x + 0.015, y + h - 0.09, title, color=DARK, weight="bold", fontsize=10.4)
        ax.text(x + 0.015, y + h - 0.19, body, color="#374151", va="top", linespacing=1.42)
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x + w + 0.025, y + h / 2),
                xytext=(x + w + 0.005, y + h / 2),
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.1),
            )
    ax.text(0.5, 0.91, "多源定位融合与任务优化证据链", ha="center", va="center", fontsize=12.5, weight="bold", color=DARK)
    ax.text(
        0.5,
        0.08,
        "核心原则：先处理定位可信度，再在固定融合轨迹上生成可执行任务；所有结果均由表格与图像闭环复核。",
        ha="center",
        color="#4B5563",
    )
    save(fig, "fig_pipeline_quality")


def plot_problem23_evidence() -> None:
    kalman = read_csv("kalman_bias_attachment2_summary.csv")
    sens = read_csv("kalman_bias_sensitivity.csv")
    bias = read_csv("attachment3_bias_structure_comparison.csv")
    ci = read_csv("bootstrap_ci_attachment3.csv").iloc[0]

    fig, axes = plt.subplots(2, 2, figsize=(10.7, 7.0))

    ax = axes[0, 0]
    labels = kalman["method"].tolist()
    x = np.arange(len(labels))
    ax.bar(x - 0.18, kalman["rmse_before"], width=0.36, color="#CBD5E1", label="对齐前")
    ax.bar(x + 0.18, kalman["rmse_after"], width=0.36, color=TEAL, label="校正/平滑后")
    ax.set_xticks(x, labels, rotation=12)
    ax.set_ylabel("RMSE / m")
    ax.set_title("(a) 附件2误差降低")
    ax.legend(frameon=False)
    style(ax)

    ax = axes[0, 1]
    x = np.arange(len(sens))
    ax.plot(x, sens["rmse_after"], marker="o", color=BLUE, label="RMSE")
    ax2 = ax.twinx()
    ax2.plot(x, sens["nll"], marker="s", color=ORANGE, label="NLL")
    ax.set_xticks(x, sens["config_name"], rotation=12)
    ax.set_ylabel("RMSE / m", color=BLUE)
    ax2.set_ylabel("负对数似然", color=ORANGE)
    ax.set_title("(b) Kalman 参数灵敏度")
    style(ax)
    ax2.spines["top"].set_visible(False)

    ax = axes[1, 0]
    y = [0, 1]
    bx = [ci["bx_ci_low"], ci["bx_ci_high"]]
    by = [ci["by_ci_low"], ci["by_ci_high"]]
    ax.hlines(y[0], bx[0], bx[1], color=BLUE, lw=4)
    ax.plot(ci["bx_median"], y[0], "o", color=BLUE)
    ax.hlines(y[1], by[0], by[1], color=ORANGE, lw=4)
    ax.plot(ci["by_median"], y[1], "o", color=ORANGE)
    ax.axvline(0, color=GRAY, lw=1, linestyle="--")
    ax.set_yticks(y, [r"$b_x$", r"$b_y$"])
    ax.set_xlabel("Bootstrap 95% 置信区间 / m")
    ax.set_title("(c) 附件3固定偏差置信区间")
    style(ax)

    ax = axes[1, 1]
    x = np.arange(len(bias))
    ax.bar(x - 0.18, bias["train_rmse"], width=0.36, color="#A7C7E7", label="训练")
    ax.bar(x + 0.18, bias["cv_rmse"], width=0.36, color=RED, alpha=0.85, label="时间块CV")
    ax.set_xticks(x, bias["model_name"])
    ax.set_ylabel("RMSE / m")
    ax.set_title("(d) 附件3偏差结构模型比较")
    ax.legend(frameon=False)
    style(ax)

    save(fig, "fig_problem23_evidence")


def plot_task_dashboard() -> None:
    plan = read_csv("task_plan_comparison.csv")
    fov = read_csv("photo_fov_sensitivity.csv")
    shoot = read_csv("shooting_target_summary.csv")
    robust = read_csv("joint_task_robustness.csv")

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2))

    ax = axes[0, 0]
    x = np.arange(len(plan))
    ax.bar(x - 0.24, plan["shooting_events"], width=0.24, color=BLUE, label="射击事件")
    ax.bar(x, plan["photo_events"], width=0.24, color=ORANGE, label="拍照事件")
    ax.bar(x + 0.24, plan["total_events"], width=0.24, color=GRAY, label="总事件")
    ax.set_xticks(x, ["纯射击", "纯拍照", "联合"])
    ax.set_ylabel("事件数")
    ax.set_title("(a) 三类任务方案规模")
    ax.legend(frameon=False, ncol=3)
    style(ax)

    ax = axes[0, 1]
    width = 0.26
    ax.bar(x - width / 2, plan["expected_shooting_hits"], width=width, color=TEAL, label="期望命中")
    ax.bar(x + width / 2, plan["photo_targets_covered"], width=width, color=GOLD, label="拍照覆盖目标")
    ax.set_xticks(x, ["纯射击", "纯拍照", "联合"])
    ax.set_ylabel("目标数")
    ax.set_title("(b) 方案收益对比")
    ax.legend(frameon=False)
    style(ax)

    ax = axes[1, 0]
    ax.plot(fov["fov_degree"], fov["covered_targets"], marker="o", color=BLUE, label="覆盖目标")
    ax.plot(fov["fov_degree"], fov["total_target_observations"], marker="s", color=TEAL, label="总观测")
    ax.plot(fov["fov_degree"], fov["multi_target_events"], marker="^", color=ORANGE, label="多目标同拍")
    ax.set_xlabel("FOV / 度")
    ax.set_ylabel("数量")
    ax.set_title("(c) 拍照视场角灵敏度")
    ax.legend(frameon=False)
    style(ax)

    ax = axes[1, 1]
    robust_sorted = robust.sort_values("worst_margin").head(10)
    colors = [RED if r == "高" else ORANGE if r == "中" else TEAL for r in robust_sorted["risk_level"]]
    ax.barh(robust_sorted["event_id"], robust_sorted["worst_margin"], color=colors)
    ax.axvline(0.02, color=RED, linestyle="--", lw=1.0, label="高风险参考线")
    ax.set_xlabel("worst margin")
    ax.set_title("(d) 联合方案最小鲁棒裕度")
    ax.invert_yaxis()
    ax.legend(frameon=False)
    style(ax)

    save(fig, "fig_task_dashboard")

    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    shoot = shoot.sort_values("target_id")
    colors = [TEAL if n > 0 else "#D1D5DB" for n in shoot["selected_shots"]]
    ax.bar(shoot["target_id"], shoot["hit_probability"], color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("至少命中一次概率")
    ax.set_xlabel("射击目标")
    ax.set_title("射击目标命中概率与可达性")
    ax.tick_params(axis="x", rotation=45)
    style(ax)
    save(fig, "fig_shooting_target_probability_quality")


def plot_broken_gantt() -> None:
    events = read_csv("joint_selected_events.csv")
    robust = read_csv("joint_task_robustness.csv")[["event_id", "risk_level", "worst_margin", "scenario_pass_rate"]]
    events = events.merge(robust, on="event_id", how="left")

    windows = [(478.5, 537.5), (747.0, 770.0)]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.4), sharey=True, gridspec_kw={"width_ratios": [2.5, 1.0]})
    y_positions = np.arange(len(events), 0, -1)
    color_map = {"shooting": BLUE, "photo": ORANGE}

    for ax, (xmin, xmax) in zip(axes, windows):
        ax.set_xlim(xmin, xmax)
        for y, (_, row) in zip(y_positions, events.iterrows()):
            start = row["start_time"]
            end = row["execute_time"]
            if end < xmin or start > xmax:
                continue
            left = max(start, xmin)
            width = min(end, xmax) - left
            color = color_map.get(row["event_type"], GRAY)
            edge = RED if row.get("risk_level") == "高" else "#263238"
            lw = 1.4 if row.get("risk_level") == "高" else 0.45
            ax.barh(y, width, left=left, height=0.64, color=color, alpha=0.86, edgecolor=edge, linewidth=lw)
            ax.plot(row["execute_time"], y, marker="|", color="white", markersize=8, markeredgewidth=1.8)
        ax.set_xlabel("时间 / s")
        style(ax)

    labels = []
    for _, row in events.iterrows():
        target = row["target_id"] if row["event_type"] == "shooting" else row["covered_targets"]
        labels.append(f"{int(row['seq']):02d} {target}")
    axes[0].set_yticks(y_positions, labels)
    axes[0].set_title("(a) 480--537 s 准备窗口")
    axes[1].set_title("(b) 748--769 s 准备窗口")
    axes[1].tick_params(axis="y", left=False, labelleft=False)

    axes[0].spines["right"].set_visible(False)
    axes[1].spines["left"].set_visible(False)
    d = 0.012
    kwargs = dict(transform=axes[0].transAxes, color=DARK, clip_on=False, lw=0.9)
    axes[0].plot((1 - d, 1 + d), (-d, +d), **kwargs)
    axes[0].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs = dict(transform=axes[1].transAxes, color=DARK, clip_on=False, lw=0.9)
    axes[1].plot((-d, +d), (-d, +d), **kwargs)
    axes[1].plot((-d, +d), (1 - d, 1 + d), **kwargs)

    handles = [
        mpl.patches.Patch(color=BLUE, label="射击准备窗口"),
        mpl.patches.Patch(color=ORANGE, label="拍照准备窗口"),
        mpl.patches.Patch(facecolor="white", edgecolor=RED, linewidth=1.4, label="高风险事件"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("联合方案断轴 Gantt 图：放大两个任务密集准备窗口", y=0.99, fontsize=12.0, weight="bold")
    save(fig, "fig_broken_gantt_quality")


def main() -> None:
    setup()
    plot_pipeline()
    plot_problem23_evidence()
    plot_task_dashboard()
    plot_broken_gantt()


if __name__ == "__main__":
    main()
