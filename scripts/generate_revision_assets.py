from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/home/lux_liang/work/projects/github/数模校赛B题")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
sys.path.insert(0, str(DATA_ROOT / "src"))

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from alignment import align_pair
from bias_structure import _blocked_folds, _fit_predict, _residual_frame
from data_io import TIME_COL, X_COL, Y_COL, read_position_workbook, read_targets
from fusion import fuse_attachment
from kinematics import add_kinematics
from task_events import (
    _select_joint_plan_milp,
    circular_angle_diff,
    generate_photo_events,
    generate_shooting_events,
    robustness_table,
    split_targets,
)


OUT_TABLES = DATA_ROOT / "outputs" / "tables"
TRAJ_DIR = DATA_ROOT / "outputs" / "trajectories"


def _build_problem3_cv_fold_table() -> pd.DataFrame:
    att3 = read_position_workbook(DATA_ROOT / "附件3.xlsx")
    sheet1 = next(s for s in att3 if "方式1" in s)
    sheet2 = next(s for s in att3 if "方式2" in s)
    baseline = align_pair(att3[sheet1], att3[sheet2], estimate_bias=True, smooth_window=7)
    frame = _residual_frame(att3[sheet1], att3[sheet2], baseline.delta, smooth_window=7)
    features = ["t_norm", "x", "y", "v", "a", "cos_theta", "sin_theta", "curvature"]
    x = frame[features].to_numpy(float)
    y = frame[["residual_x", "residual_y"]].to_numpy(float)
    folds = _blocked_folds(len(frame), 5)
    rows: list[dict[str, float | int | str]] = []
    for fold_id, test_idx in enumerate(folds, start=1):
        train_mask = np.ones(len(frame), dtype=bool)
        train_mask[test_idx] = False
        for model_name in ["M0", "M1", "M2", "M3", "M4"]:
            pred, _, _ = _fit_predict(model_name, x[train_mask], y[train_mask], x[test_idx])
            err = y[test_idx] - pred
            rmse = float(np.sqrt(np.mean(np.sum(err * err, axis=1))))
            rmse_x = float(np.sqrt(np.mean(err[:, 0] ** 2)))
            rmse_y = float(np.sqrt(np.mean(err[:, 1] ** 2)))
            rows.append(
                {
                    "model_name": model_name,
                    "fold": fold_id,
                    "n_test": int(len(test_idx)),
                    "rmse": rmse,
                    "rmse_x": rmse_x,
                    "rmse_y": rmse_y,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLES / "attachment3_cv_fold_errors.csv", index=False, encoding="utf-8-sig")
    return out


def _photo_target_count(events: pd.DataFrame) -> int:
    if events.empty or "covered_targets" not in events.columns:
        return 0
    covered = {target for text in events["covered_targets"].astype(str) for target in text.split(",")}
    return len(covered)


def _expected_hits(events: pd.DataFrame) -> float:
    if events.empty or "event_type" not in events.columns:
        return 0.0
    shooting = events[events["event_type"] == "shooting"]
    if shooting.empty:
        return 0.0
    total = 0.0
    for _, part in shooting.groupby("target_id"):
        total += 1.0 - 0.15 ** len(part)
    return float(total)


def _build_problem4_ab_robustness_table() -> pd.DataFrame:
    rows = []
    mapping = [
        ("A", "joint_selected_events.csv"),
        ("B", "joint_selected_events_B_margin_first.csv"),
    ]
    for plan_label, file_name in mapping:
        events = pd.read_csv(OUT_TABLES / file_name)
        robust = robustness_table(events)
        rows.append(
            {
                "plan": plan_label,
                "total_events": int(len(events)),
                "shooting_targets_covered": int(events[events["event_type"] == "shooting"]["target_id"].nunique()),
                "photo_targets_covered": int(_photo_target_count(events[events["event_type"] == "photo"])),
                "expected_shooting_hits": _expected_hits(events),
                "combined_target_utility": float(_expected_hits(events) + _photo_target_count(events[events["event_type"] == "photo"])),
                "mean_nominal_margin": float(events["margin"].mean()),
                "min_nominal_margin": float(events["margin"].min()),
                "mean_worst_margin": float(robust["worst_margin"].mean()),
                "min_worst_margin": float(robust["worst_margin"].min()),
                "mean_scenario_pass_rate": float(robust["scenario_pass_rate"].mean()),
                "high_risk_event_count": int((robust["risk_level"] == "高").sum()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLES / "joint_ab_robustness_compare.csv", index=False, encoding="utf-8-sig")
    return out


def _dt_specific_traj(base: pd.DataFrame, dt: float, smooth_seconds: float = 8.1) -> pd.DataFrame:
    t = base[TIME_COL].to_numpy(float)
    start, end = float(t[0]), float(t[-1])
    grid = np.round(np.arange(start, end + 1e-9, dt), 3)
    out = {TIME_COL: grid}
    for col in ["x1_aligned", "y1_aligned", "x2_aligned_corrected", "y2_aligned_corrected", X_COL, Y_COL]:
        spline = CubicSpline(t, base[col].to_numpy(float))
        out[col] = spline(grid)
    traj = pd.DataFrame(out)
    smooth_window = max(5, int(round(smooth_seconds / dt)))
    if smooth_window % 2 == 0:
        smooth_window += 1
    return add_kinematics(traj, smooth_window=smooth_window)


def _generate_shooting_events_dt(traj: pd.DataFrame, shooting_targets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    xy = traj[[X_COL, Y_COL]].to_numpy(float)
    speed = traj["speed"].to_numpy(float)
    acc = traj["acceleration"].to_numpy(float)
    times = traj[TIME_COL].to_numpy(float)
    dt = float(np.median(np.diff(times)))
    prep = max(1, int(round(1.5 / dt)))
    for _, target in shooting_targets.iterrows():
        tx, ty = float(target[X_COL]), float(target[Y_COL])
        dist = np.sqrt((xy[:, 0] - tx) ** 2 + (xy[:, 1] - ty) ** 2)
        for i in range(prep, len(traj)):
            dwin = dist[i - prep : i + 1]
            swin = speed[i - prep : i + 1]
            awin = acc[i - prep : i + 1]
            if np.all((dwin >= 5.0) & (dwin <= 30.0)) and np.all(swin <= 2.0) and np.all(awin <= 1.5):
                margin = float(
                    min(
                        np.min((dwin - 5.0) / 25.0),
                        np.min((30.0 - dwin) / 25.0),
                        np.min((2.0 - swin) / 2.0),
                        np.min((1.5 - awin) / 1.5),
                    )
                )
                rows.append(
                    {
                        "event_id": f"S_{len(rows):05d}",
                        "event_type": "shooting",
                        "target_id": str(target["编号"]),
                        "start_time": float(times[i - prep]),
                        "execute_time": float(times[i]),
                        "distance": float(dist[i]),
                        "speed": float(speed[i]),
                        "acceleration": float(acc[i]),
                        "margin": margin,
                    }
                )
    return pd.DataFrame(rows)


def _generate_photo_events_dt(traj: pd.DataFrame, photo_targets: pd.DataFrame, fov_degree: float = 45.0) -> pd.DataFrame:
    rows = []
    xy = traj[[X_COL, Y_COL]].to_numpy(float)
    speed = traj["speed"].to_numpy(float)
    acc = traj["acceleration"].to_numpy(float)
    times = traj[TIME_COL].to_numpy(float)
    dt = float(np.median(np.diff(times)))
    prep = max(1, int(round(0.5 / dt)))
    stride = prep
    target_xy = photo_targets[[X_COL, Y_COL]].to_numpy(float)
    target_ids = photo_targets["编号"].astype(str).tolist()
    half_fov = fov_degree / 2.0
    for i in range(prep, len(traj), stride):
        if np.any(speed[i - prep : i + 1] > 1.5) or np.any(acc[i - prep : i + 1] > 1.5):
            continue
        vec = target_xy - xy[i]
        dist_now = np.sqrt(np.sum(vec * vec, axis=1))
        visible = np.where((dist_now >= 10.0) & (dist_now <= 40.0))[0]
        if len(visible) == 0:
            continue
        angles = np.degrees(np.arctan2(vec[visible, 1], vec[visible, 0]))
        directions = sorted(set(round(float(a) / 5.0) * 5.0 for a in angles))
        for phi in directions:
            covered_targets = []
            covered_angles = []
            covered_distances = []
            margins = []
            for idx, ang in zip(visible, angles):
                if circular_angle_diff(phi, ang) > half_fov:
                    continue
                dwin = np.sqrt(np.sum((target_xy[idx] - xy[i - prep : i + 1]) ** 2, axis=1))
                if not np.all((dwin >= 10.0) & (dwin <= 40.0)):
                    continue
                covered_targets.append(target_ids[idx])
                covered_angles.append(float(ang))
                covered_distances.append(float(dist_now[idx]))
                margins.append(
                    float(
                        min(
                            np.min((dwin - 10.0) / 30.0),
                            np.min((40.0 - dwin) / 30.0),
                            np.min((1.5 - speed[i - prep : i + 1]) / 1.5),
                            np.min((1.5 - acc[i - prep : i + 1]) / 1.5),
                        )
                    )
                )
            if not covered_targets:
                continue
            rows.append(
                {
                    "event_id": f"P_{len(rows):05d}",
                    "event_type": "photo",
                    "start_time": float(times[i - prep]),
                    "execute_time": float(times[i]),
                    "camera_direction": float(phi),
                    "covered_targets": ",".join(covered_targets),
                    "num_covered_targets": int(len(covered_targets)),
                    "target_angles": ",".join(f"{a:.1f}" for a in covered_angles),
                    "speed": float(speed[i]),
                    "acceleration": float(acc[i]),
                    "min_distance": float(np.min(covered_distances)),
                    "max_distance": float(np.max(covered_distances)),
                    "margin": float(np.min(margins)),
                    "fov_degree": float(fov_degree),
                }
            )
    return pd.DataFrame(rows)


def _build_discretization_sensitivity_table() -> pd.DataFrame:
    att3 = read_position_workbook(DATA_ROOT / "附件3.xlsx")
    _, fused = fuse_attachment(att3, estimate_bias=True, smooth_window=7)
    base = add_kinematics(fused, smooth_window=81)
    targets = read_targets(DATA_ROOT / "附件4.xlsx")
    shooting_targets, photo_targets = split_targets(targets)
    rows = []
    for dt in [0.20, 0.10, 0.05]:
        traj = _dt_specific_traj(base, dt)
        shoot_events = _generate_shooting_events_dt(traj, shooting_targets)
        photo_events = _generate_photo_events_dt(traj, photo_targets, fov_degree=45.0)
        plan, status = _select_joint_plan_milp(
            shoot_events,
            photo_events,
            photo_targets,
            min_margin=-np.inf,
            risk_weight=0.20,
            time_limit=45.0,
        )
        rows.append(
            {
                "dt_second": dt,
                "frequency_hz": round(1.0 / dt, 2),
                "trajectory_points": int(len(traj)),
                "shooting_candidates": int(len(shoot_events)),
                "photo_candidates": int(len(photo_events)),
                "joint_candidates": int(len(shoot_events) + len(photo_events)),
                "selected_events": int(len(plan)),
                "expected_shooting_hits": _expected_hits(plan),
                "photo_targets_covered": int(_photo_target_count(plan[plan["event_type"] == "photo"])),
                "combined_target_utility": float(_expected_hits(plan) + _photo_target_count(plan[plan["event_type"] == "photo"])),
                "min_nominal_margin": float(plan["margin"].min()) if not plan.empty else np.nan,
                "time_conflict_edges": int(status["n_time_conflict_edges"]),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLES / "discretization_sensitivity.csv", index=False, encoding="utf-8-sig")
    return out


def _build_rts_candidate_impact_table() -> pd.DataFrame:
    targets = read_targets(DATA_ROOT / "附件4.xlsx")
    shooting_targets, photo_targets = split_targets(targets)
    att3 = read_position_workbook(DATA_ROOT / "附件3.xlsx")
    _, fused = fuse_attachment(att3, estimate_bias=True, smooth_window=7)
    rows = []
    for label, smooth_window in [("raw_diff", 1), ("rts_style", 81)]:
        traj = add_kinematics(fused.copy(), smooth_window=smooth_window)
        shoot_events = generate_shooting_events(traj, shooting_targets, stride=1)
        photo_events = generate_photo_events(traj, photo_targets, fov_degree=45.0, time_stride=5)
        plan, status = _select_joint_plan_milp(
            shoot_events,
            photo_events,
            photo_targets,
            min_margin=-np.inf,
            risk_weight=0.20,
            time_limit=45.0,
        )
        rows.append(
            {
                "trajectory_version": label,
                "smooth_window_points": smooth_window,
                "speed_p95": float(np.percentile(traj["speed"], 95)),
                "acceleration_p95": float(np.percentile(traj["acceleration"], 95)),
                "shooting_candidates": int(len(shoot_events)),
                "photo_candidates": int(len(photo_events)),
                "joint_candidates": int(len(shoot_events) + len(photo_events)),
                "selected_events": int(len(plan)),
                "expected_shooting_hits": _expected_hits(plan),
                "photo_targets_covered": int(_photo_target_count(plan[plan["event_type"] == "photo"])) if not plan.empty and "event_type" in plan.columns else 0,
                "combined_target_utility": float(_expected_hits(plan) + (_photo_target_count(plan[plan["event_type"] == "photo"]) if not plan.empty and "event_type" in plan.columns else 0)),
                "min_nominal_margin": float(plan["margin"].min()) if not plan.empty and "margin" in plan.columns else np.nan,
                "time_conflict_edges": int(status["n_time_conflict_edges"]),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLES / "rts_candidate_impact.csv", index=False, encoding="utf-8-sig")
    return out


def main() -> None:
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    TRAJ_DIR.mkdir(parents=True, exist_ok=True)
    _build_problem3_cv_fold_table()
    _build_problem4_ab_robustness_table()
    _build_discretization_sensitivity_table()
    _build_rts_candidate_impact_table()


if __name__ == "__main__":
    main()
