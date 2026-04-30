# 论文图表来源说明

本文未重新估计模型参数，论文中的数值结论均来自：

`/home/lux_liang/work/projects/github/数模校赛B题/outputs/`

## 图来源

- `figures/generated/fig_pipeline_overview.pdf`：由仓库根目录 `框架图.pdf` 复制生成，用于展示总体技术路线。
- `figures/paper/fig1_attachment1_alignment.pdf`：由 `outputs/trajectories/fused_attachment1_10hz.csv` 和 `附件1.xlsx` 生成，展示附件1时间对齐前后轨迹。
- `figures/paper/fig2_attachment2_correction.pdf`：由 `outputs/trajectories/fused_attachment2_10hz.csv` 和 `附件2.xlsx` 生成，展示附件2固定平移型相对偏差校正前后轨迹。
- `figures/paper/fig3_attachment2_residuals.pdf`：由 `outputs/tables/attachment2_residuals.csv` 生成，展示附件2校正后残差散点。
- `figures/paper/fig4_attachment3_fused_traj.pdf`：由 `outputs/trajectories/fused_attachment3_10hz.csv` 生成，展示附件3融合10Hz轨迹。
- `figures/paper/fig5_attachment3_kinematics.pdf`：由 `outputs/trajectories/fused_attachment3_10hz.csv` 生成，展示附件3速度和加速度曲线。
- `figures/paper/fig6_task_distribution.pdf`：由 `outputs/trajectories/fused_attachment3_10hz.csv`、`附件4.xlsx`、`outputs/tables/selected_tasks_R5_multi_uncertainty.csv` 生成，展示 R5 鲁棒主结果的任务点分布。
- `figures/paper/fig7_task_timeline.pdf`：由 `outputs/tables/selected_tasks_R5_multi_uncertainty.csv` 和 `outputs/tables/final_task_stability_audit_v3.csv` 生成，展示最终任务准备窗口 Gantt 图。
- `figures/paper/fig8_robust_model_comparison.pdf`：由 `outputs/tables/robust_task_model_comparison.csv` 和 `outputs/tables/robust_task_model_comparison_v3.csv` 生成，展示 R1/R3/R4/R5/R6 的裕度和与场景可行率。
- `figures/paper/fig9_smoothing_audit.pdf`：由 `outputs/tables/oversmoothing_audit.csv` 生成，展示平滑窗口对候选任务数量和状态估计偏移的影响。
- `figures/paper/fig10_task_feasibility_heatmap.pdf`：由 `outputs/tables/multi_uncertainty_task_pool.csv` 生成，展示多场景候选任务在目标--时间二维空间中的可行性和平均稳定裕度。

生成脚本：

`scripts/make_paper_figures.py`

## 表来源

- 表1：来自 `outputs/tables/alignment_validation.csv` 和 `outputs/tables/alignment_summary.csv`。
- 表2：来自 `outputs/tables/system_bias_test.csv`。
- 表3：来自 `outputs/tables/bias_model_selection.csv`。
- 表4：来自 `outputs/tables/robust_task_model_comparison_v3.csv`。
- 表5及附录完整复核表：来自 `outputs/tables/selected_tasks_R5_multi_uncertainty.csv` 和 `outputs/tables/final_task_stability_audit_v3.csv`。

## 填表结果

最终 Excel 结果文件为：

`/home/lux_liang/work/projects/github/数模校赛B题/outputs/result_filled.xlsx`

本论文写作过程未修改该文件。
