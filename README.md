# Mathematical-Modeling

全国大学生数学建模竞赛常用 LaTeX 模板仓库。

当前仓库已放入 `cumcmthesis` 模板，可直接从 `example.tex` 开始修改。

## 文件说明

- `cumcmthesis.cls`: 模板类文件
- `example.tex`: 示例论文源码
- `example.pdf`: 示例编译结果
- `figures/`: 示例图片资源

## 使用方法

建议使用 `XeLaTeX` 编译：

```bash
xelatex example.tex
```

电子版提交通常需要去掉承诺书和编号页，可使用：

```tex
\documentclass[withoutpreface,bwprint]{cumcmthesis}
```

保留封面页时可使用：

```tex
\documentclass[bwprint]{cumcmthesis}
```

## 上游来源

本仓库当前内容整理自：

- `latexstudio/CUMCMThesis`
- 上游地址：<https://github.com/latexstudio/CUMCMThesis>
- 本次采用的上游提交：`90d3e854534ae7dc605dfe9296785f8c17e56e22`

说明见 `UPSTREAM.md`。
