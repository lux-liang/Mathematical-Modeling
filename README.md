# Mathematical-Modeling

全国大学生数学建模竞赛 LaTeX 极简模板。

这个仓库只保留一版简洁可用的论文模板，不包含广告、教程、示例图片或无关资料。

## 文件

- `main.tex`: 论文主文件
- `cumcmthesis.cls`: 模板类文件
- `LICENSE`: 上游模板所附 MIT 许可

## 编译

使用 `XeLaTeX`：

```bash
xelatex main.tex
```

电子版提交建议使用：

```tex
\documentclass[withoutpreface,bwprint]{cumcmthesis}
```

如果需要纸质版封面和编号页，可去掉 `withoutpreface`。

## 来源

当前模板基于 `tinoryj/Mathematical-Contest-in-Modeling` 仓库中的 `CUMCM_tinoryj_Template` 整理而来，并做了极简化处理。
