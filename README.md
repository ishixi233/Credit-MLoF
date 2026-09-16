# Credit-MLoF｜信用卡违约预测

这是我们 **MH6805 Machine Learning in Finance** 小组项目的共享仓库，用来统一管理 Proposal、数据和后续实验代码。

## 我们要做什么？

使用信用卡客户的历史信息预测下一月是否违约，比较不同分类模型，并研究调整分类阈值后，**漏掉违约客户**和**误报正常客户**之间的权衡。

目前处于 **Proposal 与实验准备阶段**：仓库已有研究计划和原始数据文件，尚未加入模型代码或实验结果。

## 先看哪些文件？

| 文件 | 用途 |
|---|---|
| [英文 Proposal 草稿](Proposal/02_Credit_Default_Proposal_Example.md) | 了解研究问题、方法和评价计划；讨论后补充姓名、分工与日期 |
| [项目总体工作流程](Proposal/03_信用卡违约项目总体工作流程.md) | 查看从数据准备到报告展示的具体步骤 |
| [英文 Proposal 模板](Proposal/01_Proposal_English_Template.md) | 修改结构和表述时参考 |
| [选题与使用说明](Proposal/00_选题筛选与使用说明.md) | 查看前期选题背景和文档说明 |
| [原始数据文件](<default of credit card clients.xls>) | 后续全组共用的数据，保留原文件，清洗结果另存 |

建议先读 **英文 Proposal 草稿**，再看 **项目总体工作流程**。文档中的方案与分工仍需全组确认。

## 暂定实验方案

- **模型**：Logistic Regression、Gaussian Naive Bayes、KNN、Decision Tree、Random Forest、AdaBoost；另设多数类基准。
- **数据划分**：训练 / 验证 / 测试 = 60% / 20% / 20%，全组共用固定划分。
- **评价**：以 AP（Average Precision）为主，结合 ROC-AUC、precision、recall 和混淆矩阵，不只看准确率。
- **扩展分析**：比较默认阈值 0.5 与验证集选出的阈值。草稿中的漏报 / 误报成本比 5:1 是待确认的情景假设。
- **分工**：暂按每人两个模型安排，报告和展示共同完成；具体分配见 Proposal。

## 大家接下来做什么？

1. 阅读 Proposal，讨论并确认研究设计、模型分工和时间安排。
2. 一起核对数据字段，统一清洗规则、数据划分与评价方式。
3. 跑通共同基准后，再分别实现各自负责的模型。

沟通和任务提醒放在群里，正式文件统一保存在这个仓库。修改前先同步最新版本；较大的修改使用独立分支，通过 Pull Request 让组员检查后合并。参数和阈值在验证集上选择，测试集留到方案确定后再统一评估。

## 数据来源

UCI **Default of Credit Card Clients**。来源与引用信息见英文 Proposal；实际数据规模、字段和编码在实验前统一核验。

## 后续文件放在哪里？

以下目录已建好，按实际进度逐步加入文件。空目录中的 `.gitkeep` 仅用于让 GitHub 显示目录，有实际文件后可删除。

| 目录 | 用途 |
|---|---|
| `data/raw/` | 后续新增的原始数据；当前 Excel 原文件仍在仓库根目录 |
| `data/processed/` | 清洗和预处理后的数据 |
| `data/splits/` | 全组统一的训练、验证、测试划分索引 |
| `docs/` | 数据字典、会议记录与分工说明 |
| `configs/` | 随机种子、模型参数和实验配置 |
| `src/data/` | 数据读取、清洗与划分代码 |
| `src/features/` | 特征编码和预处理代码 |
| `src/models/` | 六个模型与基准模型的训练代码 |
| `src/evaluation/` | 统一指标计算和阈值分析代码 |
| `src/visualization/` | 绘图代码 |
| `notebooks/` | 探索性分析与模型实验笔记本 |
| `scripts/` | 运行数据处理、训练和评估的入口脚本 |
| `models/` | 训练完成的模型及预处理器 |
| `results/experiments/` | 实验配置、运行日志与参数比较记录 |
| `results/predictions/` | 验证集与测试集预测结果 |
| `results/tables/` | 指标汇总与对照表 |
| `results/figures/` | 结果图与阈值曲线 |
| `report/` | 最终报告草稿与提交版本 |
| `slides/` | 汇报 PPT 与演讲备注 |
| `references/` | 参考文献与引用记录 |
| `tests/` | 数据处理和评价函数的检查代码 |

现有 `Proposal/` 继续存放研究计划；根目录的 `default of credit card clients.xls` 保持原位置，避免已有链接失效。当前目录仅为项目骨架，不代表已经完成训练或分析。
