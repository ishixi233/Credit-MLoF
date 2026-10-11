# Credit-MLoF｜信用卡违约预测

这是我们 **MH6805 Machine Learning in Finance** 小组项目的共享仓库，用来统一管理 Proposal、数据和后续实验代码。

**组员：Shi Xi、Li Erteng、Jiang Chengrui。**

## 我们要做什么？

使用信用卡客户的历史信息预测下一月是否违约，比较不同分类模型，并研究调整分类阈值后，**漏掉违约客户**和**误报正常客户**之间的权衡。

目前处于 **实验阶段**：Proposal 已提交并收到评语；共享的数据划分、预处理、评价函数和交叉验证流程已就绪，基准、Logistic Regression 和 Gaussian NB 已在开发集上跑通。测试集尚未使用。

**Proposal 之后改了什么、还剩什么要做，见 [Proposal 之后的修改说明与工作计划](docs/Proposal后修改说明与工作计划.md)**，剩余任务和截止日期在该文档开头。

## 怎么运行

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest                      # 检查划分和评价函数
python scripts/03_run_cv.py logistic  # 在共享五折上跑一个模型
```

**不要重新运行 `scripts/02_make_splits.py`**，划分文件 `data/splits/splits.csv` 已提交，全组直接读取。

添加自己的模型只需两步：

1. 在 [src/models/registry.py](src/models/registry.py) 里照着 `logistic` 写一个 build 函数，并在 `MODELS` 里登记模型名和参数网格。树模型用 `make_preprocessor(scale=False)`，KNN 用 `scale=True`。
2. 运行 `python scripts/03_run_cv.py <模型名>`。

脚本会按五折平均 AP 选一套参数，并写出三个文件，全部提交进仓库：

| 文件 | 内容 |
|---|---|
| `results/predictions/oof_<模型名>.csv` | 折外预测，列为 `ID, fold, y, score`；阈值、成本比和校准分析都读它 |
| `results/experiments/cv_<模型名>.csv` | 每套参数、每折的 AP、ROC-AUC、Brier 和耗时 |
| `results/experiments/selected_<模型名>.json` | 选定的参数及五折汇总 |

评价一律调用 [src/evaluation/metrics.py](src/evaluation/metrics.py) 里的函数，不要各自重写：`ranking_metrics`（AP、ROC-AUC、Brier）、`select_threshold` 和 `cost_curve`（按成本比选阈值）、`decision_metrics`（混淆矩阵与成本）、`trivial_costs`（全报警 / 全不报警的参照成本）、`calibration_table`（校准表）。

`load_test()` 在 `configs/frozen.json` 提交之前会直接报错，这是有意的：模型和阈值冻结之前不能读测试集。

数据字段和预处理约定见 [数据字典](docs/data_dictionary.md)，原始数据核查结果见 [数据核查记录](docs/data_check.md)。

## 先看哪些文件？

| 文件 | 用途 |
|---|---|
| [Proposal 之后的修改说明与工作计划](docs/Proposal后修改说明与工作计划.md) | **先读这份**：剩余任务与截止日期、老师评语的回应、相对 Proposal 的改动、分工、报告结构 |
| [数据字典](docs/data_dictionary.md) · [数据核查记录](docs/data_check.md) | 字段含义、预处理约定、划分规模；原始数据的取值分布 |
| [英文 Proposal 正式稿](Proposal/02_Credit_Default_Proposal.md) | 已提交的研究设计；与修改说明不一致处以修改说明为准 |
| [Proposal PDF](Proposal/LaTeX/Credit_Default_Proposal.pdf) | 按教师 LaTeX 模板排版，共 3 页，含参考文献 |
| [LaTeX 源码](Proposal/LaTeX/Credit_Default_Proposal.tex) · [编译说明](Proposal/LaTeX/README.md) | 使用 MacTeX 或 TeXShop 编辑和重新编译 |
| [中文说明](Proposal/04_Proposal_中文说明.md) | Markdown 说明文档，解释指标、阈值分析及具体执行安排 |
| [项目总体工作流程](Proposal/03_信用卡违约项目总体工作流程.md) | Proposal 阶段写的分步流程，可作背景参考；时间安排以修改说明为准 |
| [英文 Proposal 模板](Proposal/01_Proposal_English_Template.md) | 修改结构和表述时参考 |
| [选题与使用说明](Proposal/00_选题筛选与使用说明.md) | 查看前期选题背景和文档说明 |
| [原始数据文件](<default of credit card clients.xls>) | 后续全组共用的数据，保留原文件，清洗结果另存 |
| [教师项目要求](<Proposal/Group Project Guideline.pdf>) · [课堂总结](<Proposal/Group Project Summary.pdf>) | 核对课程要求和评价流程；具体日期以当期通知为准 |

建议先读 **修改说明与工作计划**，再按上面的“怎么运行”开始实验。研究问题和整体设计以正式英文 Proposal 为准，Proposal 之后的调整以修改说明为准。

## 当前实验方案

- **模型**：Logistic Regression、Gaussian Naive Bayes、KNN、Decision Tree、Random Forest、AdaBoost；另设多数类基准。
- **数据划分**：70% 训练开发集、30% 独立测试集；训练部分内部进行五折分层交叉验证，全组共用划分与验证折。
- **评价**：用五折平均 AP（Average Precision）选配置，最终报告测试 AP、ROC-AUC、Brier score、分类错误率、precision、recall 和混淆矩阵。
- **扩展分析**：比较默认阈值 0.50 与折外验证预测选出的阈值。漏报 / 误报成本比以 5:1 为主情景（引用 German Credit 成本矩阵作为使用先例，仅作示例），并按老师评语加做 1:1、2:1、3:1、10:1 的敏感性分析；同时报告 Brier score 和校准曲线。
- **分工**：Shi Xi 负责 Logistic Regression 和 Gaussian Naive Bayes；Li Erteng 负责 KNN 和 Decision Tree；Jiang Chengrui 负责 Random Forest 和 AdaBoost。报告和展示共同完成。

## 大家接下来做什么？

1. 拉取最新代码，按“怎么运行”建好环境并跑通测试。
2. 在登记表里加入自己负责的两个模型，跑完五折并提交折外预测。
3. 按 [修改说明与工作计划](docs/Proposal后修改说明与工作计划.md) 开头的表格完成各自的分析、PPT 和报告部分。

沟通和任务提醒放在群里，正式文件统一保存在这个仓库。修改前先同步最新版本；较大的修改使用独立分支，通过 Pull Request 让组员检查后合并。参数和阈值在训练部分内部选择，随后在完整 70% 上拟合最终模型并冻结；30% 测试集留到方案确定后再统一评估。

## 数据来源

UCI **Default of Credit Card Clients**。来源与引用信息见英文 Proposal；数据规模、字段和编码已核验，见 [数据核查记录](docs/data_check.md)。

## 后续文件放在哪里？

以下目录按实际进度逐步加入文件，目前已有内容的是 `data/splits/`、`docs/`、`src/`、`scripts/`、`tests/` 和 `results/`。空目录中的 `.gitkeep` 仅用于让 GitHub 显示目录，有实际文件后可删除。

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

现有 `Proposal/` 继续存放研究计划；根目录的 `default of credit card clients.xls` 保持原位置，避免已有链接失效。
