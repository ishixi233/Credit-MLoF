# 01 Proposal_English_Template

[**返回导航**](00_选题筛选与使用说明.md) · [**上一章**](00_选题筛选与使用说明.md) · [**下一章**](02_Credit_Default_Proposal_Example.md)

> **Working template for discussion, not an official course form.** Replace square-bracketed placeholders and remove writing prompts before submission. Do not report experiments that have not been performed. Confirm the submission date, format and length against the latest course announcement.

**Course:** MH6805 Machine Learning in Finance  
**Working title:** [Specific task + data/domain + comparison or question]  
**Group members:** [Name 1], [Name 2], [Name 3]  
**Submission date:** [Confirmed date]

## 1. Motivation and Research Question

[Writing prompt: Describe one practical financial problem, who could use the analysis, and the decision it could support. Keep the scope narrow.]

This project will investigate [financial problem] using [public dataset]. The practical motivation is [specific decision or analytical need]. We will focus on [one prediction or classification task] rather than attempting to build a complete production system.

Our main research question is:

> [Can / How do / To what extent do] [specified methods or design choices] [achieve a clearly defined goal] under [a stated evaluation setting]?

The project is related to [exact heading and item in Trending FinTech Research Topics]. We narrow this broad direction to [specific task that can be completed within the course].

## 2. Data and Prediction Task

[Writing prompt: Verify the source. Distinguish the unit of observation, target, predictors and prediction time.]

We plan to use [dataset name], available from [official source and URL]. According to the source documentation, the dataset contains [approximate number of observations] and covers [population, location and historical period, if available]. These details will be checked against the downloaded files.

Each observation represents [a client / a marketing contact / a sentence]. The target is [precise label definition], and the candidate predictors include [short list]. We will exclude [identifiers and information unavailable at the intended prediction time].

Before modelling, we will inspect [missing values, special category codes, duplicates and class balance]. Repeated entities or duplicate examples will be handled consistently when constructing the evaluation split. The study will use [one named dataset version]; we will not merge overlapping versions without checking their relationship.

## 3. Proposed Methods and Scope

[Writing prompt: Choose methods you can explain. Use the same evaluation data for all models.]

We will compare the following methods:

| Model | Purpose in the comparison |
|---|---|
| [Model 1] | [Simple, interpretable benchmark] |
| [Model 2] | [Alternative modelling assumption] |
| [Model 3] | [Distance-based or another lightweight method] |
| [Model 4] | [Nonlinear, interpretable comparison] |
| [Model 5] | [Bagging-based ensemble, if appropriate] |
| [Model 6] | [Boosting-based ensemble, if appropriate] |

A [majority-class / another appropriate naive] baseline will provide a reference point. Preprocessing will include [categorical encoding, numeric scaling or TF-IDF, as appropriate], fitted only on the training partition. Hyperparameter search will be limited to [a small, pre-specified set of candidates].

Our only planned extension beyond the main comparison is [threshold analysis / error analysis / one other manageable extension]. Additional datasets, large language model fine-tuning and system deployment are outside the planned scope.

## 4. Evaluation Design

[Writing prompt: Select a split that matches the data. Do not claim a random split tests forecasting over time.]

We will use [stratified random / chronological / grouped] training, validation and test partitions because [reason]. The test partition will remain untouched until the modelling choices have been fixed. Encoding, scaling, feature selection and any resampling will be fitted within the training data only.

The primary metric will be [AP for imbalanced binary classification / macro-F1 for multiclass sentiment / another justified metric]. Secondary measures will include [two or three appropriate metrics and a confusion matrix]. These metrics are appropriate because [explain the practical consequence of errors].

[For a binary classification project:] We will compare the default probability threshold with a threshold selected on validation data according to [pre-specified rule]. Any assumed misclassification costs will be presented as hypothetical scenarios rather than measured financial losses.

[For a text classification project:] Duplicate or related texts will be kept within the same partition where identifiable. We will inspect common error types and avoid evaluating a model on examples already used in its published fine-tuning data unless the overlap is explicitly resolved.

## 5. Expected Contribution and Limitations

The expected contribution is a transparent comparison of [methods] and an explanation of [a specific practical trade-off]. We do not assume in advance that the most complex model will perform best.

The main limitations are [historical or narrow sample], [available feature/label limitations] and [evaluation scope]. Our results will describe performance under the stated experimental design; they will not establish [causal effects / live deployment performance / investment profitability, as applicable].

No numerical performance claim is made at the proposal stage. If preliminary experiments are later included, their data split and status will be stated clearly.

## 6. Work Plan and Responsibilities

[Writing prompt: The annotated introduction slide mentions 2–3 models per person. The allocation below uses two models per person without adding a deep learning requirement.]

| Member | Model responsibility | Additional responsibility |
|---|---|---|
| [Name 1] | [Models 1–2] | [Data documentation and preprocessing] |
| [Name 2] | [Models 3–4] | [Validation design and experiment records] |
| [Name 3] | [Models 5–6] | [Results tables and visualisations] |

All members will use a shared dataset split and evaluation procedure and will contribute to interpreting results and writing the report.

| Stage | Planned work | Target date |
|---|---|---|
| Proposal | Confirm question, data, scope and responsibilities | [Date] |
| Data preparation | Inspect data and freeze evaluation partitions | [Date] |
| Modelling | Fit baselines and conduct limited tuning | [Date] |
| Analysis | Compare results and complete the planned extension | [Date] |
| Reporting | Prepare the final report and presentation | [Date] |

## 7. References

1. [Dataset creator. Year. Dataset title. Repository. DOI or official URL.]
2. [Relevant original paper or course reference actually consulted.]
3. [Any other data or method source actually used.]

---

## Discussion Questions — Remove Before Submission

- Can we define the target in one sentence without ambiguity?
- Can we obtain labelled data without paid access or manual annotation?
- Is every predictor available when the prediction would be made?
- Which single result table would answer our main question?
- Can we finish the core comparison if the optional extension fails?
- Are we comparing models under the same split and metrics?
- Have we kept expected findings separate from results already obtained?

**Possible titles to consider**

1. Credit Card Default Prediction: Comparing Simple Models and Decision Thresholds
2. Predicting Bank Marketing Responses Using Pre-Contact Customer Information
3. Financial News Sentiment Classification with Lightweight Machine Learning Models

---

[**返回导航**](00_选题筛选与使用说明.md) · [**上一章**](00_选题筛选与使用说明.md) · [**下一章**](02_Credit_Default_Proposal_Example.md)
