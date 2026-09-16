# 02 Credit_Default_Proposal_Example

[**返回导航**](00_选题筛选与使用说明.md) · [**上一章**](01_Proposal_English_Template.md) · [**下一章**](03_信用卡违约项目总体工作流程.md)

> **Illustrative proposal for discussion.** The study below is planned, not completed. Member names and dates are placeholders. This is not an official template or a claim of instructor approval.

# Credit Card Default Prediction: Comparing Simple Models and Decision Thresholds

**Course:** MH6805 Machine Learning in Finance  
**Group members:** Shi Xi, Li Erteng, Jiang Chengrui  
**Submission date:** [Confirm against the latest course announcement]

## 1. Motivation and Research Question

Credit risk assessment requires balancing the detection of potential defaults against false alarms for customers who will repay. A model with high overall accuracy may still miss many defaults, and a fixed probability threshold may not reflect the different consequences of these errors.

This project will compare lightweight classification models for predicting credit card default. Our main question is: **How do model choice and decision threshold affect the trade-off between default detection and false alarms?** The project relates to credit financing and risk analysis in the suggested FinTech research directions. It will remain a comparative data-analysis project, with no requirement to build a lending platform or develop a new algorithm.

## 2. Data and Task

We plan to use the UCI *Default of Credit Card Clients* dataset [1]. The repository describes 30,000 client records with 23 candidate predictors and a binary default target. Inputs include credit limits, repayment history, bill amounts and previous payments. The data concern a historical Taiwanese customer sample, with monthly predictors covering April–September 2005.

The target will be default in the following month. We will exclude the client identifier, inspect category codes and duplicates, and document any recoding. Information used as a predictor must precede the target period. Dataset dimensions and class proportions will be verified after download.

## 3. Proposed Methods

We will compare Logistic Regression, Gaussian Naive Bayes, K-Nearest Neighbours, Decision Tree, Random Forest and AdaBoost. A majority-class classifier will provide a simple reference baseline. These methods cover linear, probabilistic, distance-based, tree and ensemble approaches taught in the course while keeping implementation manageable.

We will apply suitable categorical encoding and scale numeric features when needed, particularly for Logistic Regression and KNN. Gaussian Naive Bayes will serve as a simple benchmark despite its restrictive assumptions. Preprocessing will be fitted on training data only. Each method will use a small, pre-specified hyperparameter search rather than extensive optimisation.

The only planned extension is decision-threshold analysis. Deep learning, synthetic oversampling, extra datasets and production deployment are outside the core scope.

## 4. Evaluation Plan

We will use a fixed, stratified 60/20/20 training, validation and test split, after checking for repeated client identifiers and duplicate records. Related records, where identified, will be assigned consistently to avoid leakage. The validation partition will be used for hyperparameter and threshold selection. The test set will remain untouched until all modelling decisions have been fixed.

The primary metric will be average precision (AP), supplemented by ROC-AUC, precision, recall and confusion matrices. We will compare the default probability threshold of 0.5 with a validation-selected threshold. For one illustrative scenario, we will assign a false negative five times the cost of a false positive and choose the threshold that minimises this weighted error on the validation set. This ratio is a stated hypothetical assumption, not an estimate of a bank's actual losses. The selected threshold will then be applied unchanged to test data.

All models will use the same partitions and evaluation procedure. We will report both missed defaults and false alarms rather than relying on accuracy alone.

## 5. Expected Contribution and Limitations

The expected contribution is a reproducible comparison of simple models and a clear explanation of how threshold selection changes their practical error trade-offs. We will not assume that an ensemble necessarily outperforms a simpler model.

The study uses an older, geographically limited dataset. Its random holdout design measures generalisation within that historical sample and does not establish performance in later years or other credit markets. Feature associations will not be interpreted as causal effects. Any weighted error comparison will remain conditional on the specified cost assumption.

No empirical performance results are claimed at this stage.

## 6. Team Responsibilities and Timeline

| Member | Models | Main supporting task |
|---|---|---|
| [Name 1] | Logistic Regression; Gaussian Naive Bayes | Data inspection and preprocessing documentation |
| [Name 2] | KNN; Decision Tree | Validation design and experiment records |
| [Name 3] | Random Forest; AdaBoost | Results visualisation and threshold analysis |

All members will review the shared pipeline, interpret results and contribute to the final report and presentation.

We plan to complete data inspection and baseline models during [period 1], model comparison during [period 2], threshold and error analysis during [period 3], and report preparation during [period 4]. Specific dates will be aligned with the confirmed course deadlines.

## 7. Reference

[1] Yeh, I. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H. [Official dataset page](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).

---

## Points to Decide Before Submission

1. Confirm that the dataset and research question are acceptable for the course.
2. Replace member names, allocate responsibilities and fill in the schedule.
3. Decide whether to retain the proposed 5:1 illustrative error-cost ratio, or select another rule before examining validation results.
4. Check the instructor's latest format, length and submission instructions; this example does not establish those requirements.
5. If the group changes any design choice, update the question, methods and evaluation sections together so that they remain consistent.

---

[**返回导航**](00_选题筛选与使用说明.md) · [**上一章**](01_Proposal_English_Template.md) · [**下一章**](03_信用卡违约项目总体工作流程.md)
