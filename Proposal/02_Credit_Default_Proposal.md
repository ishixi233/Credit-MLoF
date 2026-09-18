# Credit Card Default Prediction

**Comparing Classification Models and Decision Thresholds**

**Project proposal**

**Course:** MH6805 Machine Learning in Finance  
**Institution:** Nanyang Technological University  
**Group members:** Shi Xi, Li Erteng, Jiang Chengrui

## Abstract

We will compare six classifiers for next-month credit card default using the UCI Default of Credit Card Clients dataset. Training-fold preprocessing and stratified cross-validation will support model and threshold selection before independent testing. The analysis will compare risk ranking, missed defaults and false alarms under a fixed illustrative 5:1 error-cost scenario.

## 1 Introduction and research question

Identifying customers at risk of default can support the prioritisation of credit-risk review. Missing a future default and incorrectly flagging a paying customer have different consequences, so accuracy alone is insufficient for evaluating a credit-risk classifier.

Our research question is: **How do classification models and decision thresholds affect the trade-off between detecting defaults and generating false alarms?** We will compare six established methods under a shared evaluation design and examine whether strong risk-ranking performance also produces low weighted error cost under a stated scenario.

## 2 Dataset and prediction task

We will use the UCI Default of Credit Card Clients dataset [1]. UCI documents 30,000 Taiwanese client records and 23 predictors, including credit limits, demographic attributes, repayment status, bill amounts and previous payments. Monthly histories cover April to September 2005. The target is default in the following month: 1 indicates default and 0 indicates no default.

The original Excel file is available in our [project repository](https://github.com/ishixi233/Credit-MLoF). Before modelling, we will verify its dimensions, label distribution, missing values, category codes and duplicates. Client ID will be excluded from predictors, and all recoding will be documented. Confirmed duplicates will be removed or grouped, and related records will remain within one partition and one cross-validation fold to prevent leakage. Only information available before the target period will be used.

## 3 Objectives

- Develop classification models to predict next-month credit card default.
- Compare predictive performance under a common evaluation framework.
- Evaluate the models using metrics suitable for imbalanced classification.
- Examine how decision thresholds affect missed defaults and false alarms under a stated cost scenario.

## 4 Proposed methods and preprocessing

We will implement Logistic Regression, Gaussian Naive Bayes, K-Nearest Neighbours (KNN), Decision Tree, Random Forest and AdaBoost in Python using scikit-learn. These cover linear, probabilistic, distance-based, tree and ensemble approaches. Majority-class predictions and a constant score equal to the training default rate will provide simple baselines.

Preprocessing will be part of each model pipeline. Categorical variables will be encoded consistently; numeric features will be scaled for models that require it, particularly Logistic Regression and KNN. Imputation, encoding and scaling will be fitted only on the training portion of each cross-validation fold and applied unchanged to its validation portion. Gaussian Naive Bayes will be treated as a simple benchmark whose conditional-independence and distributional assumptions may be imperfect.

We will specify small hyperparameter grids before experiments, covering regularisation strength, neighbour count, tree complexity and boosting settings as appropriate. For interpretation, we will discuss logistic-regression coefficients and tree-based feature importance, taking account of preprocessing and the limitations of these model-specific measures.

## 5 Model evaluation and threshold analysis

### 5.1 Data separation and cross-validation

We will reserve 30% of the data as an independent test set and use the remaining 70% for training and model development. The outer split and five-fold cross-validation within the training portion will be stratified, with random seed 42 and shared saved indices [2]. Group constraints will take priority if related records remain.

Mean cross-validation average precision (AP) [3] will select each model's configuration and the primary model for discussion; all six models will be reported. AP and ROC-AUC will use continuous scores for the default class. We will also report classification error rate, precision, recall and confusion matrices. Cross-validation mean and standard deviation will describe variation across folds, while final performance conclusions will rely on the independent test set.

### 5.2 Threshold selection and final testing

For each selected configuration, we will obtain out-of-fold probability estimates: each training client will be scored by a pipeline fitted without that client's validation fold. These predictions will select an alternative to the reference threshold of 0.50 [4]. A client will be flagged as at risk of default when the estimated probability is at least the threshold.

Threshold selection will minimise the illustrative weighted error cost

$$
C(t) = 5 FN(t) + FP(t),
$$

where FN denotes missed defaults and FP denotes false alarms. The 5:1 ratio follows the cost matrix documented for UCI's Statlog (German Credit Data), which assigns a cost of 5 to classifying a bad credit risk as good and 1 to the reverse error [5]. This is a precedent for our fixed illustrative scenario, not evidence that the ratio represents actual losses in the Taiwanese dataset.

The threshold search will cover all distinct decisions in the pooled out-of-fold predictions, including flagging everyone and no one; equal-cost choices will favour fewer false alarms. We will then refit each selected pipeline on the full 70% training portion and freeze both the fitted pipeline and its selected threshold before evaluating the 30% test set. Both thresholds will be compared using error counts and weighted error cost per client, C(t)/N, where N is the number of evaluated clients. Test results will not be used for further tuning, and any lack of improvement will be reported.

## 6 Current progress and expected outputs

We have selected the research topic and identified the UCI dataset as the main data source. Based on its published documentation, we have identified the intended target and candidate predictors and developed an initial modelling and evaluation plan. Data inspection and model training will follow proposal finalisation; no preliminary performance results are claimed.

We will deliver reproducible code, cross-validation and test comparison tables, threshold and error analyses, an interpretation of model behaviour, a written report and presentation slides.

## 7 Member contributions and work plan

Table 1 records the planned division of work. All members will review the common pipeline, explain their assigned models, interpret the results, and contribute to the report and English presentation. Each member will speak during the presentation and participate in questions and answers. Completed individual contributions will be recorded in the project deliverables.

**Table 1. Planned individual contributions**

| Member | Models | Supporting responsibility |
|---|---|---|
| Shi Xi | Logistic Regression; Gaussian Naive Bayes | Data checks, preprocessing and financial interpretation |
| Li Erteng | KNN; Decision Tree | Shared split, cross-validation and experiment records |
| Jiang Chengrui | Random Forest; AdaBoost | Results visualisation and threshold analysis |

After proposal finalisation, Week 1 will cover data checks, the shared split and baseline pipelines; Week 2 will cover cross-validation and model comparison; Week 3 will cover threshold selection, final fitting and testing; and Week 4 will cover interpretation, the report, slides and group review. Submission dates will follow the current course announcements.

## 8 Limitations

The data are historical and geographically limited. A random holdout assesses generalisation within this sample, not across future years or other markets. Probability estimates may be uncalibrated, and cross-validation results used for selection may be optimistic. Cost-based conclusions apply only to the stated 5:1 scenario; weighted errors are not monetary-loss estimates. Model coefficients and feature importance describe predictive associations, not causal effects or readiness for lending decisions.

## References

[1] Yeh, I. (2009). Default of Credit Card Clients [Dataset]. UCI Machine Learning Repository. [DOI: 10.24432/C55S3H](https://doi.org/10.24432/C55S3H).

[2] scikit-learn developers. Cross-validation: evaluating estimator performance. [Official documentation](https://scikit-learn.org/stable/modules/cross_validation.html).

[3] scikit-learn developers. Average precision score. [Official documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).

[4] scikit-learn developers. Tuning the decision threshold for class prediction. [Official documentation](https://scikit-learn.org/stable/modules/classification_threshold.html).

[5] Hofmann, H. (1994). Statlog (German Credit Data) [Dataset]. UCI Machine Learning Repository. [DOI: 10.24432/C5NC77](https://doi.org/10.24432/C5NC77). [Dataset documentation and cost matrix](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data).
