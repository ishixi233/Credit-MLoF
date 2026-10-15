"""模型登记表。每个模型给出 build（返回含预处理的 Pipeline）和 grid（候选参数列表）。

添加自己的模型：照着下面的写法加一个 build 函数，再在 MODELS 里登记一项，然后运行
    python scripts/03_run_cv.py <模型名>
grid 在实验前写定，较简单的配置放前面（平均 AP 并列时取靠前的）。
"""
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from src import config as C
from src.features.preprocess import make_preprocessor


def baseline_rate():
    """常数分数基准：对所有人输出训练部分的违约率。"""
    return Pipeline([("prep", make_preprocessor(scale=False)), ("clf", DummyClassifier(strategy="prior"))])


def logistic(C_reg: float, pay_onehot: bool):
    return Pipeline(
        [
            ("prep", make_preprocessor(scale=True, pay_onehot=pay_onehot)),
            ("clf", LogisticRegression(C=C_reg, max_iter=5000, random_state=C.SEED)),
        ]
    )


def gaussian_nb(pay_onehot: bool):
    return Pipeline([("prep", make_preprocessor(scale=True, pay_onehot=pay_onehot)), ("clf", GaussianNB())])


def knn(n_neighbors: int):
    """距离和投票权重固定为默认（欧氏距离、等权投票），只调 k。"""
    return Pipeline(
        [
            ("prep", make_preprocessor(scale=True)),
            ("clf", KNeighborsClassifier(n_neighbors=n_neighbors)),
        ]
    )


def decision_tree(max_depth: int):
    return Pipeline(
        [
            ("prep", make_preprocessor(scale=False)),
            ("clf", DecisionTreeClassifier(max_depth=max_depth, random_state=C.SEED)),
        ]
    )


def random_forest(max_depth):
    return Pipeline([
        ("prep", make_preprocessor(scale=False)),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=max_depth,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=C.SEED,
            n_jobs=-1
        ))
    ])


def adaboost(learning_rate):
    return Pipeline([
        ("prep", make_preprocessor(scale=False)),
        ("clf", AdaBoostClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=1,
                random_state=C.SEED
            ),
            n_estimators=100,
            learning_rate=learning_rate,
            random_state=C.SEED
        ))
    ])


MODELS = {
    "baseline_rate": {"build": baseline_rate, "grid": [{}]},
    "logistic": {
        "build": logistic,
        "grid": [{"C_reg": c, "pay_onehot": p} for p in (False, True) for c in (0.1, 1.0, 10.0)],
    },
    # 只用一套默认配置。PAY one-hot 试过：五折 AP 0.491 对 0.487，差距小于折间标准差（约 0.015），
    # 且 0/1 变量不符合高斯假设、会把八成客户的分数压到接近 0，所以不采用。
    "gaussian_nb": {"build": gaussian_nb, "grid": [{"pay_onehot": False}]},
    # k 越大决策越平滑，较简单的放前面
    "knn": {"build": knn, "grid": [{"n_neighbors": 61}, {"n_neighbors": 31}, {"n_neighbors": 11}]},
    "decision_tree": {"build": decision_tree, "grid": [{"max_depth": 3}, {"max_depth": 5}, {"max_depth": 8}]},
    "random_forest": {
        "build": random_forest,
        "grid": [
            {"max_depth": 5},
            {"max_depth": 10},
            {"max_depth": None}
        ],
    },

    "adaboost": {
        "build": adaboost,
        "grid": [
            {"learning_rate": 0.1},
            {"learning_rate": 0.5},
            {"learning_rate": 1.0}
        ],
    },
}
