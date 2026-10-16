"""报告用图的统一样式，对齐 Proposal 的 NeurIPS 模板：正文 10 pt、图注 9 pt、文本宽 5.5 in、衬线字体。

用法：
    from src.visualization.style import apply, FULL, HALF, LABELS, COLORS
    apply()
    fig, ax = plt.subplots(figsize=HALF)
PNG 用 300 dpi；同时存 PDF 供 LaTeX 直接引用（矢量，字体不糊）。
"""
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

TEXT_WIDTH = 5.5  # in，neurips_2021.sty 的 textwidth
FULL = (TEXT_WIDTH, 3.1)  # 通栏
HALF = (TEXT_WIDTH / 2 - 0.1, 2.5)  # 并排两张

MODEL_ORDER = ("random_forest", "adaboost", "logistic", "knn", "decision_tree", "gaussian_nb")
LABELS = {
    "random_forest": "Random Forest",
    "adaboost": "AdaBoost",
    "logistic": "Logistic Regression",
    "knn": "KNN",
    "decision_tree": "Decision Tree",
    "gaussian_nb": "Gaussian NB",
    "baseline_rate": "Constant baseline",
}
# 色盲友好的 Okabe–Ito 调色板，六个模型固定配色，所有图一致
COLORS = {
    "random_forest": "#0072B2",
    "adaboost": "#D55E00",
    "logistic": "#009E73",
    "knn": "#CC79A7",
    "decision_tree": "#E69F00",
    "gaussian_nb": "#56B4E9",
    "baseline_rate": "#999999",
}


def apply():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIX Two Text", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.2,
            "lines.markersize": 3.5,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
        }
    )


def save(fig, path_without_ext):
    """同时保存 PNG（看图、放 PPT）和 PDF（LaTeX 引用）。"""
    fig.savefig(f"{path_without_ext}.png")
    fig.savefig(f"{path_without_ext}.pdf")
    plt.close(fig)
