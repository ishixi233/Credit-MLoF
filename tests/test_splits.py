import pytest

from src import config as C
from src.data.load import load_clean, load_dev, load_splits, load_test


@pytest.fixture(scope="module")
def df():
    return load_clean().merge(load_splits(), on=C.ID_COL, validate="one_to_one")


def test_every_row_assigned_once(df):
    assert len(df) == 30000
    assert set(df["partition"]) == {"dev", "test"}
    assert (df.loc[df["partition"] == "test", "fold"] == -1).all()
    assert set(df.loc[df["partition"] == "dev", "fold"]) == set(range(C.N_FOLDS))


def test_sizes_and_stratification(df):
    assert abs((df["partition"] == "test").mean() - C.TEST_SIZE) < 0.002
    overall = df[C.TARGET].mean()
    parts = df.groupby(["partition", "fold"])[C.TARGET].mean()
    assert (abs(parts - overall) < 0.005).all()


def test_identical_feature_rows_stay_together(df):
    assert (df.groupby(C.FEATURES)[["partition", "fold"]].nunique() == 1).all().all()


def test_test_set_locked_until_frozen():
    X, y, fold, ids = load_dev()
    assert len(X) == len(y) == len(fold) == len(ids) and list(X.columns) == C.FEATURES
    if not C.FROZEN_FILE.exists():
        with pytest.raises(RuntimeError):
            load_test()
