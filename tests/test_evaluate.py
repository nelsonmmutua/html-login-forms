"""Tests for evaluate.py."""
from pathlib import Path
from unittest.mock import patch

import joblib
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from config import STRUCTURAL_COLS
from features import structural_features

SIG_MALICIOUS = "(body(div(form(label(input))(label(input))(button))(iframe)(style)))"
SIG_NO_FORM   = "(body(div(header(nav(ul(li(a))(li(a)))))(main(p)(p))))"


def _make_unique_sigs(base: str, n: int) -> list[str]:
    return [base.replace("body", f"body{i}", 1) for i in range(n)]


def _trained_model(tmp_path) -> Path:
    df = pd.DataFrame({
        "html_signature": _make_unique_sigs(SIG_MALICIOUS, 30) + _make_unique_sigs(SIG_NO_FORM, 30),
        "label": ["LOGIN_FORM_MALICIOUS"] * 30 + ["NO_FORM"] * 30,
    })
    df = structural_features(df)
    df["binary_label"] = (df["label"] == "LOGIN_FORM_MALICIOUS").astype(int)
    X, y = df[STRUCTURAL_COLS].values, df["binary_label"].values
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    path = tmp_path / "model.pkl"
    joblib.dump(clf, path)
    return path


def _retest_csvs(tmp_path) -> tuple[Path, Path]:
    lf = pd.DataFrame({
        "html_signature": _make_unique_sigs(SIG_MALICIOUS, 10),
        "label": ["TEST_LOGIN_FORM"] * 10,
    })
    nf = pd.DataFrame({
        "html_signature": _make_unique_sigs(SIG_NO_FORM, 10),
        "label": ["TEST_NO_FORM"] * 10,
    })
    lf_path = tmp_path / "retest_lf.csv"
    nf_path = tmp_path / "retest_nf.csv"
    lf.to_csv(lf_path)
    nf.to_csv(nf_path)
    return lf_path, nf_path


def test_evaluate_returns_metrics(tmp_path):
    from evaluate import evaluate

    model_path = _trained_model(tmp_path)
    lf_path, nf_path = _retest_csvs(tmp_path)

    result = evaluate(model_path, lf_path, nf_path)

    assert "accuracy" in result
    assert "roc_auc" in result
    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["roc_auc"] <= 1.0


def test_evaluate_confusion_matrix_sums(tmp_path):
    from evaluate import evaluate

    model_path = _trained_model(tmp_path)
    lf_path, nf_path = _retest_csvs(tmp_path)

    result = evaluate(model_path, lf_path, nf_path)

    total = result["tp"] + result["tn"] + result["fp"] + result["fn"]
    assert total == 20


def test_evaluate_fp_fn_are_dataframes(tmp_path):
    from evaluate import evaluate

    model_path = _trained_model(tmp_path)
    lf_path, nf_path = _retest_csvs(tmp_path)

    result = evaluate(model_path, lf_path, nf_path)

    assert isinstance(result["fp_samples"], pd.DataFrame)
    assert isinstance(result["fn_samples"], pd.DataFrame)


def test_evaluate_raises_on_missing_model(tmp_path):
    from evaluate import evaluate

    lf_path, nf_path = _retest_csvs(tmp_path)

    with pytest.raises(FileNotFoundError, match="Model file not found"):
        evaluate(tmp_path / "missing.pkl", lf_path, nf_path)
