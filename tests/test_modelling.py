"""Tests for train.py and predict.py."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import STRUCTURAL_COLS
from features import structural_features

SIG_MALICIOUS = "(body(div(form(label(input))(label(input))(button))(iframe)(style)))"
SIG_NO_FORM   = "(body(div(header(nav(ul(li(a))(li(a)))))(main(p)(p))))"


def _make_unique_sigs(base: str, n: int) -> list[str]:
    return [base.replace("body", f"body{i}", 1) for i in range(n)]


SAMPLE_TRAIN = pd.DataFrame({
    "html_signature": _make_unique_sigs(SIG_MALICIOUS, 30) + _make_unique_sigs(SIG_NO_FORM, 30),
    "label": ["LOGIN_FORM_MALICIOUS"] * 30 + ["NO_FORM"] * 30,
})


def _make_features(df):
    df = structural_features(df.copy())
    df["binary_label"] = (df["label"] == "LOGIN_FORM_MALICIOUS").astype(int)
    return df


def test_structural_features_row_count():
    assert len(_make_features(SAMPLE_TRAIN)) == len(SAMPLE_TRAIN)


def test_xgboost_trains_and_predicts():
    from xgboost import XGBClassifier
    df  = _make_features(SAMPLE_TRAIN)
    X, y = df[STRUCTURAL_COLS].values, df["binary_label"].values
    clf = XGBClassifier(n_estimators=10, max_depth=2, random_state=42, eval_metric="logloss")
    clf.fit(X, y)
    assert set(clf.predict(X)).issubset({0, 1})


def test_random_forest_trains_and_predicts():
    from sklearn.ensemble import RandomForestClassifier
    df  = _make_features(SAMPLE_TRAIN)
    X, y = df[STRUCTURAL_COLS].values, df["binary_label"].values
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    assert set(clf.predict(X)).issubset({0, 1})


def test_predict_returns_dataframe(tmp_path):
    import joblib
    from sklearn.ensemble import RandomForestClassifier

    from predict import predict

    df  = _make_features(SAMPLE_TRAIN)
    X, y = df[STRUCTURAL_COLS].values, df["binary_label"].values
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    model_path = tmp_path / "model.pkl"
    joblib.dump(clf, model_path)

    results = predict([SIG_MALICIOUS, SIG_NO_FORM], model_path)
    assert list(results.columns) == ["html_signature", "prediction", "malicious_probability"]
    assert len(results) == 2
    assert results["malicious_probability"].between(0, 1).all()


def test_predict_output_labels(tmp_path):
    import joblib
    from sklearn.ensemble import RandomForestClassifier

    from predict import predict

    df  = _make_features(SAMPLE_TRAIN)
    X, y = df[STRUCTURAL_COLS].values, df["binary_label"].values
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X, y)
    model_path = tmp_path / "model.pkl"
    joblib.dump(clf, model_path)

    results = predict([SIG_MALICIOUS, SIG_NO_FORM], model_path)
    assert set(results["prediction"]).issubset({"LOGIN_FORM_MALICIOUS", "NO_FORM"})


def test_predict_raises_on_empty_list(tmp_path):
    import joblib
    from sklearn.ensemble import RandomForestClassifier

    from predict import predict

    df  = _make_features(SAMPLE_TRAIN)
    X, y = df[STRUCTURAL_COLS].values, df["binary_label"].values
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    model_path = tmp_path / "model.pkl"
    joblib.dump(clf, model_path)

    with pytest.raises(ValueError, match="No signatures provided"):
        predict([], model_path)


def test_predict_raises_on_empty_string(tmp_path):
    import joblib
    from sklearn.ensemble import RandomForestClassifier

    from predict import predict

    df  = _make_features(SAMPLE_TRAIN)
    X, y = df[STRUCTURAL_COLS].values, df["binary_label"].values
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    model_path = tmp_path / "model.pkl"
    joblib.dump(clf, model_path)

    with pytest.raises(ValueError, match="empty"):
        predict([""], model_path)


def test_predict_raises_on_mismatched_parens(tmp_path):
    import joblib
    from sklearn.ensemble import RandomForestClassifier

    from predict import predict

    df  = _make_features(SAMPLE_TRAIN)
    X, y = df[STRUCTURAL_COLS].values, df["binary_label"].values
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    model_path = tmp_path / "model.pkl"
    joblib.dump(clf, model_path)

    with pytest.raises(ValueError, match="mismatched parentheses"):
        predict(["(body(div(form)"], model_path)


def test_predict_raises_on_missing_model():
    from predict import predict

    with pytest.raises(FileNotFoundError, match="Model file not found"):
        predict([SIG_MALICIOUS], Path("nonexistent_model.pkl"))


def _save_model(tmp_path) -> Path:
    import joblib
    from sklearn.ensemble import RandomForestClassifier

    df = _make_features(SAMPLE_TRAIN)
    X, y = df[STRUCTURAL_COLS].values, df["binary_label"].values
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    path = tmp_path / "model.pkl"
    joblib.dump(clf, path)
    return path


def test_predict_csv_returns_dataframe(tmp_path):
    from predict import predict_csv

    model_path = _save_model(tmp_path)
    csv_path = tmp_path / "input.csv"
    pd.DataFrame({"html_signature": [SIG_MALICIOUS, SIG_NO_FORM]}).to_csv(csv_path, index=False)

    results = predict_csv(csv_path, model_path)
    assert list(results.columns) == ["html_signature", "prediction", "malicious_probability"]
    assert len(results) == 2


def test_predict_csv_preserves_url_column(tmp_path):
    from predict import predict_csv

    model_path = _save_model(tmp_path)
    csv_path = tmp_path / "input.csv"
    pd.DataFrame({
        "url": ["https://evil.com", "https://legit.com"],
        "html_signature": [SIG_MALICIOUS, SIG_NO_FORM],
    }).to_csv(csv_path, index=False)

    results = predict_csv(csv_path, model_path)
    assert "url" in results.columns
    assert results["url"].tolist() == ["https://evil.com", "https://legit.com"]


def test_predict_csv_writes_output_file(tmp_path):
    from predict import predict_csv

    model_path = _save_model(tmp_path)
    csv_path    = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    pd.DataFrame({"html_signature": [SIG_MALICIOUS, SIG_NO_FORM]}).to_csv(csv_path, index=False)

    predict_csv(csv_path, model_path, output_path)
    assert output_path.exists()
    saved = pd.read_csv(output_path)
    assert len(saved) == 2


def test_predict_csv_raises_on_missing_column(tmp_path):
    from predict import predict_csv

    model_path = _save_model(tmp_path)
    csv_path = tmp_path / "bad_input.csv"
    pd.DataFrame({"url": ["https://example.com"]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="html_signature"):
        predict_csv(csv_path, model_path)
