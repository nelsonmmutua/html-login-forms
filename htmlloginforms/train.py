"""Train structural XGBoost and Random Forest models."""
import argparse
import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from htmlloginforms.config import (
    CLASSES,
    MALICIOUS_LABEL,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    RANDOM_STATE,
    RETEST_LOGIN_FORM,
    RETEST_NO_FORM,
    RF_PARAMS,
    ROOT,
    STRUCTURAL_COLS,
    TEST_SIZE,
    TRAIN_LOGIN_FORM,
    TRAIN_NO_FORM,
    XGB_PARAMS,
)
from htmlloginforms.features import structural_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_training_data() -> pd.DataFrame:
    lf = pd.read_csv(TRAIN_LOGIN_FORM, sep=";")
    nf = pd.read_csv(TRAIN_NO_FORM, index_col=0)
    nf.drop(columns=["schreenshot_path"], errors="ignore", inplace=True)
    lf.dropna(subset=["html_signature"], inplace=True)
    nf.dropna(subset=["html_signature"], inplace=True)
    df = pd.concat([lf, nf], ignore_index=True)
    df = df.drop_duplicates(subset=["html_signature"]).reset_index(drop=True)
    df["binary_label"] = (df["label"] == MALICIOUS_LABEL).astype(int)
    return df


def load_retest_data(login_form_path: Path, no_form_path: Path) -> pd.DataFrame:
    lf_rt = pd.read_csv(login_form_path, index_col=0)
    nf_rt = pd.read_csv(no_form_path, index_col=0)
    df_rt = pd.concat([lf_rt, nf_rt], ignore_index=True)
    df_rt.drop(columns=["schreenshot_path"], errors="ignore", inplace=True)
    df_rt.dropna(subset=["html_signature"], inplace=True)
    df_rt.reset_index(drop=True, inplace=True)
    df_rt["binary_label"] = df_rt["label"].str.contains("LOGIN_FORM", case=False, na=False).astype(int)
    return df_rt


def prepare_data(retest_login: Path, retest_no_form: Path) -> dict:
    """Load, featurise, and split data once — shared across both models."""
    df = load_training_data()
    df = structural_features(df)
    X  = df[STRUCTURAL_COLS].values
    y  = df["binary_label"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    df_rt = load_retest_data(retest_login, retest_no_form)
    df_rt = structural_features(df_rt)
    X_rt  = df_rt[STRUCTURAL_COLS].values
    y_rt  = df_rt["binary_label"].values
    log.info(
        "Train: %d | Test: %d | Retest: %d (mal=%d, nf=%d)",
        len(X_train), len(X_test), len(X_rt), y_rt.sum(), (y_rt == 0).sum(),
    )
    return dict(X_train=X_train, X_test=X_test, y_train=y_train,
                y_test=y_test, X_rt=X_rt, y_rt=y_rt)


def evaluate(model, X: np.ndarray, y: np.ndarray, label: str) -> tuple[float, float]:
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]
    acc  = accuracy_score(y, pred)
    auc  = roc_auc_score(y, prob)
    prec, rec, f1, sup = precision_recall_fscore_support(y, pred)
    log.info("%s", label)
    log.info("  Accuracy : %.1f%%   ROC-AUC : %.1f%%", acc * 100, auc * 100)
    for i, cls in enumerate(CLASSES):
        log.info("  %-25s  P=%.1f%%  R=%.1f%%  F1=%.1f%%  n=%d",
                 cls, prec[i] * 100, rec[i] * 100, f1[i] * 100, sup[i])
    return acc, auc


def train(model_type: str, data: dict):
    """Train one model type using pre-prepared data splits."""
    MODELS_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    X_rt, y_rt      = data["X_rt"],    data["y_rt"]

    if model_type == "xgboost":
        model     = XGBClassifier(**XGB_PARAMS)
        run_name  = "structural-xgboost"
        save_path = MODELS_DIR / "structural_xgboost.pkl"
        params    = {**XGB_PARAMS, "feature_track": "structural_14features"}
    else:
        model     = RandomForestClassifier(**RF_PARAMS)
        run_name  = "structural-random-forest"
        save_path = MODELS_DIR / "structural_random_forest.pkl"
        params    = {k: str(v) for k, v in RF_PARAMS.items()}
        params["feature_track"] = "structural_14features"

    with mlflow.start_run(run_name=run_name):
        model.fit(X_train, y_train)
        log.info("[%s] trained.", model_type)

        train_acc,  train_auc  = evaluate(model, X_train, y_train, f"[{model_type}] Train")
        test_acc,   test_auc   = evaluate(model, X_test,  y_test,  f"[{model_type}] Test")
        retest_acc, retest_auc = evaluate(model, X_rt,    y_rt,    f"[{model_type}] Retest")

        prec_rt, rec_rt, f1_rt, _ = precision_recall_fscore_support(y_rt, model.predict(X_rt))
        mlflow.log_params(params)
        mlflow.log_metrics({
            "train_accuracy":       train_acc,
            "test_accuracy":        test_acc,
            "retest_accuracy":      retest_acc,
            "train_roc_auc":        train_auc,
            "test_roc_auc":         test_auc,
            "retest_roc_auc":       retest_auc,
            "retest_mal_precision": float(prec_rt[1]),
            "retest_mal_recall":    float(rec_rt[1]),
            "retest_mal_f1":        float(f1_rt[1]),
        })

        joblib.dump(model, save_path)
        if model_type == "xgboost":
            mlflow.xgboost.log_model(model, name="model")
        else:
            mlflow.sklearn.log_model(
                model,
                name="model",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
            )
        log.info("[%s] Saved → %s", model_type, save_path)


def main():
    parser = argparse.ArgumentParser(description="Train a single structural model.")
    parser.add_argument("--model", choices=["xgboost", "random_forest"], default="xgboost")
    parser.add_argument("--retest-login",   type=Path, default=RETEST_LOGIN_FORM)
    parser.add_argument("--retest-no-form", type=Path, default=RETEST_NO_FORM)
    args = parser.parse_args()
    data = prepare_data(args.retest_login, args.retest_no_form)
    train(args.model, data)


if __name__ == "__main__":
    main()
