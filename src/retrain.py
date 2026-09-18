"""Retrain structural models on new labeled data appended to the existing training set."""
import argparse
import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from config import (
    MLFLOW_DB,
    MLFLOW_EXPERIMENT,
    MODELS_DIR,
    RANDOM_STATE,
    RF_PARAMS,
    STRUCTURAL_COLS,
    TEST_SIZE,
    TRAIN_LOGIN_FORM,
    TRAIN_NO_FORM,
    XGB_PARAMS,
)
from features import structural_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_and_merge(new_data_path: Path) -> pd.DataFrame:
    lf = pd.read_csv(TRAIN_LOGIN_FORM, sep=";")
    nf = pd.read_csv(TRAIN_NO_FORM, index_col=0)
    nf.drop(columns=["schreenshot_path"], errors="ignore", inplace=True)
    new_df = pd.read_csv(new_data_path)
    log.info("New data: %d rows", len(new_df))
    df = pd.concat([lf, nf, new_df], ignore_index=True)
    df.dropna(subset=["html_signature"], inplace=True)
    df = df.drop_duplicates(subset=["html_signature"]).reset_index(drop=True)
    df["binary_label"] = df["label"].str.contains("LOGIN_FORM", case=False, na=False).astype(int)
    log.info("Merged (deduplicated): %d rows | Malicious: %d | No-Form: %d",
             len(df), df["binary_label"].sum(), (df["binary_label"] == 0).sum())
    return df


def retrain(model_type: str, new_data_path: Path, output_path: Path | None):
    MODELS_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    df = load_and_merge(new_data_path)
    df = structural_features(df)
    X  = df[STRUCTURAL_COLS].values
    y  = df["binary_label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    if model_type == "xgboost":
        model     = XGBClassifier(**XGB_PARAMS)
        run_name  = "retrain-structural-xgboost"
        save_path = output_path or MODELS_DIR / "structural_xgboost.pkl"
    else:
        model     = RandomForestClassifier(**RF_PARAMS)
        run_name  = "retrain-structural-random-forest"
        save_path = output_path or MODELS_DIR / "structural_random_forest.pkl"

    with mlflow.start_run(run_name=run_name):
        model.fit(X_train, y_train)

        test_pred = model.predict(X_test)
        test_prob = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, test_pred)
        auc = roc_auc_score(y_test, test_prob)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, test_pred)

        log.info("Retrain Test — Accuracy: %.1f%%  AUC: %.1f%%", acc * 100, auc * 100)
        log.info("  MAL  P=%.1f%%  R=%.1f%%  F1=%.1f%%", prec[1] * 100, rec[1] * 100, f1[1] * 100)

        mlflow.log_params({"model_type": model_type, "new_data": str(new_data_path)})
        mlflow.log_metrics({
            "retrain_test_accuracy": acc,
            "retrain_test_roc_auc":  auc,
            "retrain_mal_precision": float(prec[1]),
            "retrain_mal_recall":    float(rec[1]),
            "retrain_mal_f1":        float(f1[1]),
        })

        joblib.dump(model, save_path)
        log.info("Saved → %s", save_path)


def main():
    parser = argparse.ArgumentParser(description="Retrain a structural model on new labeled data.")
    parser.add_argument("new_data", type=Path, help="CSV with new labeled rows.")
    parser.add_argument("--model", choices=["xgboost", "random_forest"], default="xgboost")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    retrain(args.model, args.new_data, args.output)


if __name__ == "__main__":
    main()
