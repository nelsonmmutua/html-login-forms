"""Evaluate a saved structural model against the retest set without retraining."""
import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from htmlloginforms.config import (
    CLASSES,
    RETEST_LOGIN_FORM,
    RETEST_NO_FORM,
    STRUCTURAL_COLS,
    STRUCTURAL_XGB_MODEL,
)
from htmlloginforms.features import structural_features

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def load_retest(login_form_path: Path, no_form_path: Path) -> pd.DataFrame:
    lf = pd.read_csv(login_form_path, index_col=0)
    nf = pd.read_csv(no_form_path, index_col=0)
    df = pd.concat([lf, nf], ignore_index=True)
    df.drop(columns=["schreenshot_path"], errors="ignore", inplace=True)
    df.dropna(subset=["html_signature"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["binary_label"] = df["label"].str.contains("LOGIN_FORM", case=False, na=False).astype(int)
    return df


def evaluate(model_path: Path, login_form_path: Path, no_form_path: Path) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)
    log.info("Model loaded: %s", model_path.name)

    df = load_retest(login_form_path, no_form_path)
    df = structural_features(df)
    X = df[STRUCTURAL_COLS].values
    y = df["binary_label"].values

    log.info("Retest set: %d samples (malicious=%d, no-form=%d)", len(y), y.sum(), (y == 0).sum())

    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    acc  = accuracy_score(y, preds)
    auc  = roc_auc_score(y, probs)
    prec, rec, f1, sup = precision_recall_fscore_support(y, preds)
    cm   = confusion_matrix(y, preds)

    tn, fp, fn, tp = cm.ravel()

    log.info("\n── Retest Metrics ──────────────────────────────────")
    log.info("  Accuracy : %.1f%%   ROC-AUC : %.1f%%", acc * 100, auc * 100)
    log.info("")
    for i, cls in enumerate(CLASSES):
        log.info("  %-25s  P=%.1f%%  R=%.1f%%  F1=%.1f%%  n=%d",
                 cls, prec[i] * 100, rec[i] * 100, f1[i] * 100, sup[i])
    log.info("")
    log.info("── Confusion Matrix ────────────────────────────────")
    log.info("  TP (correct malicious)   : %d", tp)
    log.info("  TN (correct no-form)     : %d", tn)
    log.info("  FP (no-form → malicious) : %d", fp)
    log.info("  FN (malicious → no-form) : %d", fn)

    fp_mask = (y == 0) & (preds == 1)
    fn_mask = (y == 1) & (preds == 0)

    fp_df = df[fp_mask][["html_signature", "label"]].copy()
    fn_df = df[fn_mask][["html_signature", "label"]].copy()
    fp_df["malicious_probability"] = probs[fp_mask]
    fn_df["malicious_probability"] = probs[fn_mask]

    if len(fp_df):
        log.info("\n── False Positives (%d) ─────────────────────────────", len(fp_df))
        log.info(fp_df.to_string(index=False))

    if len(fn_df):
        log.info("\n── False Negatives (%d) ─────────────────────────────", len(fn_df))
        log.info(fn_df.to_string(index=False))

    return {
        "accuracy": acc,
        "roc_auc":  auc,
        "precision_malicious": float(prec[1]),
        "recall_malicious":    float(rec[1]),
        "f1_malicious":        float(f1[1]),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "fp_samples": fp_df,
        "fn_samples": fn_df,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved structural model on the retest set.")
    parser.add_argument("--model",          type=Path, default=STRUCTURAL_XGB_MODEL)
    parser.add_argument("--retest-login",   type=Path, default=RETEST_LOGIN_FORM)
    parser.add_argument("--retest-no-form", type=Path, default=RETEST_NO_FORM)
    args = parser.parse_args()

    try:
        evaluate(args.model, args.retest_login, args.retest_no_form)
    except (FileNotFoundError, ValueError) as e:
        log.error("Error: %s", e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
