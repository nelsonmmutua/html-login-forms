"""Central configuration — paths, hyperparameters, and constants."""
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR   = ROOT / "data"
TRAIN_DIR  = DATA_DIR / "train"
RETEST_DIR = DATA_DIR / "retest"
MODELS_DIR = ROOT / "models"
MLFLOW_DB = ROOT / "mlflow.db"

TRAIN_LOGIN_FORM  = TRAIN_DIR  / "train_login_form-2026-08.csv"
TRAIN_NO_FORM     = TRAIN_DIR  / "train_no_form-2026-08.csv"
RETEST_LOGIN_FORM = RETEST_DIR / "proxy_data_login_form.csv"
RETEST_NO_FORM    = RETEST_DIR / "proxy_data_no_form.csv"

STRUCTURAL_XGB_MODEL = MODELS_DIR / "structural_xgboost.pkl"
STRUCTURAL_RF_MODEL  = MODELS_DIR / "structural_random_forest.pkl"

# ── Labels ────────────────────────────────────────────────────────────────────
MALICIOUS_LABEL = "LOGIN_FORM_MALICIOUS"
CLASSES = ["NO_FORM", "LOGIN_FORM_MALICIOUS"]

# ── Feature columns ───────────────────────────────────────────────────────────
CREDENTIAL_GROUP = ["form", "input", "label", "iframe", "style"]

STRUCTURAL_COLS = (
    [f"has_{t}" for t in CREDENTIAL_GROUP]
    + ["input_count", "label_count", "form_count", "button_count"]
    + ["signature_length", "total_tag_count", "form_density", "input_per_form"]
    + ["has_credential_pattern"]
)

# ── Training ──────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.2

# ── XGBoost hyperparameters ───────────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators":     200,
    "max_depth":        4,
    "learning_rate":    0.1,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "eval_metric":      "logloss",
    "random_state":     RANDOM_STATE,
    "n_jobs":           -1,
}

# ── Random Forest hyperparameters ─────────────────────────────────────────────
RF_PARAMS = {
    "n_estimators":     200,
    "max_depth":        None,
    "min_samples_leaf": 2,
    "max_features":     "sqrt",
    "class_weight":     "balanced",
    "random_state":     RANDOM_STATE,
    "n_jobs":           -1,
}

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_EXPERIMENT = "login-form-detection"
