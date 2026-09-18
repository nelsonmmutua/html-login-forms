# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`html-login-forms` is a Python ML project that detects malicious phishing login forms by analysing the structural signature of a page's HTML DOM — no raw HTML, no text content, no screenshots. The input is a compact tree string (`html_signature`) that encodes element hierarchy only.

**Binary classification:** `LOGIN_FORM_MALICIOUS` vs `NO_FORM`.

**Requires Python 3.14.** Package installation is done with `uv`.

## Common Commands

```bash
# Install all dependencies (runtime + dev) and the package in editable mode
make install   # runs: uv sync --extra dev

# Train both models in parallel (recommended)
make train-all

# Train a single model
make train        # XGBoost
make train-rf     # Random Forest

# Predict on an inline html_signature
make predict SIG="(body(div(form(input)(button))))"

# Batch predict from CSV
make predict-csv INPUT=data/retest/proxy_data_login_form.csv OUTPUT=predictions.csv

# Evaluate both saved models on the retest set
make evaluate

# Retrain on new labeled data
make retrain DATA=path/to/new_data.csv

# Run tests
make test

# Lint
make lint

# View MLflow runs
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

All `make` targets use `uv run`, which automatically uses the project's `.venv`. Run `make install` once after cloning before using any other target.

## Architecture

All source modules live in `htmlloginforms/`:

- **`config.py`** — Central configuration: file paths (`ROOT`, `DATA_DIR`, `MODELS_DIR`, `MLFLOW_DB`), hyperparameters (`XGB_PARAMS`, `RF_PARAMS`), feature column list (`STRUCTURAL_COLS`), and label constants. `ROOT` is anchored two levels up from `__file__` (project root, not `htmlloginforms/`).
- **`features.py`** — Extracts 14 structural features from an `html_signature` string: tag presence flags (`has_form`, `has_input`, etc.), raw counts, `form_density`, `input_per_form`, and `has_credential_pattern`. No sequence or n-gram features.
- **`train.py`** — Trains a single model (XGBoost or Random Forest) using pre-prepared data splits. Logs params and metrics to MLflow. Entry point: `train(model_type, data)`.
- **`train_all.py`** — Trains both models in parallel using `ProcessPoolExecutor`. Initialises the MLflow DB in the main process before forking to avoid SQLite race conditions.
- **`predict.py`** — Runs inference on one or more `html_signature` strings or a CSV file. Validates input via `_validate_signatures()` before loading the model.
- **`evaluate.py`** — Standalone retest evaluation against saved `.pkl` models. Prints accuracy, ROC-AUC, per-class P/R/F1, confusion matrix, and lists FP/FN samples with probabilities.
- **`retrain.py`** — Merges new labeled CSV data with the existing training set, deduplicates on `html_signature`, and retrains the chosen model.

**Typical flow:** `config.py` → `features.py` (extract 14 features) → `train.py` (fit XGBoost/RF) → MLflow (log metrics) → `models/` (save `.pkl`) → `evaluate.py` (retest metrics) → `predict.py` (inference).

## Data

### Schema

| Column | Type | Description |
|--------|------|-------------|
| `url` | string | Crawled page URL |
| `label` | string | `LOGIN_FORM_MALICIOUS` or `NO_FORM` |
| `html_signature` | string | Compact DOM tree, e.g. `(body(div(form(input)(button))))` |

### Files

| Path | Description |
|------|-------------|
| `data/train/train_login_form-2026-08.csv` | Training malicious — semicolon-delimited |
| `data/train/train_no_form-2026-08.csv` | Training no-form — comma-delimited with index column |
| `data/retest/proxy_data_login_form.csv` | Held-out retest malicious |
| `data/retest/proxy_data_no_form.csv` | Held-out retest no-form |

`data/` and `models/` are gitignored. Mount them as Docker volumes at runtime.

## Testing

Tests live in `tests/` and use `pytest`. 35 tests across 5 files:

| File | Covers |
|------|--------|
| `test_config.py` | Constants, feature columns, hyperparameters |
| `test_features.py` | All feature extraction functions |
| `test_modelling.py` | Training, prediction, input validation, batch CSV |
| `test_evaluate.py` | Retest evaluation, metrics, FP/FN DataFrames |
| `test_retrain.py` | Data merge, deduplication, label assignment |

```bash
pytest tests/ -v                            # all tests
pytest tests/test_features.py -v           # single file
pytest tests/ -v --cov=htmlloginforms      # with coverage
```

`pyproject.toml` sets `pythonpath = ["htmlloginforms"]` so imports work without `sys.path` hacks.

## MLflow Tracking

Training metrics are logged to a local SQLite database at `mlflow.db` (gitignored). Experiment name: `login-form-detection`.

Logged per run:
- **Params:** `n_estimators`, `max_depth`, `learning_rate`, `feature_track`
- **Metrics:** `train/test/retest_accuracy`, `train/test/retest_roc_auc`, `retest_mal_precision`, `retest_mal_recall`, `retest_mal_f1`

If `mlflow.db` is out of date after a version upgrade, delete it and retrain:
```bash
rm mlflow.db && make train-all
```

## CI/CD

`.github/workflows/ci.yml` runs on every push and PR to `main`:

1. `actions/setup-python@v5` — Python version from `.python-version`
2. `astral-sh/setup-uv@v5` — installs uv
3. `uv sync --extra dev` — install runtime deps, dev deps, and the package (editable)
4. `ruff check htmlloginforms/ tests/` — lint
5. `pytest tests/ -v --cov=htmlloginforms --cov-report=xml` — tests + coverage

## Key Dependencies

| Package | Role |
|---------|------|
| `xgboost` | XGBoost classifier |
| `scikit-learn` | Random Forest, train/test split, metrics |
| `pandas` | Data loading and feature DataFrame |
| `numpy` / `scipy` | Numerical operations |
| `mlflow-skinny` | Experiment tracking (no UI server) |
| `joblib` | Model serialisation (`.pkl`) |
| `uv` | Fast package installer (replaces pip) |
| `pytest` / `pytest-cov` | Test runner and coverage |
| `ruff` | Linter and formatter |

## Conventional Commits

All commits should follow the [Conventional Commits](https://www.conventionalcommits.org/) spec:

```
<type>(<scope>): <subject>
```

| Type | When to use |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `refactor` | Refactoring with no feature/fix |
| `build` | Dependency or build system changes |
| `test` | Adding or updating tests |
| `docs` | Documentation only |
| `ci` | CI/CD configuration changes |
| `chore` | Tooling, maintenance |
