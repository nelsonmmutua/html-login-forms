<p align="center">
  <img src="banner.svg" alt="html-login-forms banner" width="100%"/>
</p>

# html-login-forms

Detects malicious phishing login forms by analysing the structural signature of a page's HTML DOM. No raw HTML, no text content, no screenshots — only the compact tree string (`html_signature`) that encodes element hierarchy.

Binary classification: `LOGIN_FORM_MALICIOUS` vs `NO_FORM`.

---

## How it works

Each page's DOM is represented as a compact tree string, e.g.:

```
(body(div(form(label(input))(button))(iframe)(style)))
```

14 structural features are extracted (tag presence flags, counts, form density, etc.) and fed into either an XGBoost or Random Forest classifier.

---

## Project structure

```
html-login-forms/
├── htmlloginforms/
│   ├── config.py         — paths, hyperparameters, feature columns
│   ├── features.py       — structural feature extraction
│   ├── train.py          — train a single model (XGBoost or RF)
│   ├── train_all.py      — train both models in parallel
│   ├── predict.py        — run inference on html_signature strings
│   ├── evaluate.py       — evaluate a saved model on the retest set
│   └── retrain.py        — retrain on new labeled data
├── tests/
│   ├── test_config.py    — config constants and hyperparameters
│   ├── test_features.py  — structural feature extraction
│   ├── test_modelling.py — training, prediction, input validation, batch CSV
│   ├── test_evaluate.py  — retest evaluation and metrics
│   └── test_retrain.py   — data merge, deduplication, label assignment
├── data/
│   ├── train/            — training CSVs
│   └── retest/           — held-out validation CSVs
├── models/               — saved model artifacts (populated after training)
├── Makefile              — CLI shortcuts for all operations
├── Dockerfile
└── pyproject.toml
```

---

## Running locally

### 1. Prerequisites

- Python 3.14
- `uv` (fast package installer)

```bash
pip install uv
```

### 2. Clone the repo

```bash
git clone https://github.com/nelsonmmutua/html-login-forms.git
cd html-login-forms
```

### 3. Install dependencies

```bash
uv sync --extra dev
```

### 4. Add your data

Place training and retest CSVs in the expected locations:

```
data/
├── train/
│   ├── train_login_form-2026-08.csv   — semicolon-delimited
│   └── train_no_form-2026-08.csv      — comma-delimited
└── retest/
    ├── proxy_data_login_form.csv
    └── proxy_data_no_form.csv
```

### 5. Train the models

```bash
make train-all
```

This trains XGBoost and Random Forest in parallel and saves both models to `models/`.

### 6. Run predictions

```bash
# Single signature
make predict SIG="(body(div(form(label(input))(button))(iframe)(style)))"

# Batch from CSV
make predict-csv INPUT=data/retest/proxy_data_login_form.csv OUTPUT=predictions.csv
```

### 7. Evaluate on the retest set

```bash
make evaluate
```

### 8. View MLflow runs

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Open **http://localhost:5001** → click **login-form-detection** → **Runs** tab.

### 9. Run tests

```bash
make test
```

---

## Usage

**Train both models in parallel (recommended):**
```bash
make train-all
```

**Train a single model:**
```bash
make train        # XGBoost
make train-rf     # Random Forest
```

**Predict on an html_signature:**
```bash
make predict SIG="(body(div(form(input)(button))))"
make predict SIG="(body(div(form(input)(button))))" MODEL=models/structural_random_forest.pkl
```

**Batch predict from CSV:**
```bash
make predict-csv INPUT=data/retest/proxy_data_login_form.csv OUTPUT=predictions.csv
```

**Evaluate saved models on retest set:**
```bash
make evaluate
```

**Retrain on new labeled data:**
```bash
make retrain DATA=path/to/new_data.csv
```

**Run tests:**
```bash
make test
```

---

## Data schema

| Column | Description |
|--------|-------------|
| `url` | Crawled page URL |
| `label` | `LOGIN_FORM_MALICIOUS` or `NO_FORM` |
| `html_signature` | Compact DOM tree string |

---

## Docker

```bash
docker build -t html-login-forms .
docker run -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models html-login-forms
```

---

## MLflow

Training metrics are logged to a local SQLite database (`mlflow.db`). View runs with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
