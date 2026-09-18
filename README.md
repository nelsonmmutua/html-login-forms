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
├── config.py         — paths, hyperparameters, feature columns
├── features.py       — structural feature extraction
├── train.py          — train a single model (XGBoost or RF)
├── train_all.py      — train both models in parallel
├── predict.py        — run inference on html_signature strings
├── retrain.py        — retrain on new labeled data
├── tests/            — pytest test suite
├── data/
│   ├── train/        — training CSVs
│   └── retest/       — held-out validation CSVs
└── models/           — saved model artifacts (populated after training)
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Usage

**Train both models in parallel (recommended):**
```bash
python train_all.py
# or
make train-all
```

**Train a single model:**
```bash
python train.py --model xgboost
python train.py --model random_forest
```

**Predict on an html_signature:**
```bash
python predict.py "(body(div(form(input)(button))))"
python predict.py "(body(div(form(input)(button))))" --model models/structural_random_forest.pkl
```

**Retrain on new labeled data:**
```bash
python retrain.py path/to/new_data.csv --model xgboost
```

**Run tests:**
```bash
make test
# or
pytest tests/ -v
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
