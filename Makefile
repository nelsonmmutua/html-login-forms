PYTHON := python

.PHONY: install test lint train train-all predict predict-csv retrain evaluate clean

install:
	pip install -r requirements.txt

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m ruff check .

train:
	$(PYTHON) train.py --model xgboost

train-rf:
	$(PYTHON) train.py --model random_forest

train-all:
	$(PYTHON) train_all.py

predict:
	$(PYTHON) predict.py "$(SIG)" --model $(or $(MODEL),models/structural_xgboost.pkl)

predict-csv:
	$(PYTHON) predict.py --input-csv $(INPUT) --output-csv $(or $(OUTPUT),predictions.csv) --model $(or $(MODEL),models/structural_xgboost.pkl)

retrain:
	$(PYTHON) retrain.py $(DATA) --model $(or $(MODEL_TYPE),xgboost)

evaluate:
	@echo "=== XGBoost ==="
	$(PYTHON) evaluate.py --model models/structural_xgboost.pkl
	@echo ""
	@echo "=== Random Forest ==="
	$(PYTHON) evaluate.py --model models/structural_random_forest.pkl

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
