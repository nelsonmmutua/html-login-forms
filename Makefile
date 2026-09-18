PYTHON     := python
PYTHONPATH := htmlloginforms

.PHONY: install test lint train train-rf train-all predict predict-csv retrain evaluate clean

install:
	pip install -r requirements.txt

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m ruff check htmlloginforms/ tests/

train:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) htmlloginforms/train.py --model xgboost

train-rf:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) htmlloginforms/train.py --model random_forest

train-all:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) htmlloginforms/train_all.py

predict:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) htmlloginforms/predict.py "$(SIG)" --model $(or $(MODEL),models/structural_xgboost.pkl)

predict-csv:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) htmlloginforms/predict.py --input-csv $(INPUT) --output-csv $(or $(OUTPUT),predictions.csv) --model $(or $(MODEL),models/structural_xgboost.pkl)

retrain:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) htmlloginforms/retrain.py $(DATA) --model $(or $(MODEL_TYPE),xgboost)

evaluate:
	@echo "=== XGBoost ==="
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) htmlloginforms/evaluate.py --model models/structural_xgboost.pkl
	@echo ""
	@echo "=== Random Forest ==="
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) htmlloginforms/evaluate.py --model models/structural_random_forest.pkl

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
