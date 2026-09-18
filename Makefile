.PHONY: install test lint train train-rf train-all predict predict-csv retrain evaluate clean

install:
	uv sync --extra dev

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check htmlloginforms/ tests/

train:
	uv run train --model xgboost

train-rf:
	uv run train --model random_forest

train-all:
	uv run train-all

predict:
	uv run predict "$(SIG)" --model $(or $(MODEL),models/structural_xgboost.pkl)

predict-csv:
	uv run predict --input-csv $(INPUT) --output-csv $(or $(OUTPUT),predictions.csv) --model $(or $(MODEL),models/structural_xgboost.pkl)

retrain:
	uv run retrain $(DATA) --model $(or $(MODEL_TYPE),xgboost)

evaluate:
	@echo "=== XGBoost ==="
	uv run evaluate --model models/structural_xgboost.pkl
	@echo ""
	@echo "=== Random Forest ==="
	uv run evaluate --model models/structural_random_forest.pkl

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
