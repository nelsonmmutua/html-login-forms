"""Train XGBoost and Random Forest structural models in parallel."""
import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

from config import MLFLOW_DB, MLFLOW_EXPERIMENT, RETEST_LOGIN_FORM, RETEST_NO_FORM
from train import prepare_data, train

MODEL_TYPES = ["xgboost", "random_forest"]


def _worker(args: tuple) -> str:
    model_type, data = args
    train(model_type, data)
    return model_type


def train_all(retest_login: Path, retest_no_form: Path):
    # Initialise DB and experiment in the main process so workers don't race to create tables.
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    log.info("Loading and featurising data once...")
    data = prepare_data(retest_login, retest_no_form)

    log.info("Launching %d training jobs in parallel...", len(MODEL_TYPES))
    with ProcessPoolExecutor(max_workers=len(MODEL_TYPES)) as pool:
        futures = {pool.submit(_worker, (mt, data)): mt for mt in MODEL_TYPES}
        for future in as_completed(futures):
            model_type = futures[future]
            try:
                future.result()
                log.info("%s complete.", model_type)
            except Exception as exc:
                log.error("%s failed: %s", model_type, exc)

    log.info("All models trained.")


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost and Random Forest in parallel.")
    parser.add_argument("--retest-login",   type=Path, default=RETEST_LOGIN_FORM)
    parser.add_argument("--retest-no-form", type=Path, default=RETEST_NO_FORM)
    args = parser.parse_args()
    train_all(args.retest_login, args.retest_no_form)


if __name__ == "__main__":
    main()
