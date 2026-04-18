import sys
import logging
import mlflow
import json
import pickle
import os
from model_trainings import (
  train_random_forest_classifier,
  train_random_forest_classifier_v2,
  train_logistic_regression_classifier,
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REQUIRED_ENV_VARS = ["MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"]

DATA_FILE = "data/taxi-rides-training-data.parquet"


def check_env_vars() -> None:
  missing = [var for var in REQUIRED_ENV_VARS if not os.environ.get(var)]
  if missing:
    logger.error(f"Missing required environment variables: {', '.join(missing)}")
    sys.exit(1)

def train_model(model_type: str):
  check_env_vars()

  logger.info("Training outlier detection classifier")

  valid_model_types = ['random_forest', 'random_forest_v2', 'logistic_regression']
  if model_type not in valid_model_types:
    raise ValueError(f"Unknown model_type '{model_type}'. Valid options: {', '.join(valid_model_types)}")

  mlflow.set_experiment(model_type)
  # Turn on auto logging. See https://mlflow.org/docs/latest/ml/tracking/autolog
  mlflow.autolog()
  with mlflow.start_run():
    if model_type == 'random_forest':
      model, metadata = train_random_forest_classifier(DATA_FILE)
    elif model_type == 'random_forest_v2':
      model, metadata = train_random_forest_classifier_v2(DATA_FILE)
    elif model_type == 'logistic_regression':
      model, metadata = train_logistic_regression_classifier(DATA_FILE)
    # log some custom metrics
    for false_key, false_value in metadata["False"].items():
      mlflow.log_metric(f"False_{false_key}", false_value)
    for false_key, false_value in metadata["True"].items():
      mlflow.log_metric(f"True_{false_key}", false_value)
    logger.info("Model training completed")

    os.makedirs("models", exist_ok=True)

    model_output_file = f"models/{model_type}.pkl"
    logger.info(f"Storing model to: {model_output_file}")
    with open(model_output_file, "wb") as f:
      pickle.dump(model, f)
    mlflow.log_artifact(model_output_file)

    metadata_output_file = f"models/{model_type}.metadata.json"
    logger.info(f"Writing metadata to: {metadata_output_file}")
    with open(metadata_output_file, 'w') as metadata_file:
      json.dump(metadata, metadata_file, indent=4)
    mlflow.log_artifact(metadata_output_file)


if __name__ == "__main__":
  model_type = sys.argv[1]  # random_forest, random_forest_v2, logistic_regression
  train_model(model_type)


