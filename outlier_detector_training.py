from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.linear_model import LogisticRegression
import sys
import logging
import mlflow
import pandas
import json
import pickle
import os

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(model_type: str):
  data_file = "data/taxi-rides-training-data.parquet"
  logger.info(f"Processing taxi ride data from: {data_file}")
  data = pandas.read_parquet(data_file)

  logger.info("Training outlier detection classifier")

  valid_model_types = ['random_forest', 'random_forest_v2', 'logistic_regression']
  if model_type not in valid_model_types:
    raise ValueError(f"Unknown model_type '{model_type}'. Valid options: {', '.join(valid_model_types)}")

  # You need to set these environment variables in order to track you experiments at dagshub
  # export MLFLOW_TRACKING_URI=https://dagshub.com/peter.rietzler.privat/ais-dev2il-mlops-taxi-rides-v2.mlflow
  # export MLFLOW_TRACKING_USERNAME=dagshub username
  # export MLFLOW_TRACKING_PASSWORD=dagshub access token

  mlflow.set_experiment(model_type)
  # Turn on auto logging. See https://mlflow.org/docs/latest/ml/tracking/autolog
  mlflow.autolog()
  with mlflow.start_run():
    if model_type == 'random_forest':
      model, metadata = train_random_forest_classifier(data)
    elif model_type == 'random_forest_v2':
      model, metadata = train_random_forest_classifier_v2(data)
    elif model_type == 'logistic_regression':
      model, metadata = train_logistic_regression_classifier(data)
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


def train_random_forest_classifier(labeled_taxi_rides_data: pd.DataFrame) -> \
    tuple[RandomForestClassifier, dict]:
  # Features and target
  X = labeled_taxi_rides_data[['ride_time', 'trip_distance']]
  y = labeled_taxi_rides_data['outlier']

  # Split data for training and testing
  # As the dataset is imbalanced, stratify=y will ensure that the split maintains the proportion of classes
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42, stratify=y
  )

  # Use class_weight='balanced' to handle class imbalance
  clf = RandomForestClassifier(class_weight='balanced', random_state=42)
  clf.fit(X_train, y_train)

  # Predict on test set
  y_pred = clf.predict(X_test)
  report = classification_report(y_test, y_pred, digits=4, output_dict=True)

  return (clf, report)


def train_random_forest_classifier_v2(labeled_taxi_rides_data: pd.DataFrame) -> \
    tuple[RandomForestClassifier, dict]:
  # Features and target
  X = labeled_taxi_rides_data[['ride_time', 'trip_distance']]
  y = labeled_taxi_rides_data['outlier']

  # Split data for training and testing
  # As the dataset is imbalanced, stratify=y will ensure that the split maintains the proportion of classes
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42, stratify=y
  )

  clf = Pipeline([
    ('add_avg_speed', AverageSpeedAdder()),
    # Use class_weight='balanced' to handle class imbalance
    ('rf', RandomForestClassifier(class_weight='balanced', random_state=42))
  ])
  clf.fit(X_train, y_train)

  # Predict on test set
  y_pred = clf.predict(X_test)
  report = classification_report(y_test, y_pred, digits=4, output_dict=True)

  return (clf, report)


def train_logistic_regression_classifier(
    labeled_taxi_rides_data: pd.DataFrame) -> tuple[LogisticRegression, dict]:
  # Features and target
  X = labeled_taxi_rides_data[['ride_time', 'trip_distance']]
  y = labeled_taxi_rides_data['outlier']

  # Split data for training and testing
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42, stratify=y
  )

  clf = LogisticRegression(class_weight='balanced', random_state=42,
                           max_iter=1000)
  clf.fit(X_train, y_train)

  y_pred = clf.predict(X_test)
  report = classification_report(y_test, y_pred, digits=4, output_dict=True)

  return (clf, report)


class AverageSpeedAdder(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None):
    return self

  def transform(self, X):
    X = X.copy()
    X['average_speed'] = np.where(X['ride_time'] > 0,
                                  X['trip_distance'] / (X['ride_time'] / 3600),
                                  0)
    return X


# TODO - move
def detect_outliers(taxi_rides_data: pd.DataFrame, model) -> pd.DataFrame:
  raw_data = taxi_rides_data

  data = pd.DataFrame()
  raw_data['tpep_pickup_datetime'] = pd.to_datetime(
      raw_data['tpep_pickup_datetime'])
  raw_data['tpep_dropoff_datetime'] = pd.to_datetime(
      raw_data['tpep_dropoff_datetime'])
  data['ride_time'] = (raw_data['tpep_dropoff_datetime'] - raw_data[
    'tpep_pickup_datetime']).dt.total_seconds()
  data['date'] = raw_data['tpep_pickup_datetime'].dt.date
  data['ride_id'] = raw_data.index
  data['trip_distance'] = raw_data['trip_distance']

  # Features for prediction
  X = data[['ride_time', 'trip_distance']]

  # Predict outliers
  data['outlier'] = model.predict(X)

  # Return only the rows classified as outliers
  return data[data['outlier'] == 1]


if __name__ == "__main__":
  model_type = sys.argv[1]  # random_forest, random_forest_v2, logistic_regression
  train_model(model_type)


