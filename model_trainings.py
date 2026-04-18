from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
import pandas
import pandas as pd
import numpy as np


def train_random_forest_classifier(data_file: str) -> tuple[RandomForestClassifier, dict]:
  data = pandas.read_parquet(data_file)
  X = data[['ride_time', 'trip_distance']]
  y = data['outlier']

  # As the dataset is imbalanced, stratify=y will ensure that the split maintains the proportion of classes
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42, stratify=y
  )

  # Use class_weight='balanced' to handle class imbalance
  clf = RandomForestClassifier(class_weight='balanced', random_state=42)
  clf.fit(X_train, y_train)

  y_pred = clf.predict(X_test)
  report = classification_report(y_test, y_pred, digits=4, output_dict=True)

  return (clf, report)


def train_random_forest_classifier_v2(data_file: str) -> tuple[RandomForestClassifier, dict]:
  data = pandas.read_parquet(data_file)
  X = data[['ride_time', 'trip_distance']]
  y = data['outlier']

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

  y_pred = clf.predict(X_test)
  report = classification_report(y_test, y_pred, digits=4, output_dict=True)

  return (clf, report)


def train_logistic_regression_classifier(data_file: str) -> tuple[LogisticRegression, dict]:
  data = pandas.read_parquet(data_file)
  X = data[['ride_time', 'trip_distance']]
  y = data['outlier']

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

  X = data[['ride_time', 'trip_distance']]

  data['outlier'] = model.predict(X)

  return data[data['outlier'] == 1]
