import pytest
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ml.model import compute_model_metrics, train_model

def test_train_model():
    """
    Test that train_model returns a RandomForestClassifier.
    """
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)

def test_compute_model_metrics():
    """
    Test that compute_model_metrics returns the expected values.
    """
    y = np.array([1, 1, 0, 0])
    preds = np.array([1, 0, 0, 0])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == 1.0
    assert recall == 0.5
    assert fbeta == pytest.approx(0.6667, abs=0.001)

def test_train_test_split_size():
    """
    Test that the train and test datasets have the expected size and type.
    """
    data = pd.read_csv("data/census.csv")
    train, test = train_test_split(data, test_size=0.20, random_state=42)
    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    assert len(train) + len(test) == len(data)
    assert len(test) == pytest.approx(len(data) * 0.20, abs=1)
