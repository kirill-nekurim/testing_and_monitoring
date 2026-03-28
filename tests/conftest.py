import os
from typing import Optional

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

os.environ.setdefault('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000/')
os.environ.setdefault('DEFAULT_RUN_ID', 'test-run')


def _build_dummy_pipeline() -> Pipeline:
    X = pd.DataFrame(
        {
            'race': ['White'] * 20,
            'sex': ['Male'] * 20,
            'capital.gain': np.random.randint(0, 10, size=20),
        }
    )
    pipe = Pipeline([('clf', DummyClassifier(strategy='prior'))])
    pipe.fit(X, [0, 1] * 10)
    return pipe


@pytest.fixture
def sklearn_pipeline() -> Pipeline:
    return _build_dummy_pipeline()


@pytest.fixture
def mock_mlflow_load(monkeypatch, sklearn_pipeline: Pipeline):
    def _load(run_id: Optional[str] = None, model_uri: Optional[str] = None):
        return sklearn_pipeline

    monkeypatch.setattr('ml_service.mlflow_utils.load_model', _load)
