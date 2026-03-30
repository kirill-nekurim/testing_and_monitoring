from __future__ import annotations

import threading
from typing import NamedTuple

from sklearn.pipeline import Pipeline

from ml_service import mlflow_utils


class ModelData(NamedTuple):
    model: Pipeline | None
    run_id: str | None


def infer_estimator_name(pipeline: Pipeline) -> str:
    if hasattr(pipeline, 'steps') and pipeline.steps:
        last = pipeline.steps[-1][1]
        return type(last).__name__
    return type(pipeline).__name__


class Model:
    """
    Thread-safe container for the currently active model.
    """

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.data = ModelData(model=None, run_id=None)

    def get(self) -> ModelData:
        with self.lock:
            return self.data

    def set(self, run_id: str) -> None:
        model = mlflow_utils.load_model(run_id=run_id)
        with self.lock:
            self.data = ModelData(model=model, run_id=run_id)

    @property
    def features(self) -> list[str]:
        m = self.data.model
        if m is None:
            return []
        return list(m.feature_names_in_)
