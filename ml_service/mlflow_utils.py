from __future__ import annotations

from typing import Any, Optional

import mlflow

from ml_service import config


def configure_mlflow() -> None:
    uri = config.tracking_uri()
    if uri:
        mlflow.set_tracking_uri(uri)


def get_model_uri(run_id: str) -> str:
    return f'runs:/{run_id}/model'


def load_model(model_uri: Optional[str] = None, run_id: Optional[str] = None) -> Any:
    """
    Downloads artifacts locally (if needed) and loads the sklearn model / pipeline.
    """
    if not model_uri:
        model_uri = get_model_uri(run_id)  # type: ignore[arg-type]
    return mlflow.sklearn.load_model(model_uri)
