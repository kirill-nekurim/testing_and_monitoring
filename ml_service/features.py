from __future__ import annotations

import pandas as pd

from ml_service.schemas import PredictRequest


FEATURE_COLUMNS = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education.num',
    'marital.status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital.gain',
    'capital.loss',
    'hours.per.week',
    'native.country',
]


def _attr_for_column(column: str) -> str:
    return column.replace('.', '_')


def request_row_dict(req: PredictRequest, columns: list[str]) -> dict[str, object]:
    """Build a plain dict of feature values for logging/metrics (pipeline column names as keys)."""
    return {col: getattr(req, _attr_for_column(col)) for col in columns}


def validate_required_features(req: PredictRequest, needed_columns: list[str]) -> list[str]:
    """
    Return a list of pipeline column names that are missing (None) in the request.
    """
    missing: list[str] = []
    for column in needed_columns:
        if column not in FEATURE_COLUMNS:
            missing.append(column)
            continue
        val = getattr(req, _attr_for_column(column))
        if val is None:
            missing.append(column)
    return missing


def to_dataframe(req: PredictRequest, needed_columns: list[str] | None = None) -> pd.DataFrame:
    if needed_columns is None:
        columns = FEATURE_COLUMNS
    else:
        columns = list(needed_columns)
    row = []
    for column in columns:
        if column not in FEATURE_COLUMNS:
            raise ValueError(f'Unsupported pipeline column for API schema: {column!r}')
        row.append(getattr(req, _attr_for_column(column)))
    return pd.DataFrame([row], columns=columns)
