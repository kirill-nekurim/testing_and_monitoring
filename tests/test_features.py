import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from ml_service.features import (
    FEATURE_COLUMNS,
    request_row_dict,
    to_dataframe,
    validate_required_features,
)
from ml_service.schemas import PredictRequest


def test_to_dataframe_subset_order():
    req = PredictRequest(
        age=30,
        race='White',
        sex='Male',
        capital_gain=100,
    )
    df = to_dataframe(req, needed_columns=['race', 'sex', 'capital.gain'])
    assert list(df.columns) == ['race', 'sex', 'capital.gain']
    assert df.iloc[0]['race'] == 'White'


def test_validate_required_features_missing():
    req = PredictRequest(race='White', sex=None, capital_gain=0)
    missing = validate_required_features(req, ['race', 'sex', 'capital.gain'])
    assert 'sex' in missing


def test_request_row_dict_roundtrip():
    req = PredictRequest(race='X', sex='Y', capital_gain=1)
    row = request_row_dict(req, ['race', 'sex', 'capital.gain'])
    assert row == {'race': 'X', 'sex': 'Y', 'capital.gain': 1}


def test_to_dataframe_unknown_column_raises():
    req = PredictRequest(race='White')
    with pytest.raises(ValueError, match='Unsupported pipeline column'):
        to_dataframe(req, needed_columns=['unknown_column'])


def test_feature_columns_cover_schema_aliases():
    assert 'capital.gain' in FEATURE_COLUMNS
    assert 'native.country' in FEATURE_COLUMNS


def test_sklearn_pipeline_feature_names_in_match_helpers():
    X = pd.DataFrame(
        {
            'race': ['White'] * 4,
            'sex': ['Male'] * 4,
            'capital.gain': [0, 1, 2, 3],
        }
    )
    pipe = Pipeline([('clf', DummyClassifier())])
    pipe.fit(X, [0, 1, 0, 1])
    cols = list(pipe.feature_names_in_)
    req = PredictRequest(
        race='White',
        sex='Male',
        capital_gain=10,
    )
    df = to_dataframe(req, needed_columns=cols)
    assert df.shape == (1, len(cols))
