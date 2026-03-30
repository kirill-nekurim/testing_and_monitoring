import pytest
from fastapi.testclient import TestClient
from mlflow.exceptions import RestException

from ml_service.app import create_app


def test_predict_missing_features_returns_422(mock_mlflow_load):
    app = create_app()
    with TestClient(app) as client:
        r = client.post('/predict', json={'race': 'White'})
        assert r.status_code == 422
        body = r.json()
        assert 'missing_features' in body['detail']


def test_predict_success(mock_mlflow_load):
    app = create_app()
    with TestClient(app) as client:
        r = client.post(
            '/predict',
            json={
                'race': 'White',
                'sex': 'Male',
                'capital.gain': 0,
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert 'probability' in data
        assert 'prediction' in data


def test_update_model_invalid_run(mock_mlflow_load, monkeypatch):
    def boom(run_id=None, model_uri=None):
        raise RestException({'error_code': 'RESOURCE_DOES_NOT_EXIST', 'message': 'nope'})

    app = create_app()
    with TestClient(app) as client:
        monkeypatch.setattr('ml_service.mlflow_utils.load_model', boom)
        r = client.post('/updateModel', json={'run_id': 'missing-run-id'})
        assert r.status_code == 404


def test_metrics_endpoint_exposes_prometheus(mock_mlflow_load):
    app = create_app()
    with TestClient(app) as client:
        r = client.get('/metrics')
        assert r.status_code == 200
        text = r.text
        assert 'ml_http_requests_total' in text
        assert 'ml_preprocess_duration_seconds' in text


def test_health_contains_run_id(mock_mlflow_load):
    app = create_app()
    with TestClient(app) as client:
        r = client.get('/health')
        assert r.status_code == 200
        assert r.json()['run_id'] == 'test-run'
