"""
End-to-end style tests: app starts, model loads, prediction succeeds.
"""

from fastapi.testclient import TestClient

from ml_service.app import create_app


def test_service_predict_end_to_end(mock_mlflow_load):
    app = create_app()
    with TestClient(app) as client:
        res = client.post(
            '/predict',
            json={
                'race': 'White',
                'sex': 'Male',
                'capital.gain': 0,
            },
        )
        assert res.status_code == 200
        payload = res.json()
        assert 0.0 <= payload['probability'] <= 1.0
        assert payload['prediction'] in (0, 1)
