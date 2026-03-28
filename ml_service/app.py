from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request, Response
from mlflow.exceptions import MlflowException, RestException

from ml_service import config
from ml_service.evidently_drift import record_observation, start_drift_background_task
from ml_service.features import (
    FEATURE_COLUMNS,
    request_row_dict,
    to_dataframe,
    validate_required_features,
)
from ml_service.metrics import (
    metrics_response,
    observe_feature_row,
    observe_http,
    observe_inference,
    observe_model_update_failure,
    observe_model_update_success,
    observe_prediction,
    observe_preprocess,
    set_model_metadata,
)
from ml_service.mlflow_utils import configure_mlflow
from ml_service.model import Model, infer_estimator_name
from ml_service.schemas import (
    PredictRequest,
    PredictResponse,
    UpdateModelRequest,
    UpdateModelResponse,
)

MODEL = Model()
_drift_task: Optional[asyncio.Task] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _drift_task
    configure_mlflow()
    run_id = config.default_run_id()
    try:
        MODEL.set(run_id=run_id)
    except Exception as e:
        raise RuntimeError(f'Failed to load initial model for run_id={run_id!r}: {e}') from e

    md = MODEL.get()
    if md.model is not None:
        set_model_metadata(
            run_id=run_id,
            estimator_name=infer_estimator_name(md.model),
            features=list(MODEL.features),
        )

    _drift_task = start_drift_background_task()
    yield

    if _drift_task and not _drift_task.done():
        _drift_task.cancel()
        try:
            await _drift_task
        except asyncio.CancelledError:
            pass


def create_app() -> FastAPI:
    app = FastAPI(title='MLflow FastAPI service', version='1.0.0', lifespan=lifespan)

    @app.middleware('http')
    async def add_metrics_middleware(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start
        path = request.url.path
        if path.startswith('/predict'):
            handler = 'predict'
        elif path.startswith('/updateModel'):
            handler = 'updateModel'
        elif path.startswith('/health'):
            handler = 'health'
        elif path.startswith('/metrics'):
            handler = 'metrics'
        else:
            handler = 'other'
        observe_http(
            method=request.method,
            handler=handler,
            status_code=response.status_code,
            duration_sec=duration,
        )
        return response

    @app.get('/health')
    def health() -> dict[str, Any]:
        model_state = MODEL.get()
        run_id = model_state.run_id
        return {'status': 'ok', 'run_id': run_id}

    @app.get('/metrics')
    def metrics() -> Response:
        payload, media_type = metrics_response()
        return Response(content=payload, media_type=media_type)

    @app.post('/predict', response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        model_state = MODEL.get()
        model = model_state.model
        if model is None:
            raise HTTPException(status_code=503, detail='Model is not loaded yet')

        needed = list(MODEL.features)
        unsupported = [c for c in needed if c not in FEATURE_COLUMNS]
        if unsupported:
            raise HTTPException(
                status_code=500,
                detail=f'Model requires features not exposed by the API: {unsupported}',
            )

        missing = validate_required_features(request, needed)
        if missing:
            raise HTTPException(
                status_code=422,
                detail={
                    'message': 'Required feature values are missing or null',
                    'missing_features': missing,
                },
            )

        t_pre0 = time.perf_counter()
        try:
            df = to_dataframe(request, needed_columns=needed)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        t_pre1 = time.perf_counter()
        observe_preprocess('predict', t_pre1 - t_pre0)

        row_for_metrics = request_row_dict(request, needed)
        observe_feature_row(row_for_metrics)

        t_inf0 = time.perf_counter()
        try:
            probability = float(model.predict_proba(df)[0][1])
        except Exception as e:
            raise HTTPException(status_code=400, detail=f'Inference failed: {e}') from e
        t_inf1 = time.perf_counter()
        observe_inference('predict', t_inf1 - t_inf0)

        prediction = int(probability >= 0.5)
        observe_prediction(probability, prediction)

        obs = dict(row_for_metrics)
        obs['prediction'] = prediction
        obs['probability'] = probability
        record_observation(obs)

        return PredictResponse(prediction=prediction, probability=probability)

    @app.post('/updateModel', response_model=UpdateModelResponse)
    def update_model(req: UpdateModelRequest) -> UpdateModelResponse:
        run_id = req.run_id.strip()
        if not run_id:
            raise HTTPException(status_code=422, detail='run_id must be non-empty')

        try:
            MODEL.set(run_id=run_id)
        except RestException as e:
            observe_model_update_failure()
            if getattr(e, 'error_code', None) == 'RESOURCE_DOES_NOT_EXIST':
                raise HTTPException(status_code=404, detail='Run or model artifact was not found') from e
            raise HTTPException(status_code=502, detail=f'MLflow error: {e!s}') from e
        except MlflowException as e:
            observe_model_update_failure()
            raise HTTPException(status_code=502, detail=f'Failed to load model: {e!s}') from e
        except OSError as e:
            observe_model_update_failure()
            raise HTTPException(status_code=502, detail=f'Failed to load model artifact: {e!s}') from e
        except Exception as e:
            observe_model_update_failure()
            raise HTTPException(status_code=502, detail=f'Failed to load model: {e!s}') from e

        md = MODEL.get()
        if md.model is None:
            observe_model_update_failure()
            raise HTTPException(status_code=500, detail='Model is not loaded after update')

        set_model_metadata(
            run_id=run_id,
            estimator_name=infer_estimator_name(md.model),
            features=list(MODEL.features),
        )
        observe_model_update_success()

        return UpdateModelResponse(run_id=run_id)

    return app


app = create_app()
