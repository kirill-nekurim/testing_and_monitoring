"""
Prometheus metrics for the ML service (histograms support p75–p99.9 via Grafana histogram_quantile).
"""

from __future__ import annotations

import threading
from typing import Any

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, Info, generate_latest

# Buckets tuned for request latency SLAs (seconds)
_LATENCY_BUCKETS = (
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2.5,
    5.0,
    7.5,
    10.0,
    20.0,
    30.0,
    float('inf'),
)

_PROB_BUCKETS = tuple(round(i * 0.05, 3) for i in range(21)) + (1.0,)

http_requests_total = Counter(
    'ml_http_requests_total',
    'Total HTTP requests',
    ['method', 'handler', 'status'],
)

http_request_duration_seconds = Histogram(
    'ml_http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'handler'],
    buckets=_LATENCY_BUCKETS,
)

http_errors_total = Counter(
    'ml_http_errors_total',
    'HTTP 5xx responses',
    ['handler'],
)

preprocess_duration_seconds = Histogram(
    'ml_preprocess_duration_seconds',
    'Feature preprocessing duration',
    ['handler'],
    buckets=_LATENCY_BUCKETS,
)

inference_duration_seconds = Histogram(
    'ml_inference_duration_seconds',
    'Model inference duration',
    ['handler'],
    buckets=_LATENCY_BUCKETS,
)

model_probability = Histogram(
    'ml_model_probability',
    'Distribution of positive-class probabilities',
    buckets=_PROB_BUCKETS,
)

model_prediction_total = Counter(
    'ml_model_prediction_total',
    'Predicted class counts',
    ['class_label'],
)

feature_numeric_value = Histogram(
    'ml_feature_numeric_value',
    'Observed numeric feature values',
    ['feature'],
    buckets=(
        -1e9,
        0,
        1,
        10,
        100,
        1_000,
        10_000,
        100_000,
        1_000_000,
        float('inf'),
    ),
)

feature_categorical_total = Counter(
    'ml_feature_categorical_total',
    'Categorical feature value observations',
    ['feature', 'value_bucket'],
)

model_updates_total = Counter(
    'ml_model_updates_total',
    'Model reload attempts from MLflow',
    ['status'],
)

model_info = Info(
    'ml_model',
    'Currently loaded production model metadata',
)

model_metadata_version = Gauge(
    'ml_model_metadata_version',
    'Monotonic version incremented on each successful model activation',
)

model_required_features_count = Gauge(
    'ml_model_required_features_count',
    'Number of features required by the active pipeline',
)

process_cpu_percent = Gauge(
    'ml_process_cpu_percent',
    'Approximate process CPU usage percent (best-effort, psutil)',
)

process_rss_bytes = Gauge(
    'ml_process_resident_memory_bytes',
    'Process resident set size in bytes (best-effort)',
)

_lock = threading.Lock()
_metadata_version = 0


def observe_http(
    *,
    method: str,
    handler: str,
    status_code: int,
    duration_sec: float,
) -> None:
    label = str(status_code)
    http_requests_total.labels(method=method, handler=handler, status=label).inc()
    http_request_duration_seconds.labels(method=method, handler=handler).observe(duration_sec)
    if status_code >= 500:
        http_errors_total.labels(handler=handler).inc()


def observe_preprocess(handler: str, seconds: float) -> None:
    preprocess_duration_seconds.labels(handler=handler).observe(seconds)


def observe_inference(handler: str, seconds: float) -> None:
    inference_duration_seconds.labels(handler=handler).observe(seconds)


def observe_prediction(probability: float, prediction: int) -> None:
    model_probability.observe(probability)
    model_prediction_total.labels(class_label=str(prediction)).inc()


def observe_feature_row(row: dict[str, Any]) -> None:
    """Record coarse feature statistics for monitoring distributions."""
    for key, val in row.items():
        if val is None:
            continue
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            feature_numeric_value.labels(feature=key).observe(float(val))
        else:
            s = str(val)
            bucket = s if len(s) <= 64 else s[:61] + '...'
            feature_categorical_total.labels(feature=key, value_bucket=bucket).inc()


def set_model_metadata(*, run_id: str, estimator_name: str, features: list[str]) -> None:
    global _metadata_version
    with _lock:
        _metadata_version += 1
        model_metadata_version.set(_metadata_version)
        model_required_features_count.set(len(features))
        model_info.info(
            {
                'run_id': run_id,
                'estimator': estimator_name,
                'features': '|'.join(features),
            }
        )


def observe_model_update_success() -> None:
    model_updates_total.labels(status='success').inc()


def observe_model_update_failure() -> None:
    model_updates_total.labels(status='failure').inc()


def update_resource_gauges() -> None:
    try:
        import psutil

        p = psutil.Process()
        process_cpu_percent.set(p.cpu_percent(interval=None))
        process_rss_bytes.set(p.memory_info().rss)
    except Exception:
        pass


def metrics_response() -> tuple[bytes, str]:
    update_resource_gauges()
    return generate_latest(), CONTENT_TYPE_LATEST
