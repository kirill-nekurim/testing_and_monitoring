"""
Accumulate observations and periodically push Evidently drift snapshots to RemoteWorkspace.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Optional
from collections import deque
from typing import Any

import pandas as pd

from ml_service import config

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_buffer: deque[dict[str, Any]] | None = None


def _ensure_buffer() -> deque[dict[str, Any]]:
    global _buffer
    if _buffer is None:
        _buffer = deque(maxlen=config.evidently_buffer_size())
    return _buffer


def record_observation(row: dict[str, Any]) -> None:
    """Append one row (features + prediction + probability)."""
    if not config.evidently_project_id():
        return
    with _lock:
        _ensure_buffer().append(dict(row))


def _build_and_push_report() -> None:
    project_id = config.evidently_project_id()
    if not project_id:
        return

    from evidently import Report
    from evidently.presets import DataDriftPreset
    from evidently.ui.workspace import RemoteWorkspace

    with _lock:
        rows = list(_ensure_buffer())
    min_rows = max(40, config.evidently_buffer_size() // 4)
    if len(rows) < min_rows:
        logger.debug('skip evidently: buffer=%s < %s', len(rows), min_rows)
        return

    mid = len(rows) // 2
    reference_data = pd.DataFrame(rows[:mid])
    current_data = pd.DataFrame(rows[mid:])

    drift_report = Report(metrics=[DataDriftPreset()])
    snapshot = drift_report.run(reference_data=reference_data, current_data=current_data)

    workspace = RemoteWorkspace(config.evidently_url())
    workspace.add_run(project_id, snapshot)
    logger.info('Evidently drift report saved to project %s', project_id)


async def drift_report_loop() -> None:
    while True:
        await asyncio.sleep(config.evidently_report_interval_sec())
        try:
            await asyncio.to_thread(_build_and_push_report)
        except Exception:
            logger.exception('Evidently drift report failed')


def start_drift_background_task() -> Optional[asyncio.Task]:
    if not config.evidently_project_id():
        logger.info('EVIDENTLY_PROJECT_ID is not set; drift background task disabled')
        return None
    return asyncio.create_task(drift_report_loop())
