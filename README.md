# MLflow + FastAPI service

Сервис на FastAPI, который:

- при старте приложения загружает ML‑модель из MLflow
- имеет хэндлер `POST /predict` — принимает на вход признаки (лишние поля отбрасываются, недостающие для модели — ошибка валидации)
- имеет хэндлер `POST /updateModel`, который принимает `run_id` и подменяет текущую модель на модель из этого run
- экспортирует метрики Prometheus на `GET /metrics` (порт приложения должен быть в диапазоне **8890–8899** для скрейпа на стенде курса)

## Переменные окружения

- **`MLFLOW_TRACKING_URI`**: URI MLflow Tracking Server (например, `http://158.160.2.37:5000/`)
- **`DEFAULT_RUN_ID`**: модель при старте загрузится из этого run ([эксперимент 37](http://158.160.2.37:5000/#/experiments/37))
- **`EVIDENTLY_URL`**: URL Evidently UI (по умолчанию `http://158.160.2.37:8000/`)
- **`EVIDENTLY_PROJECT_ID`**: UUID проекта в Evidently для отчётов о дрифте (создайте проект в UI и укажите ID)
- **`EVIDENTLY_BUFFER_SIZE`**: размер буфера наблюдений (по умолчанию `500`)
- **`EVIDENTLY_REPORT_INTERVAL_SEC`**: интервал фоновой отправки отчёта (по умолчанию `300`)

### Grafana

Пароль к [Grafana](http://158.160.2.37:3000/) выдаётся на Степике после указания логина ВМ из списка курса. Описание метрик, дашборда и алертов — в [`report.md`](report.md).

## Запуск

```bash
export MLFLOW_TRACKING_URI=http://158.160.2.37:5000/
export DEFAULT_RUN_ID=<your_run_id>
# опционально для дрифта:
# export EVIDENTLY_PROJECT_ID=<uuid>
docker compose up --build
```

Сервис будет доступен на `http://<ip>:1488` (маппинг на порт **8890** внутри контейнера).

## Тесты

С корнем проекта в `PYTHONPATH` (см. `pytest.ini`):

```bash
pip install -r requirements.txt
pytest -q
```

Без локального `pip` — через тот же образ, что и сервис:

```bash
docker compose build
docker compose run --rm mlflow_example pytest -q
```
