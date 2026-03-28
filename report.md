# Отчёт: тестирование и мониторинг ML-сервиса

## 1. Доработки надёжности и обработка ошибок

| Ситуация | Поведение | HTTP | Где в коде |
|----------|-----------|------|------------|
| Не переданы или `null` все обязательные для текущей модели признаки | Явная проверка списка колонок пайплайна (`feature_names_in_`) | **422** с `missing_features` | [`ml_service/features.py`](ml_service/features.py) `validate_required_features`, [`ml_service/app.py`](ml_service/app.py) `/predict` |
| Некорректное тело запроса / типы (Pydantic) | Стандартная валидация FastAPI | **422** (автоматически) | [`ml_service/schemas.py`](ml_service/schemas.py) |
| Пайплайн требует колонку, которой нет в схеме API | Защита от несовместимости артефакта и контракта | **500** с перечислением колонок | [`ml_service/app.py`](ml_service/app.py) |
| Ошибка сборки `DataFrame` (неизвестная колонка в пайплайне) | `ValueError` → понятное сообщение | **400** | [`ml_service/features.py`](ml_service/features.py) `to_dataframe` |
| Ошибка `predict_proba` (несовместимые данные и т.п.) | Исключение при инференсе | **400** | [`ml_service/app.py`](ml_service/app.py) |
| Модель не загружена | Проверка перед инференсом | **503** | [`ml_service/app.py`](ml_service/app.py) |
| `run_id` пустой после `strip()` | Валидация | **422** | [`ml_service/app.py`](ml_service/app.py) `/updateModel` |
| Run или артефакт не найдены в MLflow (`RestException`, `RESOURCE_DOES_NOT_EXIST`) | Без смены рабочей модели (загрузка не удалась до замены) | **404** | [`ml_service/app.py`](ml_service/app.py) |
| Прочие ошибки MLflow / диска при загрузке модели | Логичная ошибка интеграции | **502** / **500** | [`ml_service/app.py`](ml_service/app.py) |
| Сбой при стартовой загрузке модели | Явный `RuntimeError` при старте приложения | процесс не поднимается | [`ml_service/app.py`](ml_service/app.py) `lifespan` |

Загрузка модели атомарна: сначала скачивается и строится объект в памяти, затем подменяется указатель в [`Model`](ml_service/model.py), поэтому при ошибке `load_model` прежняя модель (если была) не затирается частичным состоянием.

---

## 2. Тесты

Добавлены автотесты на **pytest** (каталог [`tests/`](tests/)).

| Критерий из задания | Файлы / что проверяется |
|---------------------|-------------------------|
| Функции предобработки | [`tests/test_features.py`](tests/test_features.py): `to_dataframe`, `validate_required_features`, `request_row_dict`, согласование с `sklearn` пайплайном |
| Успешность загрузки и инференса | [`tests/conftest.py`](tests/conftest.py) мок `load_model` → реальный `sklearn` `Pipeline`; [`tests/test_handlers.py`](tests/test_handlers.py) успешный `/predict` |
| Логика хэндлеров и валидация | [`tests/test_handlers.py`](tests/test_handlers.py): 422 при нехватке фич, 404 при несуществующем `run_id`, `/metrics` |
| Сервис целиком | [`tests/test_app_e2e.py`](tests/test_app_e2e.py): подъём приложения через `TestClient`, полный `/predict` |

Запуск: `pytest -q` (см. [`README.md`](README.md)).

---

## 3. Метрики (Prometheus)

Эндпоинт: **`GET /metrics`**. Реализация: [`ml_service/metrics.py`](ml_service/metrics.py). Порт приложения в Docker: **8890** (диапазон 8890–8899 для скрейпа).

### 3.1. Технические метрики

- `ml_http_requests_total{method, handler, status}` — число запросов.
- `ml_http_request_duration_seconds{method, handler}` — гистограмма длительности HTTP (для перцентилей в Grafana).
- `ml_http_errors_total{handler}` — ответы **5xx** (инкремент в middleware при `status >= 500`).
- `ml_process_cpu_percent`, `ml_process_resident_memory_bytes` — ресурсы процесса (best-effort через `psutil` при скрейпе `/metrics`).

**Перцентили (p75, p90, p95, p99, p99.9)** считаются в Grafana по `histogram_quantile` от `_bucket` метрик, например:

```promql
histogram_quantile(0.95, sum by (le, handler) (rate(ml_http_request_duration_seconds_bucket[5m])))
```

Аналогично для `ml_preprocess_duration_seconds` и `ml_inference_duration_seconds`.

### 3.2. Метрики по входным данным

- `ml_preprocess_duration_seconds{handler}` — время подготовки признаков.
- `ml_feature_numeric_value{feature}` — гистограмма числовых признаков.
- `ml_feature_categorical_total{feature, value_bucket}` — счётчики категориальных значений (строка обрезается для длины).

### 3.3. Метрики модели

- `ml_inference_duration_seconds{handler}` — время `predict_proba`.
- `ml_model_probability` — гистограмма вероятности положительного класса.
- `ml_model_prediction_total{class_label}` — счётчики предсказанных классов.

### 3.4. Модель в проде и обновления

- `ml_model_info` (Info) — `run_id`, `estimator`, строка `features` (разделитель `|`).
- `ml_model_metadata_version` — увеличивается при каждой успешной активации модели (старт и успешный `/updateModel`).
- `ml_model_required_features_count` — число признаков у активного пайплайна.
- `ml_model_updates_total{status}` — `success` / `failure` при попытках загрузки в `/updateModel`.

---

## 4. Grafana: дашборд и импорт

1. Войти в Grafana курса: [http://158.160.2.37:3000/](http://158.160.2.37:3000/) (логин/пароль — по инструкции на Степике).
2. **Dashboards → Import** → загрузить файл [`grafana/ml-service-dashboard.json`](grafana/ml-service-dashboard.json).
3. Выбрать datasource **Prometheus**, соответствующий вашему стеку (UID подставляется при импорте как `${DS_PROMETHEUS}`).

В дашборде: панели по запросам, перцентилям латентности, 5xx, предобработке/инференсу, распределениям фич и предсказаний, ресурсам и текстовое описание блока «модель в проде».

---

## 5. Алертинг (не менее 5 правил)

Рекомендуется **Grafana Unified Alerting** (или Alertmanager) с контакт-поинтом **Telegram** / e-mail / webhook так, чтобы проверяющий мог увидеть сообщение.

| # | Тема | Пример условия (PromQL / смысл) | Как показать срабатывание |
|---|------|----------------------------------|---------------------------|
| 1 | Время ответа сервиса | `histogram_quantile(0.99, ...)` по `ml_http_request_duration_seconds_bucket` для `handler=predict` **>** порога (напр. 2s) | Нагрузочный скрипт на `/predict` или временно занизить порог |
| 2 | Время инференса | p95 `ml_inference_duration_seconds` **>** порога | Искусственная задержка недоступна без кода — используйте нагрузку или временно низкий порог |
| 3 | Число 5xx | `rate(ml_http_errors_total[5m])` **>** 0 | Временно сломать зависимость или вызвать ошибку, дающую 500 (осторожно в проде) |
| 4 | Ресурсы (CPU / память) | `ml_process_cpu_percent` или `ml_process_resident_memory_bytes` выше порога | Нагрузка CPU или большие батчи запросов |
| 5 | Распределение вероятностей | резкое изменение `histogram_quantile` по `ml_model_probability` или дисбаланс `ml_model_prediction_total` | Сместить входные данные тестовой нагрузкой |
| 6 | Доп. (фичи) | рост редких `value_bucket` в `ml_feature_categorical_total` | Подать выборку с новыми категориями |

**Демонстрация для проверяющего:** в описании PR или в чате курса приложите скрин срабатывания из Grafana/Alerting и/или сообщение в Telegram-чате (пригласительная ссылка на чат — по требованию задания).

---

## 6. Evidently: дрифт данных и предсказаний

- Накопление: в памяти (thread-safe `deque`) хранятся строки **признаки + `prediction` + `probability`** после каждого успешного `/predict` (если задан `EVIDENTLY_PROJECT_ID`).
- Периодичность: фоновая корутина [`ml_service/evidently_drift.py`](ml_service/evidently_drift.py) раз в `EVIDENTLY_REPORT_INTERVAL_SEC` (по умолчанию 300 с) строит снимок `Report(metrics=[DataDriftPreset()])` по половине буфера как reference и половине как current и отправляет в **Evidently RemoteWorkspace** (`workspace.add_run`).
- Переменные окружения: `EVIDENTLY_URL`, `EVIDENTLY_PROJECT_ID` (создайте проект в [Evidently UI](http://158.160.2.37:8000/) и подставьте UUID), `EVIDENTLY_BUFFER_SIZE` (по умолчанию 500).

Колонки в отчёте соответствуют фактическим ключам в буфере (включая исходные признаки и поля модели), что позволяет видеть сдвиг и по входам, и по распределению предсказаний/скоров.

---

## 7. Сдача на Степике

1. Открыть **Pull Request** в своём форке с этими изменениями.
2. В ответе на Степике указать **ссылку на PR**.
3. При алертах в Telegram — приложить **пригласительную ссылку** на чат с ботом Grafana.
4. Убедиться, что инстанс сервиса доступен для **Prometheus** на порту из **8890–8899** (в `docker-compose` проброшен **1488→8890**).
