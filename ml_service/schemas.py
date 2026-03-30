from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    age: Optional[int] = Field(default=None, description='Возраст человека')
    workclass: Optional[str] = Field(default=None, description='Тип занятости')
    fnlwgt: Optional[int] = Field(
        default=None,
        description='Вес наблюдения в данных переписи',
    )
    education: Optional[str] = Field(default=None, description='Образование')
    education_num: Optional[int] = Field(
        default=None,
        alias='education.num',
        description='Уровень образования в виде числа',
    )
    marital_status: Optional[str] = Field(
        default=None,
        alias='marital.status',
        description='Семейное положение',
    )
    occupation: Optional[str] = Field(
        default=None,
        description='Профессия / род деятельности',
    )
    relationship: Optional[str] = Field(
        default=None,
        description='Роль человека в семье',
    )
    race: Optional[str] = Field(default=None, description='Расовая группа')
    sex: Optional[str] = Field(
        default=None,
        description='Пол человека (Male / Female)',
    )
    capital_gain: Optional[int] = Field(
        default=None,
        alias='capital.gain',
        description='Доход от капитала (прибыль от продажи активов)',
    )
    capital_loss: Optional[int] = Field(
        default=None,
        alias='capital.loss',
        description='Убытки от капитала',
    )
    hours_per_week: Optional[int] = Field(
        default=None,
        alias='hours.per.week',
        description='Количество рабочих часов в неделю',
    )
    native_country: Optional[str] = Field(
        default=None,
        alias='native.country',
        description='Страна происхождения',
    )


class PredictResponse(BaseModel):
    prediction: int = Field(description='Предсказанный класс')
    probability: float = Field(description='Вероятность предсказанного класса')


class UpdateModelRequest(BaseModel):
    run_id: str = Field(min_length=1, description='MLflow run_id')


class UpdateModelResponse(BaseModel):
    run_id: str = Field(description='MLflow run_id')
