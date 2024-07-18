from typing import TypedDict
from autogluon.tabular import TabularPredictor
import numpy as np


class RuFeatures(TypedDict):
    Площ_дома: float
    Площ_Участка: float
    Санузел: int
    Кол_воЭтаж: int
    Расст_центр_Близко_к_городу: bool
    Расст_центр_Далеко_от_города: bool
    Расст_центр_Нет_данных: bool
    Есть_баня: bool
    Есть_бассейн: bool
    Есть_магазин: bool
    Есть_аптека: bool
    Есть_детский_сад: bool
    Есть_школа: bool
    Есть_wifi: bool
    Есть_tv: bool
    Кол_воКомн_encoded: int
    Свободная_планировка: bool
    Ремонт_дизайнерский: bool
    Ремонт_евро: bool
    Ремонт_косметический: bool
    Ремонт_требует_ремонта: bool
    МатериалСтен_бревно: bool
    МатериалСтен_брус: bool
    МатериалСтен_газоблоки: bool
    МатериалСтен_железобетонные_панели: bool
    МатериалСтен_кирпич: bool
    МатериалСтен_металл: bool
    МатериалСтен_пеноблоки: bool
    МатериалСтен_сэндвич_панели: bool
    МатериалСтен_экспериментальные_материалы: bool
    Есть_парковка: bool
    Есть_гараж: bool
    Возможна_ипотека: bool
    Есть_терраса: bool
    Есть_асфальт: bool
    Есть_общ_транспорт: bool
    Есть_жд: bool
    ВозрастДома: int
    ВозрастДома_Squared: int
    Возраст_Established_20_40_years: bool
    Возраст_Modern_10_20_years: bool
    Возраст_New_0_5_years: bool
    Возраст_Old_40_plus_years: bool
    Возраст_Recent_5_10_years: bool
    Есть_электричество: bool
    Есть_газ: bool
    Есть_отопление: bool
    Есть_канализация: bool
    ЗП: float
    Население: int
    Город: str
    Регион: str
    Округ: str


class MskFeatures(TypedDict):
    Площ_дома: float
    Площ_Участка: float
    Санузел: int
    Расстояние_от_МКАД: float
    Кол_воЭтаж: int
    Есть_баня: bool
    Есть_бассейн: bool
    Есть_магазин: bool
    Есть_аптека: bool
    Есть_детский_сад: bool
    Есть_школа: bool
    Есть_wifi: bool
    Есть_tv: bool
    Кол_воКомн_encoded: int
    Свободная_планировка: bool
    Ремонт_дизайнерский: bool
    Ремонт_евро: bool
    Ремонт_косметический: bool
    Ремонт_требует_ремонта: bool
    МатериалСтен_бревно: bool
    МатериалСтен_брус: bool
    МатериалСтен_газоблоки: bool
    МатериалСтен_железобетонные_панели: bool
    МатериалСтен_кирпич: bool
    МатериалСтен_металл: bool
    МатериалСтен_пеноблоки: bool
    МатериалСтен_сэндвич_панели: bool
    МатериалСтен_экспериментальные_материалы: bool
    Есть_парковка: bool
    Есть_гараж: bool
    Возможна_ипотека: bool
    Есть_терраса: bool
    Есть_асфальт: bool
    Есть_общ_транспорт: bool
    Есть_жд: bool
    ВозрастДома: int
    ВозрастДома_Squared: int
    Возраст_Established_20_40_years: bool
    Возраст_Modern_10_20_years: bool
    Возраст_New_0_5_years: bool
    Возраст_Old_40_plus_years: bool
    Возраст_Recent_5_10_years: bool
    Есть_электричество: bool
    Есть_газ: bool
    Есть_отопление: bool
    Есть_канализация: bool
    Город: str


def predict_price_ru(predictor: TabularPredictor, features: RuFeatures) -> float:
    prediction = predictor.predict(features)
    # Convert log price back to actual price
    return float(np.expm1(prediction[0]))


def predict_price_msk(predictor: TabularPredictor, features: MskFeatures) -> float:
    prediction = predictor.predict(features)
    # Convert log price back to actual price
    return float(np.expm1(prediction[0]))
