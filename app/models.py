from pydantic import BaseModel, Field
from enum import Enum
from typing import TypedDict


class WallMaterial(str, Enum):
    brevno = "бревно"
    brus = "брус"
    zhelezo_panels = "железобетонные панели"
    kirpich = "кирпич"
    metall = "металл"
    penoblock = "пеноблоки"
    sandvich_panels = "сэндвич-панели"
    experiment_materials = "экспериментальные материалы"
    gazobloki = "газоблоки"


class Renovation(str, Enum):
    designer = "дизайнерский"
    euro = "евро"
    cosmetic = "косметический"
    needs_renovation = "требует ремонта"


class Bathroom(str, Enum):
    both = "на улице и в доме"
    inside = "в доме"
    outside = "на улице"


class BaseInput(BaseModel):
    house_area: float = Field(..., description="Площадь дома")
    land_area: float = Field(..., description="Площадь участка")
    bathroom: Bathroom = Field(..., description="Санузел")
    wall_material: WallMaterial = Field(
        ..., description="Материалы стен")
    floors: int = Field(..., description="Количество этажей")
    has_sauna: bool = Field(..., description="Наличие бани")
    has_pool: bool = Field(..., description="Наличие бассейна")
    has_shop: bool = Field(..., description="Наличие магазина поблизости")
    has_pharmacy: bool = Field(..., description="Наличие аптеки поблизости")
    has_kindergarten: bool = Field(...,
                                   description="Наличие детского сада поблизости")
    has_school: bool = Field(..., description="Наличие школы поблизости")
    has_wifi: bool = Field(..., description="Наличие Wi-Fi")
    has_tv: bool = Field(..., description="Наличие ТВ")
    rooms: int = Field(..., description="Количество комнат")
    has_open_plan: bool = Field(...,
                                description="Наличие свободной планировки")
    renovation: Renovation = Field(..., description="Тип ремонта")
    has_parking: bool = Field(..., description="Наличие парковки")
    has_garage: bool = Field(..., description="Наличие гаража")
    mortgage_available: bool = Field(..., description="Возможность ипотеки")
    has_terrace: bool = Field(..., description="Наличие террасы")
    has_asphalt: bool = Field(...,
                              description="Наличие асфальтированной дороги")
    has_public_transport: bool = Field(...,
                                       description="Наличие общественного транспорта")
    has_railway: bool = Field(...,
                              description="Наличие железнодорожного сообщения")
    construction_year: int = Field(..., description="Год постройки")
    has_electricity: bool = Field(..., description="Наличие электричества")
    has_gas: bool = Field(..., description="Наличие газа")
    has_heating: bool = Field(..., description="Наличие отопления")
    has_sewerage: bool = Field(..., description="Наличие канализации")
    city: str = Field(..., description="Город")


class RuInput(BaseInput):
    distance_to_center: float = Field(
        ..., description="Расстояние до центра")
    region: str = Field(..., description="Регион в формате Оренбургская область/Республика татарстан/Краснодарский край")


class MskInput(BaseInput):
    distance_to_mkad: float = Field(..., description="Расстояние от МКАД")


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
