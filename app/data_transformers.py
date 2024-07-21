from datetime import datetime
from typing import Optional, TypedDict
from app.models import Bathroom, MskFeatures, MskInput, Renovation, RuFeatures, RuInput, WallMaterial
import pandas as pd
import re


class AdditionalInfo(TypedDict):
    Население: int
    ЗП: float
    Округ: str


def add_additional_info(city: str, region: str) -> AdditionalInfo:
    def normalize_string(s):
        if pd.isna(s):
            return s
        s = s.lower()
        s = re.sub(r'[^a-zа-я0-9\s]', '', s)
        s = re.sub(r'\s+', ' ', s)
        return s.strip()

    def normalize_region_name(name):
        # Add your region name normalization logic here
        return name

    # Load salary data
    df_salary_data = pd.read_csv('./data/additional_data/salary_region.csv')
    df_salary_data['Регион'] = df_salary_data['Регион'].apply(
        normalize_region_name)

    # Create dictionaries for mapping
    okrug_dict = dict(zip(df_salary_data['Регион'], df_salary_data['Округ']))
    salary_dict = dict(zip(df_salary_data['Регион'], df_salary_data['ЗП']))

    # Load population data
    population_data = pd.read_csv('./data/additional_data/city_population.csv')
    population_data = population_data.rename(columns={
        "key": "Город",
        "population": "Население"
    })

    # Normalize city and region
    normalized_city = normalize_string(city)
    normalized_region = normalize_region_name(region)

    # Create full city name (city + region)
    full_city_name = f"{normalized_city} {normalized_region}"

    # Get population
    population = population_data[population_data['Город']
                                 == full_city_name]['Население'].values
    population = population[0] if len(population) > 0 else 0

    # Get salary and district
    salary = salary_dict.get(normalized_region, 0)
    district = okrug_dict.get(normalized_region, "")

    return {
        "Население": population,
        "ЗП": salary,
        "Округ": district
    }


class AgeData(TypedDict):
    ВозрастДома: int
    Возраст_Established_20_40_years: bool
    Возраст_Modern_10_20_years: bool
    Возраст_New_0_5_years: bool
    Возраст_Old_40_plus_years: bool
    Возраст_Recent_5_10_years: bool


def encode_age_data(construction_year: int) -> AgeData:
    current_year = datetime.now().year
    # Ensure age is not negative
    age = max(0, current_year - construction_year)

    def categorize_age(age: int) -> str:
        if age < 5:
            return 'New (0-5 years)'
        elif age < 10:
            return 'Recent (5-10 years)'
        elif age < 20:
            return 'Modern (10-20 years)'
        elif age < 40:
            return 'Established (20-40 years)'
        else:
            return 'Old (40+ years)'

    age_category = categorize_age(age)

    return {
        "ВозрастДома": age,
        "Возраст_Established_20_40_years": (age_category == 'Established (20-40 years)'),
        "Возраст_Modern_10_20_years": (age_category == 'Modern (10-20 years)'),
        "Возраст_New_0_5_years": (age_category == 'New (0-5 years)'),
        "Возраст_Old_40_plus_years": (age_category == 'Old (40+ years)'),
        "Возраст_Recent_5_10_years": (age_category == 'Recent (5-10 years)')
    }


class CityDistanceData(TypedDict):
    Расст_центр_Близко_к_городу: bool
    Расст_центр_Далеко_от_города: bool
    Расст_центр_Нет_данных: bool


def encode_city_distance(distance: Optional[float]) -> CityDistanceData:
    def categorize_distance(dist: Optional[float]) -> str:
        if dist is None or pd.isna(dist):
            return 'Нет данных'
        if dist < 17:
            return 'Близко к городу'
        else:
            return 'Далеко от города'

    distance_category = categorize_distance(distance)

    return CityDistanceData(
        Расст_центр_Близко_к_городу=distance_category == 'Близко к городу',
        Расст_центр_Далеко_от_города=distance_category == 'Далеко от города',
        Расст_центр_Нет_данных=distance_category == 'Нет данных'
    )


def encode_bathroom(bathroom: Bathroom) -> int:
    bathroom_lower = str(bathroom).lower()

    if 'на улице' in bathroom_lower and 'в доме' in bathroom_lower:
        return 2
    if 'в доме' in bathroom_lower:
        return 1
    if 'на улице' in bathroom_lower:
        return 0
    else:
        return 0


def prepare_features_ru(input_data: RuInput) -> RuFeatures:
    additional_info = add_additional_info(input_data.city, input_data.region)
    age_data = encode_age_data(input_data.construction_year)
    city_distance_data = encode_city_distance(input_data.distance_to_center)
    bathroom = encode_bathroom(input_data.bathroom)

    features: RuFeatures = {
        "Площ_дома": input_data.house_area,
        "Площ_Участка": input_data.land_area,
        "Кол_воЭтаж": input_data.floors,
        "Есть_баня": input_data.has_sauna,
        "Есть_бассейн": input_data.has_pool,
        "Есть_магазин": input_data.has_shop,
        "Есть_аптека": input_data.has_pharmacy,
        "Есть_детский_сад": input_data.has_kindergarten,
        "Есть_школа": input_data.has_school,
        "Есть_wifi": input_data.has_wifi,
        "Есть_tv": input_data.has_tv,
        "Кол_воКомн_encoded": input_data.rooms,
        "Свободная_планировка": input_data.has_open_plan,
        "Ремонт_дизайнерский": input_data.renovation == Renovation.designer,
        "Ремонт_евро": input_data.renovation == Renovation.euro,
        "Ремонт_косметический": input_data.renovation == Renovation.cosmetic,
        "Ремонт_требует_ремонта": input_data.renovation == Renovation.needs_renovation,
        "МатериалСтен_бревно":  input_data.wall_material == WallMaterial.brevno,
        "МатериалСтен_брус": input_data.wall_material == WallMaterial.brus,
        "МатериалСтен_газоблоки": input_data.wall_material == WallMaterial.gazobloki,
        "МатериалСтен_железобетонные_панели": input_data.wall_material == WallMaterial.zhelezo_panels,
        "МатериалСтен_кирпич": input_data.wall_material == WallMaterial.kirpich,
        "МатериалСтен_металл": input_data.wall_material == WallMaterial.metall,
        "МатериалСтен_пеноблоки": input_data.wall_material == WallMaterial.penoblock,
        "МатериалСтен_сэндвич_панели": input_data.wall_material == WallMaterial.sandvich_panels,
        "МатериалСтен_экспериментальные_материалы": input_data.wall_material == WallMaterial.experiment_materials,
        "Есть_парковка": input_data.has_parking,
        "Есть_гараж": input_data.has_garage,
        "Возможна_ипотека": input_data.mortgage_available,
        "Есть_терраса": input_data.has_terrace,
        "Есть_асфальт": input_data.has_asphalt,
        "Есть_общ_транспорт": input_data.has_public_transport,
        "Есть_жд": input_data.has_railway,
        "Есть_электричество": input_data.has_electricity,
        "Есть_газ": input_data.has_gas,
        "Есть_отопление": input_data.has_heating,
        "Есть_канализация": input_data.has_sewerage,
        "Город": input_data.city,
        "Регион": input_data.region,
        "Санузел": bathroom,
        **age_data,
        **additional_info,
        **city_distance_data,
    }
    return features


def prepare_features_msk(input_data: MskInput) -> MskFeatures:
    age_data = encode_age_data(input_data.construction_year)
    bathroom = encode_bathroom(input_data.bathroom)

    features: MskFeatures = {
        "Площ_дома": input_data.house_area,
        "Площ_Участка": input_data.land_area,
        "Расстояние_от_МКАД": input_data.distance_to_mkad,
        "Кол_воЭтаж": input_data.floors,
        "Есть_баня": input_data.has_sauna,
        "Есть_бассейн": input_data.has_pool,
        "Есть_магазин": input_data.has_shop,
        "Есть_аптека": input_data.has_pharmacy,
        "Есть_детский_сад": input_data.has_kindergarten,
        "Есть_школа": input_data.has_school,
        "Есть_wifi": input_data.has_wifi,
        "Есть_tv": input_data.has_tv,
        "Кол_воКомн_encoded": input_data.rooms,
        "Свободная_планировка": input_data.has_open_plan,
        "Ремонт_дизайнерский": input_data.renovation == Renovation.designer,
        "Ремонт_евро": input_data.renovation == Renovation.euro,
        "Ремонт_косметический": input_data.renovation == Renovation.cosmetic,
        "Ремонт_требует_ремонта": input_data.renovation == Renovation.needs_renovation,
        "МатериалСтен_бревно":  input_data.wall_material == WallMaterial.brevno,
        "МатериалСтен_брус": input_data.wall_material == WallMaterial.brus,
        "МатериалСтен_газоблоки": input_data.wall_material == WallMaterial.gazobloki,
        "МатериалСтен_железобетонные_панели": input_data.wall_material == WallMaterial.zhelezo_panels,
        "МатериалСтен_кирпич": input_data.wall_material == WallMaterial.kirpich,
        "МатериалСтен_металл": input_data.wall_material == WallMaterial.metall,
        "МатериалСтен_пеноблоки": input_data.wall_material == WallMaterial.penoblock,
        "МатериалСтен_сэндвич_панели": input_data.wall_material == WallMaterial.sandvich_panels,
        "МатериалСтен_экспериментальные_материалы": input_data.wall_material == WallMaterial.experiment_materials,
        "Есть_парковка": input_data.has_parking,
        "Есть_гараж": input_data.has_garage,
        "Возможна_ипотека": input_data.mortgage_available,
        "Есть_терраса": input_data.has_terrace,
        "Есть_асфальт": input_data.has_asphalt,
        "Есть_общ_транспорт": input_data.has_public_transport,
        "Есть_жд": input_data.has_railway,
        "Есть_электричество": input_data.has_electricity,
        "Есть_газ": input_data.has_gas,
        "Есть_отопление": input_data.has_heating,
        "Есть_канализация": input_data.has_sewerage,
        "Город": input_data.city,
        "Санузел": bathroom,
        **age_data,
    }
    return features
