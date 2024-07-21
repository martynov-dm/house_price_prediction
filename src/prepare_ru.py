from pathlib import Path
import pandas as pd
import os
from box import ConfigBox
from sklearn.model_selection import train_test_split
from ruamel.yaml import YAML

from prepare_func import add_district_and_salary, add_population, add_region, encode_amenities, encode_as_float, encode_city_center_distance, encode_infrastructure, encode_mortgage, encode_parking, encode_repair, encode_rooms, encode_terrace, encode_toilet, encode_transport, encode_tv_wifi, encode_utilities, encode_wall_material, process_floors, process_prices, process_year, remove_outliers, remove_unused, rename_columns

# Print current working directory
print(f"Current working directory: {os.getcwd()}")

# Use absolute path
script_dir = Path(__file__).parent.absolute()
path_ru_ads = script_dir.parent / 'data' / 'avito_data' / 'ru'

print(f"Looking for CSV files in: {path_ru_ads}")

# Check if directory exists
if not path_ru_ads.exists():
    raise FileNotFoundError(f"Directory not found: {path_ru_ads}")

# Get all CSV files in the 'msk' subdirectory
files_ru = list(path_ru_ads.glob('*.csv'))

print(f"Found {len(files_ru)} CSV files")
for f in files_ru:
    print(f"  - {f}")

if not files_ru:
    raise ValueError(f"No CSV files found in {path_ru_ads}")

dfs_ru = []
for f in files_ru:
    try:
        data = pd.read_csv(f)
        dfs_ru.append(data)
    except Exception as e:
        print(f"Error reading file {f}: {e}")

if not dfs_ru:
    raise ValueError("No data frames created. Check if CSV files are valid.")

df_original_ru = pd.concat(dfs_ru, ignore_index=True)
df_original_ru = df_original_ru[df_original_ru['Расстояние от МКАД'].isna()]
df_original_ru = df_original_ru[(df_original_ru['Область'] != 'Московская обл.') &
                                (df_original_ru['Город'] != 'г. Москва')]


df_working_ru = df_original_ru.copy()
df_working_ru = process_prices(df_working_ru)
df_working_ru = remove_unused(df_working_ru)

columns_to_encode_ru = ['Площ.дома', 'Площ.Участка']
df_working_ru = encode_as_float(df_working_ru, columns_to_encode_ru)
df_working_ru = remove_outliers(df_working_ru, columns_to_encode_ru)
df_working_ru = encode_toilet(df_working_ru)
df_working_ru = encode_city_center_distance(df_working_ru)


def process_dataframe(df):
    df = encode_amenities(df)
    df = encode_infrastructure(df)
    df = encode_tv_wifi(df)
    df = encode_rooms(df)
    df = encode_repair(df)
    df = encode_wall_material(df)
    df = encode_parking(df)
    df = encode_mortgage(df)
    df = encode_terrace(df)
    df = encode_transport(df)
    df = process_year(df)
    df = encode_utilities(df)
    df = process_floors(df)
    df = rename_columns(df)

    return df


df_working_ru = process_dataframe(df_working_ru)

df_working_ru = add_region(df_working_ru)
df_working_ru = add_district_and_salary(df_working_ru)
df_working_ru = add_population(df_working_ru)

print(df_working_ru.columns)

df_working_ru = df_working_ru.drop(
    ['Описание', 'Заголовок', 'Расстояние_от_МКАД'], axis=1)


cat_cols_ru = ['Город', 'Регион', 'Округ']

df_working_ru[cat_cols_ru] = df_working_ru[cat_cols_ru].astype('category')

output_dir = script_dir.parent / 'data' / 'prepared' / 'ru'
output_dir.mkdir(parents=True, exist_ok=True)


yaml = YAML(typ="safe")
params = ConfigBox(yaml.load(open("params.yaml", encoding="utf-8")))

random_seed = params.prepare_ru.random_seed
test_size = params.prepare_ru.test_size

train_data, test_data = train_test_split(
    df_working_ru, test_size=test_size, random_state=random_seed)

train_data.to_csv(output_dir / 'train.csv', index=False)
test_data.to_csv(output_dir / 'test.csv', index=False)
