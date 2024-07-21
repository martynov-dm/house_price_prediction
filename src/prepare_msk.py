from pathlib import Path
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from box import ConfigBox
from ruamel.yaml import YAML

from prepare_func import encode_amenities, encode_as_float, encode_infrastructure, encode_mortgage, encode_parking, encode_repair, encode_rooms, encode_terrace, encode_toilet, encode_transport, encode_tv_wifi, encode_utilities, encode_wall_material, process_floors, process_prices, process_year, remove_outliers, remove_unused, rename_columns

# Print current working directory
print(f"Current working directory: {os.getcwd()}")

# Use absolute path
script_dir = Path(__file__).parent.absolute()
path_msk_ads = script_dir.parent / 'data' / 'avito_data' / 'msk'

print(f"Looking for CSV files in: {path_msk_ads}")

# Check if directory exists
if not path_msk_ads.exists():
    raise FileNotFoundError(f"Directory not found: {path_msk_ads}")

# Get all CSV files in the 'msk' subdirectory
files_msk = list(path_msk_ads.glob('*.csv'))

print(f"Found {len(files_msk)} CSV files")
for f in files_msk:
    print(f"  - {f}")

if not files_msk:
    raise ValueError(f"No CSV files found in {path_msk_ads}")

dfs_msk = []
for f in files_msk:
    try:
        data = pd.read_csv(f)
        dfs_msk.append(data)
    except Exception as e:
        print(f"Error reading file {f}: {e}")

if not dfs_msk:
    raise ValueError("No data frames created. Check if CSV files are valid.")

df_original_msk = pd.concat(dfs_msk, ignore_index=True)

df_working_msk = df_original_msk.copy()

df_working_msk = process_prices(df_working_msk)
df_working_msk = remove_unused(df_working_msk)

columns_to_encode_msk = ['Площ.дома', 'Площ.Участка', 'Расстояние от МКАД']
df_working_msk = encode_as_float(df_working_msk, columns_to_encode_msk)
df_working_msk = remove_outliers(df_working_msk, columns_to_encode_msk)
df_working_msk = encode_toilet(df_working_msk)


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


df_working_msk = process_dataframe(df_working_msk)

df_working_msk = df_working_msk.drop(
    ['Описание', 'Заголовок', 'Адрес', 'Область'], axis=1)


cat_cols_msk = ['Город']

df_working_msk[cat_cols_msk] = df_working_msk[cat_cols_msk].astype('category')


output_dir = script_dir.parent / 'data' / 'prepared' / 'msk'
output_dir.mkdir(parents=True, exist_ok=True)


yaml = YAML(typ="safe")
params = ConfigBox(yaml.load(open("params.yaml", encoding="utf-8")))

random_seed = params.prepare_msk.random_seed
test_size = params.prepare_msk.test_size

train_data, test_data = train_test_split(
    df_working_msk, test_size=test_size, random_state=random_seed)

# Save the processed data
train_data.to_csv(output_dir / 'train.csv', index=False)
test_data.to_csv(output_dir / 'test.csv', index=False)
