import pandas as pd
import numpy as np
from pathlib import Path

def first_clean_data_egypt(df):
    print(f"Raw data shape: {df.shape}")

    mask = df['title'].str.contains('apartment', case=False, na=False) | df['type'].str.contains('apartment', case=False, na=False)
    df = df[mask]

    df['type'] = "Apartment"

    df.rename(columns={
        'building year': 'building_year',
        'finish_type': 'finish_type',
        'governorate': 'region',
        'size': 'netHabitableSurface'
    }, inplace=True)

    df.replace(["N/A", "", "nan", "NaN", "لا يوجد"], np.nan, inplace=True)

    df.drop_duplicates(inplace=True)

    df.drop(columns=['id', 'URL', 'title', 'Status', 'location'], inplace=True, errors='ignore')

    numeric_cols = ['price', 'netHabitableSurface', 'rooms', 'bathrooms', 'floor', 'building_year']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df[df['price'] >= 150000]    

    df.dropna(subset=['price', 'netHabitableSurface', 'region'], inplace=True)

    df = df[~df['type'].str.contains('Land', na=False)]

    df['finish_type'] = df['finish_type'].fillna('Lux')
    df['view'] = df['view'].fillna('Other')

    for col in ['rooms', 'bathrooms', 'floor', 'building_year']:
        df[col] = df[col].fillna(df[col].mean(numeric_only=True)).round().astype(int)

    print(f"After initial clean: {df.shape}")
    return df

def clean_data_egypt(df):
    df_cleaned = first_clean_data_egypt(df)

    output_dir = Path.cwd() / "datasets" / "clean"
    output_dir.mkdir(parents=True, exist_ok=True)

    df_cleaned.to_csv(output_dir / "clean_data_egypt.csv", index=False)
    df_cleaned.to_pickle(output_dir / "clean_data_egypt.pkl")

    return df_cleaned


raw_data_path = Path.cwd() / "datasets" / "raw" / "properties_data.csv"

df_raw = pd.read_csv(raw_data_path)
print("=== Raw Data ===")
print(df_raw.head(), "\n")

df_cleaned = first_clean_data_egypt(df_raw.copy())
print("=== After First Clean ===")
print(df_cleaned.head(), "\n")

output_dir = Path.cwd() / "datasets" / "clean"
output_dir.mkdir(parents=True, exist_ok=True)

df_cleaned.to_csv(output_dir / "clean_data_egypt.csv", index=False)
df_cleaned.to_pickle(output_dir / "clean_data_egypt.pkl")

print(f"✅ Cleaned data saved to: {output_dir}")


raw_data_path = Path.cwd() / "datasets" / "clean" / "clean_data_egypt.csv"
df = pd.read_csv(raw_data_path)

columns_to_check = ['type', 'region', 'finish_type', 'view', 'floor', 'rooms', 'building_year', 'bathrooms']

for col in columns_to_check:
    print(f"\n=== Value counts for '{col}' ===")
    print(df[col].value_counts(dropna=False))
