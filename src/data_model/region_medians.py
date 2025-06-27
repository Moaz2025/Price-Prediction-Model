import pandas as pd
import joblib
from pathlib import Path

# Load clean dataset
df = pd.read_csv("datasets/clean/clean_data.csv")

# Optional: remove outliers
df = df[df['price'] <= df['price'].quantile(0.99)]

# Compute median price per region
region_medians = df.groupby('region')['price'].median().to_dict()

# Save to file
Path("models").mkdir(parents=True, exist_ok=True)
joblib.dump(region_medians, "models/region_medians.pkl")

print("Saved region_medians.pkl with", len(region_medians), "regions.")
