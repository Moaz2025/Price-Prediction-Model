import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# === Load trained model and region medians ===
model = joblib.load("models/ensemble_model.pkl")
region_medians = joblib.load("models/region_medians.pkl")

# === Define input property ===
new_data = {
    'region': 'Faisal',
    'netHabitableSurface': 250,
    'rooms': 3,
    'bathrooms': 2,
    'finish_type': 'Super Lux',
    'type': 'Apartment',
    'view': 'Other',
    'floor': 6,
    'building_year': 2020
}

X_new = pd.DataFrame([new_data])

# Preserve building_year for feature engineering
building_year_temp = X_new['building_year'].copy()

# === Feature engineering (must match training phase) ===
X_new['is_new_building'] = (building_year_temp > 2010).astype(int)
X_new['age'] = 2025 - building_year_temp
X_new['density_ratio'] = X_new['rooms'] / (X_new['netHabitableSurface'] + 1e-3)

# Add region encoded feature (fallback to median if missing)
fallback_median = np.median(list(region_medians.values()))
X_new['region_encoded'] = X_new['region'].map(region_medians).fillna(fallback_median)

# Drop unused column
X_new.drop(columns=['building_year'], inplace=True)

# === Predict ===
log_price = model.predict(X_new)
predicted_price = np.expm1(log_price)[0]

print(f"Estimated Property Price: {predicted_price:,.0f} EGP")
