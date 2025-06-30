import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.ticker as mtick

# === Load trained model and region medians ===
model = joblib.load("models/ensemble_model.pkl")
region_medians = joblib.load("models/region_medians.pkl")

# Add region encoded feature (fallback to median if missing)
fallback_median = np.median(list(region_medians.values()))

# Load the clean dataset and apply the same feature engineering for evaluation
df = pd.read_csv("datasets/clean/clean_data.csv")
df = df[df['price'] <= df['price'].quantile(0.99)]

df['is_new_building'] = (df['building_year'] > 2010).astype(int)
df['age'] = 2025 - df['building_year']
df['density_ratio'] = df['rooms'] / (df['netHabitableSurface'] + 1e-3)
df['region_encoded'] = df['region'].map(region_medians).fillna(fallback_median)
df.drop(columns=['building_year'], inplace=True)

X_eval = df.drop(columns=['price'])
y_true = df['price']
y_pred_log = model.predict(X_eval)
y_pred = np.expm1(y_pred_log)

# === Evaluation metrics ===
r2 = r2_score(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print("\n=== Evaluation on Full Dataset ===")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:,.2f} EGP")

average_price = df["price"].mean()

print("=== Average Property Price ===")
print(f"Average Price: {average_price:,.2f} EGP")

median_price = df["price"].median()

print("=== Median Property Price ===")
print(f"Median Price: {median_price:,.2f} EGP")

# === Plot Predictions vs Actual ===
plt.figure(figsize=(10, 6))
plt.scatter(y_true, y_pred, alpha=0.3)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--r')
plt.xlabel("Actual Price (EGP)")
plt.ylabel("Predicted Price (EGP)")
plt.title("Predicted vs Actual Prices")
plt.grid(True)
plt.tight_layout()
plt.savefig("models/predicted_vs_actual.png")
plt.show()

# === Residuals Plot ===
residuals = y_true - y_pred

plt.figure(figsize=(10, 5))
plt.scatter(y_true, residuals, alpha=0.3, color='orange')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Actual Price (EGP)")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residuals vs Actual Prices")
plt.grid(True)
plt.tight_layout()
plt.savefig("models/residuals_vs_actual.png")
plt.show()


# === Histogram of Errors ===
plt.figure(figsize=(8, 4))
plt.hist(residuals, bins=50, color='teal', edgecolor='black')
plt.title("Distribution of Prediction Errors")
plt.xlabel("Error (Actual - Predicted)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig("models/error_distribution.png")
plt.show()


plt.figure(figsize=(8, 4))
plt.hist(y_pred, bins=50, color='seagreen', edgecolor='black')
plt.title("Distribution of Predicted Prices")
plt.xlabel("Predicted Price (EGP)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig("models/predicted_price_distribution.png")
plt.show()


# Add predictions to the dataframe
df['predicted_price'] = y_pred

# Boxplot by region (top 10 regions)
top_regions = df['region'].value_counts().head(10).index
subset = df[df['region'].isin(top_regions)]

plt.figure(figsize=(14, 6))
subset_sorted = subset.copy()
subset_sorted['region'] = pd.Categorical(subset_sorted['region'], categories=top_regions, ordered=True)
sns.boxplot(x='region', y='predicted_price', data=subset_sorted, color='skyblue')
plt.xticks(rotation=45)
plt.title("Predicted Prices by Region")
plt.ylabel("Predicted Price (EGP)")
plt.grid(True)
plt.tight_layout()
plt.savefig("models/predicted_price_by_region.png")
plt.show()


# === Relative Error (%)
relative_error = (residuals / y_true) * 100

plt.figure(figsize=(8, 4))
plt.hist(relative_error, bins=50, color='purple', edgecolor='black')
plt.title("Distribution of Relative Errors (%)")
plt.xlabel("Relative Error (%)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig("models/relative_error_distribution.png")
plt.show()
