import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score

# === Load trained model and region medians ===
model = joblib.load("models/ensemble_model.pkl")
region_medians = joblib.load("models/region_medians.pkl")
fallback_median = np.median(list(region_medians.values()))

# === Load data and apply preprocessing ===
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

# === Metrics ===
r2 = r2_score(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
avg_price = y_true.mean()
median_price = y_true.median()

print("\n=== Evaluation on Full Dataset ===")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:,.2f} EGP")
print(f"Average Price: {avg_price:,.2f} EGP")
print(f"Median Price: {median_price:,.2f} EGP")

# === Formatter for axes (millions) ===
formatter = mtick.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M")

# === 1. Predicted vs Actual ===
plt.figure(figsize=(10, 6))
plt.scatter(y_true, y_pred, alpha=0.3, color='dodgerblue', edgecolor='k')
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--r')
plt.xlabel("Actual Price (EGP)")
plt.ylabel("Predicted Price (EGP)")
plt.title("Predicted vs Actual Prices")
plt.gca().xaxis.set_major_formatter(formatter)
plt.gca().yaxis.set_major_formatter(formatter)
plt.grid(True)
plt.tight_layout()
plt.savefig("models/predicted_vs_actual.png")
plt.show()

# === 2. Histogram of Prediction Errors ===
errors = y_true - y_pred
plt.figure(figsize=(10, 5))
plt.hist(errors, bins=50, color='orangered', edgecolor='black')
plt.xlabel("Prediction Error (Actual - Predicted)")
plt.ylabel("Count")
plt.title("Distribution of Prediction Errors")
plt.gca().xaxis.set_major_formatter(formatter)
plt.grid(True)
plt.tight_layout()
plt.savefig("models/prediction_error_histogram.png")
plt.show()

# === 3. Histogram of Actual vs Predicted Prices ===
plt.figure(figsize=(10, 5))
plt.hist(y_true, bins=50, alpha=0.6, label="Actual", color='skyblue', edgecolor='black')
plt.hist(y_pred, bins=50, alpha=0.6, label="Predicted", color='green', edgecolor='black')
plt.xlabel("Price (EGP)")
plt.ylabel("Count")
plt.title("Actual vs Predicted Price Distribution")
plt.gca().xaxis.set_major_formatter(formatter)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("models/actual_vs_predicted_histogram.png")
plt.show()

# === 4. Residuals vs Actual Price ===
residuals = y_true - y_pred

plt.figure(figsize=(10, 5))
plt.scatter(y_true, residuals, alpha=0.3, color='purple', edgecolor='k')
plt.axhline(0, linestyle='--', color='red')
plt.xlabel("Actual Price (EGP)")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residuals vs Actual Prices")
plt.gca().xaxis.set_major_formatter(formatter)
plt.grid(True)
plt.tight_layout()
plt.savefig("models/residuals_vs_actual.png")
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(df['netHabitableSurface'], bins=50, color='teal', edgecolor='black')
plt.xlabel("Area (m²)")
plt.ylabel("Count")
plt.title("Distribution of Net Habitable Surface Area")
plt.grid(True)
plt.tight_layout()
plt.savefig("models/distribution_area.png")
plt.show()

top_regions = df['region'].value_counts().nlargest(10).index
df_top = df[df['region'].isin(top_regions)]

plt.figure(figsize=(12, 6))
df_top.boxplot(column='price', by='region', vert=False)
plt.title("Price Distribution by Region (Top 10)")
plt.suptitle("")
plt.xlabel("Price (EGP)")
plt.ylabel("Region")
plt.gca().xaxis.set_major_formatter(formatter)
plt.tight_layout()
plt.savefig("models/boxplot_region_price.png")
plt.show()
