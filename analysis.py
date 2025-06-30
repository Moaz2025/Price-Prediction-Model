import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import numpy as np

# Load the cleaned dataset
df = pd.read_csv("datasets/clean/clean_data.csv")

# === Descriptive Statistics ===
print("\n=== Summary Statistics ===")
print(df.describe().T[['mean', '50%', 'std', 'min', 'max']].rename(columns={'50%': 'median'}))

# === Distribution Plots ===
numerical_features = ['netHabitableSurface', 'rooms', 'bathrooms', 'floor', 'building_year']

# Prepare price data
prices = df['price']
bins = np.logspace(np.log10(prices.min() + 1), np.log10(prices.max() + 1), 50)

# Plot histogram
plt.figure(figsize=(10, 5))
sns.histplot(
    prices,
    bins=bins,
    kde=False,
    color='royalblue',
    edgecolor='black',
    alpha=1  # fully opaque
)

plt.xscale('log')
plt.xlabel("Price (EGP)")
plt.ylabel("Count")
plt.title("Distribution of Property Prices (Log-Spaced Bins)")

# Format x-axis to show prices in millions
plt.gca().xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))

plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("models/distribution_price_logbins_bold.png")
plt.show()

for feature in numerical_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[feature], kde=True, bins=50, color='skyblue')
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"models/distribution_{feature}.png")
    plt.show()

# === Price by Region (Top 10) ===
top_regions = df['region'].value_counts().nlargest(10).index
region_prices = df[df['region'].isin(top_regions)].groupby('region')['price'].median().sort_values()

plt.figure(figsize=(10, 6))
region_prices.plot(kind='barh', color='orange')
plt.title("Median Property Price by Region (Top 10)")
plt.xlabel("Median Price (EGP)")
plt.grid(True)
plt.tight_layout()
plt.savefig("models/median_price_by_region.png")
plt.show()
