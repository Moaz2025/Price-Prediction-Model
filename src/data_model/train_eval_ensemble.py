import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.ensemble import VotingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# === Load and prepare data ===
df = pd.read_csv("datasets/clean/clean_data.csv")
df = df[df['price'] <= df['price'].quantile(0.99)]

# === Feature engineering ===
df['is_new_building'] = (df['building_year'] > 2010).astype(int)
df['age'] = 2025 - df['building_year']
df['density_ratio'] = df['rooms'] / (df['netHabitableSurface'] + 1e-3)
region_medians = df.groupby('region')['price'].median().to_dict()
df['region_encoded'] = df['region'].map(region_medians)

# === Save region encoding for offline use ===
Path("models").mkdir(parents=True, exist_ok=True)
joblib.dump(region_medians, "models/region_medians.pkl")

# === Target and feature selection ===
y = np.log1p(df['price'])
X = df.drop(columns=['price', 'building_year'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Preprocessing ===
numerical_features = X.select_dtypes(exclude=['object']).columns
categorical_features = X.select_dtypes(include=['object']).columns

numerical_pipeline = Pipeline([('minmax', MinMaxScaler())])
categorical_pipeline = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

preprocessor = ColumnTransformer([
    ("cat", categorical_pipeline, categorical_features),
    ("num", numerical_pipeline, numerical_features)
])

# === Regressors ===
xgb = XGBRegressor(tree_method='hist', random_state=42)
rf = RandomForestRegressor(random_state=42)
hgb = HistGradientBoostingRegressor(random_state=42)

ensemble = VotingRegressor([
    ('xgb', xgb),
    ('rf', rf),
    ('hgb', hgb)
])

pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('ensemble', ensemble)
])

# === Hyperparameter optimization ===
search_space = {
    'ensemble__xgb__n_estimators': Integer(100, 500),
    'ensemble__xgb__max_depth': Integer(3, 7),
    'ensemble__xgb__learning_rate': Real(0.01, 0.2, prior='log-uniform'),
    'ensemble__xgb__subsample': Real(0.5, 1.0),
    'ensemble__xgb__colsample_bytree': Real(0.5, 1.0),
    'ensemble__xgb__reg_lambda': Real(0, 5),
    'ensemble__xgb__reg_alpha': Real(0, 5),
}

opt = BayesSearchCV(
    estimator=pipeline,
    search_spaces=search_space,
    n_iter=20,
    cv=3,
    scoring='r2',
    verbose=1,
    n_jobs=-1,
    random_state=42
)

# === Train ===
opt.fit(X_train, y_train)

# === Evaluate ===
y_pred = np.expm1(opt.predict(X_test))
y_true = np.expm1(y_test)
r2 = r2_score(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

# === Save model ===
joblib.dump(opt.best_estimator_, "models/ensemble_model.pkl")

print("\n\nFinal Model Trained and Saved!")
print(f"Best CV R²: {opt.best_score_:.4f}")
print(f"Test R²: {r2:.4f}")
print(f"Test RMSE: {rmse:,.2f} EGP")
print(f"Best Parameters: {opt.best_params_}")
