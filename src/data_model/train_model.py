import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import xgboost as xgb
import joblib


def preprocess_data(X_train):
    # numerical features
    numerical_features = X_train.select_dtypes(exclude=['object']).columns
    numerical_pipeline = Pipeline([
        ('minmax', MinMaxScaler())
    ])

    # categorical features
    categorical_features = X_train.select_dtypes(include=['object']).columns
    categorical_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
    ])

    transformer = ColumnTransformer([
        ("categorical", categorical_pipeline, categorical_features),
        ("numerical", numerical_pipeline, numerical_features)
    ])

    return transformer


def train_model(X_train, y_train):
    transformer = preprocess_data(X_train)

    model = Pipeline([
        ('preprocessing', transformer),
        ('xgboost', xgb.XGBRegressor(
            max_depth=3,
            n_estimators=300,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)
    return model


def eval_model(model, X, y):
    cv_results = cross_validate(model, X, y,
                                scoring=['neg_mean_squared_error', 'r2'],
                                return_train_score=True)

    print("Cross-validation results:")
    for key, values in cv_results.items():
        print(f"{key}: {values}")

    scores = cross_val_score(model, X, y, scoring='r2')
    print(f"\nMean R² score: {scores.mean():.2f} +/- {scores.std():.2f}")


def train_eval_save_model():
    clean_data_path = Path.cwd() / "datasets" / "clean" / "clean_data.csv"
    clean_df = pd.read_csv(clean_data_path)

    # Feature selection
    xgb_features = ['region', 'netHabitableSurface', 'rooms', 'bathrooms',
                    'finish_type', 'type', 'view', 'floor', 'building_year']
    clean_df = clean_df[xgb_features + ['price']]

    # Remove extreme price outliers (top 1%)
    upper_bound = clean_df['price'].quantile(0.99)
    clean_df = clean_df[clean_df['price'] <= upper_bound]

    # Log-transform the target
    clean_df['price'] = np.log1p(clean_df['price'])

    X = clean_df.drop('price', axis=1)
    y = clean_df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train)

    eval_model(model, X, y)

    model_path = Path.cwd() / "models" / "model.pkl"
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == '__main__':
    train_eval_save_model()
