from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import Literal
import uvicorn

# === Load model and region encoding ===
model = joblib.load("models/ensemble_model.pkl")
region_medians = joblib.load("models/region_medians.pkl")
fallback_median = np.median(list(region_medians.values()))

# === FastAPI app ===
app = FastAPI(title="Property Price Prediction")

# === Request schema ===
class PropertyRequest(BaseModel):
    region: str
    netHabitableSurface: float
    rooms: int
    bathrooms: int
    finish_type: str
    type: str
    view: str
    floor: int
    building_year: int

# === Enable CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
def predict_price(data: PropertyRequest):
    try:
        # Convert to DataFrame
        X_new = pd.DataFrame([data.dict()])

        # Preserve building_year
        building_year_temp = X_new['building_year'].copy()

        # Feature engineering
        X_new['is_new_building'] = (building_year_temp > 2010).astype(int)
        X_new['age'] = 2025 - building_year_temp
        X_new['density_ratio'] = X_new['rooms'] / (X_new['netHabitableSurface'] + 1e-3)
        X_new['region_encoded'] = X_new['region'].map(region_medians).fillna(fallback_median)
        X_new.drop(columns=['building_year'], inplace=True)

        # Predict
        log_price = model.predict(X_new)
        predicted_price = float(np.expm1(log_price)[0])

        return {int(predicted_price)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# === Run server on port 8081 ===
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8081, reload=True)
