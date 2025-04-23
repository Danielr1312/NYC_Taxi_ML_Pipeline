from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from src.inference import (
    load_model,
    load_preprocessor,
    preprocess_input_data,
    make_predictions
)
from src.config import MODEL_PATH, PREPROCESSOR_PATH, API_KEY, WEATHER_DIR, DROP_COLUMNS

app = FastAPI()

# Load model and preprocessor on startup
model = load_model(MODEL_PATH)
preprocessor = load_preprocessor(PREPROCESSOR_PATH)

# Define request schema
class TaxiTrip(BaseModel):
    data: list  # list of dicts (each representing a row)

@app.post("/predict")
def predict_fare(trip: TaxiTrip):
    try:
        input_df = pd.DataFrame(trip.data)
        processed_data, _ = preprocess_input_data(
            input_df,
            preprocessor,
            API_KEY,
            WEATHER_DIR,
            DROP_COLUMNS
        )

        if processed_data is None:
            raise HTTPException(status_code=400, detail="No valid data left after cleaning. Check your input.")

        predictions = make_predictions(model, processed_data)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
