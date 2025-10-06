import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from fastapi.middleware.cors import CORSMiddleware

app=FastAPI()

# Allow requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"] if React runs there
    allow_credentials=True,
    allow_methods=["*"],  # includes OPTIONS, POST, etc.
    allow_headers=["*"],
)



model= load_model("exoplanet_model.h5")
scaler=joblib.load("scaler.pkl")
encoder=joblib.load("encoder.pkl")


class PlanetData(BaseModel):
    orbital_period:float
    transit_depth:float
    transit_duration:float
    signal_to_noise:float
    insolation_flux:float

@app.get("/")
def root():
    return {"status": "running"}

    
@app.post("/predict")
def predict(data: PlanetData):
    print("✅ Step 1: Received request:", data.dict())

    try:
        X = np.array([[data.orbital_period, data.transit_depth,
                       data.transit_duration, data.signal_to_noise,
                       data.insolation_flux]])
        print("✅ Step 2: Input array:", X)

        X_scaled = scaler.transform(X)
        print("✅ Step 3: Scaled input:", X_scaled)

        preds = model.predict(X_scaled, verbose=0)
        print("✅ Step 4: Model predicted:", preds)

        preds_class = np.argmax(preds, axis=1)
        print("✅ Step 5: Pred class:", preds_class)

        label = encoder.inverse_transform(preds_class)[0]
        print("✅ Step 6: Label:", label)

        return {
            "prediction": label,
            "confidence": float(np.max(preds))
        }

    except Exception as e:
        print("❌ Error:", str(e))
        return {"error": str(e)}
