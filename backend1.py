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
    
@app.post("/predict")
def predict(data:PlanetData):
    X=np.array([[data.orbital_period,data.transit_depth,data.transit_duration,data.signal_to_noise,data.insolation_flux]])
    X_scaled=scaler.transform(X)
    
    preds=model.predict(X_scaled)
    preds_class=np.argmax(preds,axis=1)
    label=encoder.inverse_transform(preds_class)[0]
    
    return{
        "prediction":label,
        "confidence":float(np.max(preds))
    }
