from fastapi import FastAPI
from pydantic import BaseModel
import os
import pickle

app = FastAPI(title="Trip Cost Predictor API")

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

def load_pickle(filename: str):
    path = os.path.join(ARTIFACT_DIR, filename)
    with open(path, "rb") as f:
        return pickle.load(f)

# Load model and encoders at startup
model = load_pickle("trip_total_cost_model.pkl")
district_encoder = load_pickle("district_encoder.pkl")
class_map = load_pickle("class_map.pkl")

class PredictRequest(BaseModel):
    district: str
    days: int
    adults: int
    children: int
    hotel_class: str
    transport_mode: str

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        district_encoded = district_encoder.transform([req.district])[0]
        hotel_class_encoded = class_map.get(req.hotel_class, req.hotel_class)

        X = [[
            district_encoded,
            req.days,
            req.adults,
            req.children,
            hotel_class_encoded,
            req.transport_mode
        ]]

        y = model.predict(X)[0]
        return {"predicted_total_cost_lkr": float(y)}
    except Exception as e:
        return {"error": str(e)}