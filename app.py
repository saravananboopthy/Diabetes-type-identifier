from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

app = FastAPI()

# Enable CORS for frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
MODEL_PATH = "diabetes_model.pkl"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

class PatientData(BaseModel):
    pregnancies: int
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    dpf: float # DiabetesPedigreeFunction
    age: int

@app.get("/")
async def read_index():
    return FileResponse("index.html")

@app.post("/predict")
async def predict(data: PatientData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not found. Please train the model first.")
    
    # Prepare data for prediction
    input_data = np.array([[
        data.pregnancies, data.glucose, data.blood_pressure,
        data.skin_thickness, data.insulin, data.bmi, data.dpf, data.age
    ]])
    
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    
    result_map = {0: "No Diabetes", 1: "Type 1 Diabetes", 2: "Type 2 Diabetes"}
    
    return {
        "prediction": int(prediction),
        "label": result_map[int(prediction)],
        "confidence": float(np.max(probabilities)),
        "probabilities": {
            "No Diabetes": float(probabilities[0]),
            "Type 1": float(probabilities[1]),
            "Type 2": float(probabilities[2])
        }
    }

# Serve static files if needed (though we'll keep it simple for now)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
