from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load trained model
model = joblib.load("model.pkl")

# Define input schema
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Create FastAPI app
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Prediction API"}

@app.post("/predict")
def predict_species(features: IrisFeatures):
    data = np.array([[  # input must be 2D
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]])
    prediction = model.predict(data)[0]
    species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
    return {"prediction": species_map[prediction]}
