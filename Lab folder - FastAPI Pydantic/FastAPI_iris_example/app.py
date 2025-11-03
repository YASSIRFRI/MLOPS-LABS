from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle

app = FastAPI()


class iris(BaseModel):
    a: float
    b: float
    c: float
    d: float


# Load the model using pickle
try:
    model = pickle.load(open('model_iris', 'rb'))
except FileNotFoundError:
    print("Warning: Model file not found. Please run the iris.ipynb notebook first.")
    model = None


@app.get("/")
def home():
    return {'message': 'ML model for Iris prediction'}


@app.post('/make_predictions')
async def make_predictions(features: iris):
    if model is None:
        return {"error": "Model not loaded. Please train the model first."}

    prediction = model.predict([[features.a, features.b, features.c, features.d]])[0]
    species_names = ['Setosa', 'Versicolor', 'Virginica']
    species = species_names[int(prediction)] if int(prediction) < len(species_names) else str(prediction)

    return {"prediction": species, "prediction_value": str(prediction)}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
