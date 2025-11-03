from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, validator
import uvicorn
import pickle
import numpy as np

app = FastAPI()

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")


# Pydantic model for wine quality features with validation
class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

    @validator('fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
               'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'sulphates', 'alcohol')
    def must_be_positive(cls, v, field):
        if v < 0:
            raise ValueError(f'{field.name} must be positive')
        return v

    @validator('pH')
    def pH_must_be_valid(cls, v):
        if v < 0 or v > 14:
            raise ValueError('pH must be between 0 and 14')
        return v

    @validator('density')
    def density_must_be_reasonable(cls, v):
        if v < 0.9 or v > 1.1:
            raise ValueError('Density must be between 0.9 and 1.1')
        return v

    @validator('alcohol')
    def alcohol_must_be_reasonable(cls, v):
        if v < 0 or v > 20:
            raise ValueError('Alcohol content must be between 0% and 20%')
        return v


# Load the trained model
try:
    model = pickle.load(open('wine_quality_model.pkl', 'rb'))
    print("Wine quality model loaded successfully!")
except FileNotFoundError:
    model = None
    print("Warning: Model file not found. Please train the model first.")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page with prediction form"""
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "prediction": None}
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict_form(
    request: Request,
    fixed_acidity: float = Form(...),
    volatile_acidity: float = Form(...),
    citric_acid: float = Form(...),
    residual_sugar: float = Form(...),
    chlorides: float = Form(...),
    free_sulfur_dioxide: float = Form(...),
    total_sulfur_dioxide: float = Form(...),
    density: float = Form(...),
    pH: float = Form(...),
    sulphates: float = Form(...),
    alcohol: float = Form(...)
):
    """Handle form submission and make prediction"""
    try:
        # Validate using Pydantic
        features = WineFeatures(
            fixed_acidity=fixed_acidity,
            volatile_acidity=volatile_acidity,
            citric_acid=citric_acid,
            residual_sugar=residual_sugar,
            chlorides=chlorides,
            free_sulfur_dioxide=free_sulfur_dioxide,
            total_sulfur_dioxide=total_sulfur_dioxide,
            density=density,
            pH=pH,
            sulphates=sulphates,
            alcohol=alcohol
        )

        if model is None:
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "prediction": None,
                    "error": "Model not loaded. Please train the model first."
                }
            )

        # Make prediction
        feature_array = np.array([[
            features.fixed_acidity,
            features.volatile_acidity,
            features.citric_acid,
            features.residual_sugar,
            features.chlorides,
            features.free_sulfur_dioxide,
            features.total_sulfur_dioxide,
            features.density,
            features.pH,
            features.sulphates,
            features.alcohol
        ]])

        prediction = model.predict(feature_array)[0]
        prediction = round(prediction, 1)

        # Determine quality rating
        if prediction < 5:
            quality_rating = "Poor"
            rating_emoji = "😞"
        elif prediction < 6:
            quality_rating = "Average"
            rating_emoji = "😐"
        elif prediction < 7:
            quality_rating = "Good"
            rating_emoji = "😊"
        else:
            quality_rating = "Excellent"
            rating_emoji = "🎉"

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "prediction": prediction,
                "quality_rating": quality_rating,
                "rating_emoji": rating_emoji,
                "features": features.dict()
            }
        )

    except ValueError as e:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "prediction": None,
                "error": f"Validation error: {str(e)}"
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "prediction": None,
                "error": f"Error: {str(e)}"
            }
        )


@app.post('/api/predict')
async def predict_api(features: WineFeatures):
    """API endpoint for predictions (JSON)"""
    if model is None:
        return {"error": "Model not loaded. Please train the model first."}

    try:
        # Make prediction
        feature_array = np.array([[
            features.fixed_acidity,
            features.volatile_acidity,
            features.citric_acid,
            features.residual_sugar,
            features.chlorides,
            features.free_sulfur_dioxide,
            features.total_sulfur_dioxide,
            features.density,
            features.pH,
            features.sulphates,
            features.alcohol
        ]])

        prediction = model.predict(feature_array)[0]
        prediction = round(prediction, 1)

        # Determine quality rating
        if prediction < 5:
            quality_rating = "Poor"
        elif prediction < 6:
            quality_rating = "Average"
        elif prediction < 7:
            quality_rating = "Good"
        else:
            quality_rating = "Excellent"

        return {
            "prediction": prediction,
            "quality_rating": quality_rating,
            "features": features.dict()
        }

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
