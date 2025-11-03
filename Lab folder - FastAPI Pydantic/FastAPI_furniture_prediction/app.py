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

# Pydantic model for furniture features with validation
class FurnitureFeatures(BaseModel):
    category: int
    sellable_online: int
    other_colors: int
    depth: float
    height: float
    width: float

    @validator('category')
    def category_must_be_valid(cls, v):
        if v < 0 or v > 16:
            raise ValueError('Category must be between 0 and 16')
        return v

    @validator('sellable_online')
    def sellable_online_must_be_valid(cls, v):
        if v not in [0, 1]:
            raise ValueError('Sellable online must be 0 (False) or 1 (True)')
        return v

    @validator('other_colors')
    def other_colors_must_be_valid(cls, v):
        if v not in [0, 1]:
            raise ValueError('Other colors must be 0 (No) or 1 (Yes)')
        return v

    @validator('depth', 'height', 'width')
    def dimensions_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Dimensions must be positive numbers')
        return v


# Load the trained model
try:
    model = pickle.load(open('model.pkl', 'rb'))
    print("Model loaded successfully!")
except FileNotFoundError:
    model = None
    print("Warning: Model file not found. Please run the notebook to generate model.pkl")


# Category mapping for display
CATEGORY_MAP = {
    0: 'Bar furniture', 1: 'Beds', 2: 'Bookcases & shelving units',
    3: 'Cabinets & cupboards', 4: 'Café furniture', 5: 'Chairs',
    6: 'Chests of drawers & drawer units', 7: "Children's furniture",
    8: 'Nursery furniture', 9: 'Outdoor furniture', 10: 'Room dividers',
    11: 'Sideboards, buffets & console tables', 12: 'Sofas & armchairs',
    13: 'TV & media furniture', 14: 'Tables & desks', 15: 'Trolleys', 16: 'Wardrobes'
}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page with prediction form"""
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "categories": CATEGORY_MAP, "prediction": None}
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict_form(
    request: Request,
    category: int = Form(...),
    sellable_online: int = Form(...),
    other_colors: int = Form(...),
    depth: float = Form(...),
    height: float = Form(...),
    width: float = Form(...)
):
    """Handle form submission and make prediction"""
    try:
        # Validate using Pydantic
        features = FurnitureFeatures(
            category=category,
            sellable_online=sellable_online,
            other_colors=other_colors,
            depth=depth,
            height=height,
            width=width
        )

        if model is None:
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "categories": CATEGORY_MAP,
                    "prediction": None,
                    "error": "Model not loaded. Please train the model first."
                }
            )

        # Make prediction
        feature_array = np.array([[
            features.category,
            features.sellable_online,
            features.other_colors,
            features.depth,
            features.height,
            features.width
        ]])

        prediction = model.predict(feature_array)[0]
        prediction = round(prediction, 2)

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "categories": CATEGORY_MAP,
                "prediction": prediction,
                "category_name": CATEGORY_MAP.get(features.category, "Unknown"),
                "features": features.dict()
            }
        )

    except ValueError as e:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "categories": CATEGORY_MAP,
                "prediction": None,
                "error": f"Validation error: {str(e)}"
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "categories": CATEGORY_MAP,
                "prediction": None,
                "error": f"Error: {str(e)}"
            }
        )


@app.post('/api/predict')
async def predict_api(features: FurnitureFeatures):
    """API endpoint for predictions (JSON)"""
    if model is None:
        return {"error": "Model not loaded. Please train the model first."}

    try:
        # Make prediction
        feature_array = np.array([[
            features.category,
            features.sellable_online,
            features.other_colors,
            features.depth,
            features.height,
            features.width
        ]])

        prediction = model.predict(feature_array)[0]
        prediction = round(prediction, 2)

        return {
            "prediction": prediction,
            "category": CATEGORY_MAP.get(features.category, "Unknown"),
            "features": features.dict()
        }

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
