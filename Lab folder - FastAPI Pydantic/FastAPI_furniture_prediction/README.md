# FastAPI Furniture Price Prediction

This is a complete FastAPI implementation for furniture price prediction with:
- ✅ **Pydantic models** for data validation
- ✅ **Jinja2 templates** for web interface
- ✅ **Both web form and JSON API** endpoints

## Features

### Pydantic Validation
The `FurnitureFeatures` model includes:
- Type validation (int, float)
- Range validation (category: 0-16, boolean fields: 0-1)
- Custom validators for positive dimensions
- Automatic error messages

### Dual Interface
1. **Web Form** (`/`) - Beautiful HTML interface
2. **JSON API** (`/api/predict`) - RESTful API for programmatic access

## Installation

```bash
pip install fastapi uvicorn jinja2 python-multipart scikit-learn numpy
```

## Setup

1. Copy the trained model from parent directory:
```bash
copy "..\model.pkl" .
```

Or train the model using the Furniture prediction notebook.

2. Make sure you have the `templates/` folder with `index.html`

## Run

```bash
python app.py
```

Or using uvicorn:

```bash
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

## Usage

### Web Interface

Visit: http://localhost:8080/

Fill in the form:
- **Category**: Select furniture type (0-16)
- **Sellable Online**: 0 (No) or 1 (Yes)
- **Other Colors**: 0 (No) or 1 (Yes)
- **Depth**: Furniture depth in cm
- **Height**: Furniture height in cm
- **Width**: Furniture width in cm

Click "Predict Price" to get the result!

### API Documentation

Visit: http://localhost:8080/docs

Interactive Swagger UI for testing the API.

### JSON API Example

```bash
curl -X POST "http://localhost:8080/api/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "category": 5,
    "sellable_online": 1,
    "other_colors": 0,
    "depth": 40.0,
    "height": 75.0,
    "width": 80.0
  }'
```

Response:
```json
{
  "prediction": 1234.56,
  "category": "Chairs",
  "features": {
    "category": 5,
    "sellable_online": 1,
    "other_colors": 0,
    "depth": 40.0,
    "height": 75.0,
    "width": 80.0
  }
}
```

## Features Explained

### Input Features (6 features):

1. **category** (int: 0-16): Furniture category
   - 0: Bar furniture
   - 1: Beds
   - 2: Bookcases & shelving units
   - 3: Cabinets & cupboards
   - 4: Café furniture
   - 5: Chairs
   - 6: Chests of drawers & drawer units
   - 7: Children's furniture
   - 8: Nursery furniture
   - 9: Outdoor furniture
   - 10: Room dividers
   - 11: Sideboards, buffets & console tables
   - 12: Sofas & armchairs
   - 13: TV & media furniture
   - 14: Tables & desks
   - 15: Trolleys
   - 16: Wardrobes

2. **sellable_online** (int: 0 or 1): Can be sold online
   - 0: False (not available online)
   - 1: True (available online)

3. **other_colors** (int: 0 or 1): Available in other colors
   - 0: No (single color only)
   - 1: Yes (multiple colors)

4. **depth** (float > 0): Furniture depth in centimeters

5. **height** (float > 0): Furniture height in centimeters

6. **width** (float > 0): Furniture width in centimeters

### Output:
- **price** (float): Predicted price in dollars

## Pydantic Validators

The app includes custom validators:

```python
@validator('category')
def category_must_be_valid(cls, v):
    if v < 0 or v > 16:
        raise ValueError('Category must be between 0 and 16')
    return v

@validator('depth', 'height', 'width')
def dimensions_must_be_positive(cls, v):
    if v <= 0:
        raise ValueError('Dimensions must be positive numbers')
    return v
```

## Error Handling

The app handles:
- Invalid input values (Pydantic validation)
- Missing model file
- Prediction errors
- Form validation errors

All errors are displayed in a user-friendly format.

## Technologies

- **FastAPI**: Modern web framework
- **Pydantic**: Data validation
- **Jinja2**: Template engine
- **scikit-learn**: ML model
- **Uvicorn**: ASGI server

## Comparison with Flask

This FastAPI implementation demonstrates several advantages over Flask:

1. **Automatic validation** - Pydantic handles all input validation
2. **Auto-generated docs** - Swagger UI at `/docs`
3. **Type safety** - Type hints throughout
4. **Modern async** - Async/await support
5. **Better performance** - ASGI instead of WSGI

## Testing

Try these test cases:

### Valid Input:
```json
{
  "category": 5,
  "sellable_online": 1,
  "other_colors": 0,
  "depth": 50.0,
  "height": 85.0,
  "width": 60.0
}
```

### Invalid Category (should fail):
```json
{
  "category": 20,
  "sellable_online": 1,
  "other_colors": 0,
  "depth": 50.0,
  "height": 85.0,
  "width": 60.0
}
```

### Invalid Dimension (should fail):
```json
{
  "category": 5,
  "sellable_online": 1,
  "other_colors": 0,
  "depth": -10.0,
  "height": 85.0,
  "width": 60.0
}
```

## Assignment Completion

This implementation satisfies all requirements from the lab:

✅ Run Furniture prediction notebook to get model.pkl
✅ Produce an API for the model using FastAPI
✅ Use Pydantic for data validation
✅ Present results using Jinja2 template engine
✅ Support both web form and JSON API
