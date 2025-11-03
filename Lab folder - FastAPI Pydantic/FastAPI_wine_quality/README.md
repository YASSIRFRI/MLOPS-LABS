# FastAPI Wine Quality Prediction

Complete FastAPI implementation for wine quality prediction based on the dataset from the previous Flask assignment.

## Features

- ✅ **FastAPI** web framework
- ✅ **Pydantic validation** for all 11 wine features
- ✅ **Jinja2 templates** for beautiful web interface
- ✅ **Dual API** - Web form and JSON endpoints
- ✅ **Custom validators** for realistic wine properties
- ✅ **Quality ratings** - Poor, Average, Good, Excellent

## Installation

```bash
pip install fastapi uvicorn jinja2 python-multipart scikit-learn numpy pandas
```

## Setup

Copy the trained model from the wine_quality_prediction folder:

```bash
copy "..\..\wine_quality_prediction\wine_quality_model.pkl" .
```

Or train a new model using the wine quality dataset.

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

Fill in all 11 wine properties and get instant quality prediction!

### API Documentation

Visit: http://localhost:8080/docs

Interactive Swagger UI with all endpoints documented.

### JSON API

```bash
curl -X POST "http://localhost:8080/api/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.70,
    "citric_acid": 0.00,
    "residual_sugar": 1.9,
    "chlorides": 0.076,
    "free_sulfur_dioxide": 11,
    "total_sulfur_dioxide": 34,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4
  }'
```

Response:
```json
{
  "prediction": 5.6,
  "quality_rating": "Average",
  "features": {
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.7,
    ...
  }
}
```

## Features Explained

### Input Features (11 features):

1. **fixed_acidity** (float > 0): Most acids involved with wine (g/dm³)
   - Typical range: 4-16

2. **volatile_acidity** (float > 0): Acetic acid content (g/dm³)
   - Too high = vinegar taste
   - Typical range: 0.1-1.6

3. **citric_acid** (float > 0): Adds freshness and flavor (g/dm³)
   - Typical range: 0-1

4. **residual_sugar** (float > 0): Sugar remaining after fermentation (g/dm³)
   - Typical range: 0.9-15

5. **chlorides** (float > 0): Salt amount in wine (g/dm³)
   - Typical range: 0.01-0.6

6. **free_sulfur_dioxide** (float > 0): Prevents microbial growth (mg/dm³)
   - Typical range: 1-72

7. **total_sulfur_dioxide** (float > 0): Total SO2 (mg/dm³)
   - Typical range: 6-290

8. **density** (0.9 - 1.1): Depends on alcohol and sugar (g/cm³)
   - Typical range: 0.99-1.00

9. **pH** (0 - 14): Acidity level
   - Typical range: 2.7-4.0 (wines are acidic)

10. **sulphates** (float > 0): Wine additive (g/dm³)
    - Typical range: 0.3-2

11. **alcohol** (0 - 20%): Alcohol percentage
    - Typical range: 8-15%

### Output:

- **quality** (0-10): Wine quality score
  - 0-4: Poor 😞
  - 5: Average 😐
  - 6: Good 😊
  - 7-10: Excellent 🎉

## Pydantic Validators

### Custom Validation Rules:

```python
@validator('fixed_acidity', 'volatile_acidity', ...)
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
```

## Test Cases

### Good Quality Wine:
```json
{
  "fixed_acidity": 8.5,
  "volatile_acidity": 0.4,
  "citric_acid": 0.3,
  "residual_sugar": 2.5,
  "chlorides": 0.08,
  "free_sulfur_dioxide": 20,
  "total_sulfur_dioxide": 60,
  "density": 0.995,
  "pH": 3.3,
  "sulphates": 0.7,
  "alcohol": 11.0
}
```

### Poor Quality (High Volatile Acidity):
```json
{
  "fixed_acidity": 7.0,
  "volatile_acidity": 1.2,
  "citric_acid": 0.0,
  "residual_sugar": 2.0,
  "chlorides": 0.1,
  "free_sulfur_dioxide": 10,
  "total_sulfur_dioxide": 30,
  "density": 0.998,
  "pH": 3.5,
  "sulphates": 0.5,
  "alcohol": 9.0
}
```

### Invalid Input (Negative pH):
```json
{
  "fixed_acidity": 7.4,
  "volatile_acidity": 0.7,
  "citric_acid": 0.0,
  "residual_sugar": 1.9,
  "chlorides": 0.076,
  "free_sulfur_dioxide": 11,
  "total_sulfur_dioxide": 34,
  "density": 0.9978,
  "pH": -1.0,
  "sulphates": 0.56,
  "alcohol": 9.4
}
```
*This will be rejected by Pydantic validation!*

## Comparison: Flask vs FastAPI

| Feature | Flask Version | FastAPI Version |
|---------|--------------|-----------------|
| Data Validation | Manual checks | Pydantic (automatic) |
| Type Safety | None | Full type hints |
| API Docs | Manual | Auto-generated |
| Input Validation | Custom code | Declarative validators |
| Error Messages | Generic | Detailed, helpful |
| Async Support | No | Yes |
| Performance | Good | Better (ASGI) |

## Technologies Used

- **FastAPI**: Modern, fast web framework
- **Pydantic**: Data validation and settings management
- **Jinja2**: Template engine (same as Flask)
- **scikit-learn**: Machine learning library
- **NumPy**: Numerical computing
- **Uvicorn**: Lightning-fast ASGI server

## Assignment Completion

This implementation completes task #4 from the lab PDF:

✅ Task 1: Run Furniture prediction notebook ✓
✅ Task 2: Create FastAPI for Furniture model ✓
✅ Task 3: Use Jinja2 templates ✓
✅ Task 4: Do the same for Wine Quality dataset ✓

## Benefits of This Implementation

1. **Type Safety**: All inputs are type-checked
2. **Validation**: Realistic ranges for wine properties
3. **User-Friendly**: Beautiful, responsive interface
4. **Flexible**: Both web form and JSON API
5. **Well-Documented**: Auto-generated API docs
6. **Production-Ready**: Error handling, validation, logging

## Next Steps

1. Add more sophisticated ML models
2. Implement feature importance visualization
3. Add batch prediction capability
4. Create data visualization charts
5. Deploy to cloud platform (Heroku, AWS, etc.)
