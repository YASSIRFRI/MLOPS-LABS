# FastAPI Iris Prediction Example

This is the Iris prediction example using FastAPI and Pydantic.

## Installation

```bash
pip install fastapi uvicorn scikit-learn
```

## Train Model

First, run the iris.ipynb notebook to generate the model file:

```bash
# Copy the iris.ipynb from parent directory or run it
# This will create model_iris file
```

Or copy the model file:

```bash
copy "..\model_iris" .
```

## Run

```bash
python app.py
```

Or using uvicorn directly:

```bash
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

## Test

Visit:
- http://localhost:8080/ - Home page
- http://localhost:8080/docs - Interactive API documentation (Swagger UI)

### Test the Prediction

In the Swagger UI (http://localhost:8080/docs):

1. Click on POST `/make_predictions`
2. Click "Try it out"
3. Enter test values:
   ```json
   {
     "a": 5.1,
     "b": 3.5,
     "c": 1.4,
     "d": 0.2
   }
   ```
4. Click "Execute"
5. See the prediction result!

## Features

The model expects 4 features (Iris dataset):
- `a`: Sepal length (cm)
- `b`: Sepal width (cm)
- `c`: Petal length (cm)
- `d`: Petal width (cm)

## Output

The model predicts one of three Iris species:
- Setosa (0)
- Versicolor (1)
- Virginica (2)
