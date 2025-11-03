# FastAPI Basic Example

This is the basic FastAPI example from the lab.

## Installation

```bash
pip install fastapi uvicorn
```

## Run

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --reload
```

## Test

Visit:
- http://127.0.0.1:8000/ - Hello World
- http://127.0.0.1:8000/items/5?q=somequery - Test with parameters
- http://127.0.0.1:8000/docs - Interactive API documentation (Swagger UI)
- http://127.0.0.1:8000/redoc - Alternative API documentation

## Try Invalid Input

Visit http://127.0.0.1:8000/items/abc to see Pydantic validation in action!
