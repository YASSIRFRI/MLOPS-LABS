# FastAPI, Pydantic & Pytest Lab - Complete Implementation

**Course**: CS3 – College of Computing
**Professor**: Fahd KALLOUBI
**Year**: 2025/2026
**Lab**: MLOps - Python Libraries (FastAPI, Pydantic, Pytest)

---

## ✅ All Lab Requirements Implemented!

This folder contains **complete implementations** for all sections of the MLOps lab PDF:

1. ✅ **Pytest** - Testing framework with fixtures and parametrization
2. ✅ **Pydantic** - Data validation with BaseModel and validators
3. ✅ **FastAPI** - Modern web framework with auto-generated docs
4. ✅ **ML Deployments** - Iris, Furniture, and Wine Quality models

---

## 📦 Folder Structure

```
Lab folder - FastAPI Pydantic/
├── Pytest/                                   # Section 1: Pytest Examples
│   ├── my_module.py                         # Module to test
│   ├── test_my_module.py                    # Test with fixtures
│   ├── test_my_module_again.py              # Test with parametrize
│   └── conftest.py                          # Shared fixtures
│
├── Pydantic_examples/                        # Section 2: Pydantic Examples
│   ├── example1_basic.py                    # Basic BaseModel usage
│   ├── example2_recursive.py                # Recursive models
│   └── example3_validators.py               # Custom validators
│
├── FastAPI_basic_example/                    # Section 3: Basic FastAPI
│   ├── main.py                              # Basic routing example
│   └── README.md                            # Documentation
│
├── FastAPI_iris_example/                     # Iris ML Model
│   ├── app.py                               # FastAPI app for Iris
│   └── README.md                            # Documentation
│
├── FastAPI_furniture_prediction/             # To Do Task 1-3: Furniture
│   ├── app.py                               # FastAPI with Pydantic
│   ├── templates/                           # Jinja2 templates
│   │   └── index.html                       # Web interface
│   └── README.md                            # Documentation
│
├── FastAPI_wine_quality/                     # To Do Task 4: Wine Quality
│   ├── app.py                               # FastAPI with Pydantic
│   ├── templates/                           # Jinja2 templates
│   │   └── index.html                       # Web interface
│   └── README.md                            # Documentation
│
├── Furniture prediction notebook.ipynb       # Model training
├── iris.ipynb                               # Iris model training
├── furniture.csv                            # Dataset
├── model.pkl                                # Trained furniture model
└── README.md                                # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install pytest fastapi uvicorn pydantic jinja2 python-multipart scikit-learn numpy pandas
```

### 2. Test Pytest

```bash
cd Pytest
pytest
```

You should see tests passing!

### 3. Test Pydantic

```bash
cd Pydantic_examples
python example1_basic.py
python example2_recursive.py
python example3_validators.py
```

### 4. Test FastAPI Basic

```bash
cd FastAPI_basic_example
python main.py
```

Visit: http://localhost:8000/docs

### 5. Test ML Models

**Iris Prediction:**
```bash
cd FastAPI_iris_example
copy ..\model_iris .
python app.py
```
Visit: http://localhost:8080/docs

**Furniture Prediction:**
```bash
cd FastAPI_furniture_prediction
copy ..\model.pkl .
python app.py
```
Visit: http://localhost:8080/

**Wine Quality Prediction:**
```bash
cd FastAPI_wine_quality
copy ..\..\wine_quality_prediction\wine_quality_model.pkl .
python app.py
```
Visit: http://localhost:8080/

---

## 📚 Section 1: Pytest

### What's Implemented:

✅ Basic test with fixture
✅ Parametrized tests
✅ Shared fixtures in conftest.py

### Run Tests:

```bash
cd Pytest
pytest                    # Run all tests
pytest -v                 # Verbose output
pytest test_my_module.py  # Run specific file
```

### Expected Output:

```
============================= test session starts ==============================
collected 4 items

test_my_module.py .                                                       [ 25%]
test_my_module_again.py ...                                               [100%]

============================== 4 passed in 0.12s ===============================
```

### Key Concepts:

- **Fixtures**: Reusable test data (`@pytest.fixture`)
- **Parametrize**: Test multiple inputs (`@pytest.mark.parametrize`)
- **Conftest**: Shared fixtures across test files
- **Assertions**: `assert` statements for validation

---

## 📚 Section 2: Pydantic

### What's Implemented:

✅ Basic BaseModel usage
✅ Recursive models (models within models)
✅ Custom validators
✅ Type conversion
✅ Validation errors

### Run Examples:

```bash
cd Pydantic_examples

python example1_basic.py      # Basic usage & type conversion
python example2_recursive.py  # Nested models
python example3_validators.py # Custom validation
```

### Key Concepts:

- **BaseModel**: Base class for data models
- **Type Hints**: Automatic type validation
- **Optional**: Fields that can be None
- **Validators**: Custom validation logic (`@validator`)
- **Automatic Conversion**: String to int, etc.

---

## 📚 Section 3: FastAPI

### What's Implemented:

✅ Basic routing (`@app.get`, `@app.post`)
✅ Path parameters (`/items/{item_id}`)
✅ Query parameters (`?q=somequery`)
✅ Pydantic models for request bodies
✅ Auto-generated documentation (Swagger UI)
✅ Iris ML model deployment
✅ Furniture prediction with Jinja2 templates
✅ Wine Quality prediction with Jinja2 templates

### Key Features:

1. **Automatic Validation**: Pydantic validates all inputs
2. **Auto-Generated Docs**: Visit `/docs` for Swagger UI
3. **Type Safety**: Full type hints throughout
4. **Fast Performance**: ASGI server (Uvicorn)
5. **Modern Python**: async/await support

---

## 🎯 To Do Tasks (From Lab PDF)

### ✅ Task 1: Run Furniture Notebook

**Status**: ✅ Complete

The notebook is available and can be run to train the model:
- `Furniture prediction notebook.ipynb`
- Generates `model.pkl` file

### ✅ Task 2: Create FastAPI for Furniture Model

**Status**: ✅ Complete

**Location**: `FastAPI_furniture_prediction/`

**Features**:
- Pydantic model with 6 validated features
- Custom validators for ranges
- Both web form and JSON API
- Categorical encoding support

**Run**:
```bash
cd FastAPI_furniture_prediction
python app.py
```

### ✅ Task 3: Use Jinja2 Templates

**Status**: ✅ Complete

**Location**: `FastAPI_furniture_prediction/templates/`

**Features**:
- Beautiful, responsive HTML interface
- Form with dropdown menus
- Real-time validation
- Styled results display

### ✅ Task 4: Same for Wine Quality Dataset

**Status**: ✅ Complete

**Location**: `FastAPI_wine_quality/`

**Features**:
- All 11 wine features validated
- Custom validators for realistic ranges
- Quality rating system (Poor/Average/Good/Excellent)
- Organized form by property categories
- Both web and API interfaces

**Run**:
```bash
cd FastAPI_wine_quality
python app.py
```

---

## 🔥 Key Technologies

### Pytest
- **Purpose**: Testing framework
- **Why**: Simpler than unittest, powerful features
- **Key Feature**: Fixtures for test data reuse

### Pydantic
- **Purpose**: Data validation
- **Why**: Type-safe, automatic validation
- **Key Feature**: Converts types automatically

### FastAPI
- **Purpose**: Web API framework
- **Why**: Fast, modern, auto-documented
- **Key Features**:
  - Automatic OpenAPI documentation
  - Built-in Pydantic validation
  - ASGI for better performance
  - Type hints everywhere
  - Modern async support

---

## 📊 Comparison: Flask vs FastAPI

| Feature | Flask | FastAPI |
|---------|-------|---------|
| Server | WSGI | ASGI (faster) |
| Validation | Manual | Automatic (Pydantic) |
| Docs | Manual | Auto-generated |
| Type Hints | Optional | Required |
| Async | Plugin needed | Built-in |
| API Testing | External tool | Built-in Swagger UI |
| Error Messages | Generic | Detailed, helpful |
| Performance | Good | Better |

---

## 🎓 Learning Outcomes

By completing this lab, you've learned:

### Testing (Pytest)
- ✅ Write unit tests
- ✅ Use fixtures for reusable test data
- ✅ Parametrize tests for multiple inputs
- ✅ Organize tests with conftest.py

### Data Validation (Pydantic)
- ✅ Create data models with BaseModel
- ✅ Use type hints for automatic validation
- ✅ Write custom validators
- ✅ Handle validation errors

### Web APIs (FastAPI)
- ✅ Create REST APIs
- ✅ Handle GET and POST requests
- ✅ Validate request data with Pydantic
- ✅ Generate automatic API documentation
- ✅ Use Jinja2 templates
- ✅ Deploy ML models

### ML Deployment
- ✅ Load pickled models
- ✅ Create prediction endpoints
- ✅ Handle user input
- ✅ Return formatted results

---

## 🌟 Highlights

### Automatic API Documentation

Every FastAPI app automatically generates documentation:

- **Swagger UI**: `/docs` - Interactive testing interface
- **ReDoc**: `/redoc` - Alternative documentation
- **OpenAPI Schema**: `/openapi.json` - Machine-readable spec

### Data Validation

Pydantic validates:
- ✅ Types (int, float, str, etc.)
- ✅ Ranges (min/max values)
- ✅ Required vs optional fields
- ✅ Complex nested structures
- ✅ Custom business logic

### Error Handling

Automatic, detailed error messages:
```json
{
  "detail": [
    {
      "loc": ["body", "category"],
      "msg": "Category must be between 0 and 16",
      "type": "value_error"
    }
  ]
}
```

---

## 🧪 Testing the Implementations

### Test Pytest:
```bash
cd Pytest && pytest -v
```

### Test Basic FastAPI:
```bash
cd FastAPI_basic_example
python main.py
# Visit: http://localhost:8000/docs
# Try: http://localhost:8000/items/5?q=test
# Try: http://localhost:8000/items/abc (see validation error)
```

### Test Iris Model:
```bash
cd FastAPI_iris_example
python app.py
# Visit: http://localhost:8080/docs
# POST to /make_predictions with:
# {"a": 5.1, "b": 3.5, "c": 1.4, "d": 0.2}
```

### Test Furniture Prediction:
```bash
cd FastAPI_furniture_prediction
python app.py
# Visit: http://localhost:8080/
# Fill form and submit
```

### Test Wine Quality:
```bash
cd FastAPI_wine_quality
python app.py
# Visit: http://localhost:8080/
# Fill form with wine properties
```

---

## 📝 Assignment Completion Checklist

### Section 1: Pytest
- [x] Create my_module.py with function to test
- [x] Create test file with basic test
- [x] Add fixture for test data
- [x] Create parametrized test
- [x] Move fixture to conftest.py
- [x] Run all tests successfully

### Section 2: Pydantic
- [x] Basic BaseModel example
- [x] Type conversion example
- [x] Recursive models example
- [x] Custom validators example
- [x] Validation error handling

### Section 3: FastAPI
- [x] Basic routing example
- [x] Path parameters
- [x] Query parameters
- [x] Pydantic model integration
- [x] Auto-generated documentation

### To Do Tasks
- [x] Run Furniture prediction notebook
- [x] Create FastAPI app for Furniture model
- [x] Add Pydantic validation
- [x] Use Jinja2 templates for Furniture
- [x] Create FastAPI app for Wine Quality
- [x] Add Jinja2 templates for Wine Quality
- [x] Both web form and JSON API for each model

---

## 🎉 Success!

**All lab requirements are fully implemented and tested!**

You now have:
- ✅ Complete Pytest examples
- ✅ Complete Pydantic examples
- ✅ Complete FastAPI examples
- ✅ 3 ML model deployments (Iris, Furniture, Wine)
- ✅ Both web interfaces and JSON APIs
- ✅ Comprehensive documentation

---

## 📖 Additional Resources

### Official Documentation:
- **Pytest**: https://docs.pytest.org/
- **Pydantic**: https://pydantic-docs.helpmanual.io/
- **FastAPI**: https://fastapi.tiangolo.com/

### Tutorials:
- Pytest Tutorial: https://docs.pytest.org/en/latest/getting-started.html
- Pydantic Tutorial: https://pydantic-docs.helpmanual.io/usage/models/
- FastAPI Tutorial: https://fastapi.tiangolo.com/tutorial/

---

## 💡 Tips for Success

1. **Start with Simple Examples**: Run each example to understand the basics
2. **Read Error Messages**: Both Pydantic and FastAPI give helpful errors
3. **Use Swagger UI**: Test APIs interactively at `/docs`
4. **Check Validation**: Try invalid inputs to see validation in action
5. **Read Documentation**: Each folder has detailed README files

---

## 🚀 Next Steps

1. **Experiment**: Modify examples to understand concepts better
2. **Combine Skills**: Use Pytest to test your FastAPI endpoints
3. **Deploy**: Host your apps on Heroku, AWS, or other platforms
4. **Extend**: Add more features to the ML prediction apps
5. **Build**: Create your own FastAPI projects

---

## 📞 Support

All implementations are documented with:
- Detailed README files in each folder
- Inline code comments
- Usage examples
- Testing instructions

If you have issues:
1. Check the specific folder's README
2. Verify all dependencies are installed
3. Make sure you're in the correct directory
4. Check that model files are copied correctly

---

**Congratulations on completing the lab!** 🎊

You're now ready to build production-grade APIs with FastAPI, validate data with Pydantic, and write tests with Pytest!
