# Wine Quality Prediction Web Application

A Flask-based web application that predicts wine quality using machine learning. This project was created as part of the CI3 Flask Lab assignment at UM6P.

## Project Overview

This application uses various machine learning algorithms to predict wine quality based on physicochemical properties. The model is trained on the UCI Wine Quality Dataset and deployed as a web application using Flask.

## Features

- **Multiple ML Models**: Compares Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, and Gradient Boosting
- **Best Model Selection**: Automatically selects the best performing model based on R² score
- **Interactive Web Interface**: User-friendly form to input wine characteristics
- **Real-time Predictions**: Get instant quality predictions (0-10 scale)
- **Heroku Deployment Ready**: Includes all necessary files for cloud deployment

## Dataset

- **Source**: UCI Machine Learning Repository
- **Type**: Red Wine Quality
- **Features**: 11 physicochemical properties
  - Fixed acidity
  - Volatile acidity
  - Citric acid
  - Residual sugar
  - Chlorides
  - Free sulfur dioxide
  - Total sulfur dioxide
  - Density
  - pH
  - Sulphates
  - Alcohol
- **Target**: Wine quality score (0-10)

## Project Structure

```
wine_quality_prediction/
│
├── app.py                              # Flask application
├── wine_quality_model_training.ipynb   # Jupyter notebook for model training
├── wine_quality_model.pkl              # Trained model (generated)
├── requirements.txt                    # Python dependencies
├── Procfile                            # Heroku deployment configuration
├── .gitignore                          # Git ignore file
├── README.md                           # This file
│
├── templates/
│   └── index.html                      # Main webpage
│
└── static/
    └── style.css                       # CSS styles (optional)
```

## Installation & Setup

### 1. Clone or Download the Project

```bash
cd wine_quality_prediction
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Train the Model

Open and run the Jupyter notebook:
```bash
jupyter notebook wine_quality_model_training.ipynb
```

Run all cells to:
- Download the dataset
- Perform data preprocessing
- Train multiple models
- Compare performance
- Save the best model as `wine_quality_model.pkl`

### 6. Run the Flask Application

```bash
python app.py
```

Visit `http://127.0.0.1:5000/` in your web browser.

## Usage

1. Open the web application in your browser
2. Enter values for all 11 wine characteristics
3. Click "Predict Quality"
4. View the predicted quality score and rating

## Model Performance

The notebook trains and compares 6 different models:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

The best model is automatically selected based on Test R² score.

## Deployment on Heroku

### Prerequisites
- Heroku account
- Heroku CLI installed
- Git installed

### Deployment Steps

1. **Login to Heroku**
   ```bash
   heroku login
   ```

2. **Create Heroku App**
   ```bash
   heroku create wine-quality-predictor-app --region eu
   ```

3. **Initialize Git Repository**
   ```bash
   git init
   ```

4. **Add Files to Git**
   ```bash
   git add .
   ```

5. **Commit Changes**
   ```bash
   git commit -am "initial commit"
   ```

6. **Deploy to Heroku**
   ```bash
   git push heroku master
   ```

7. **Open Application**
   ```bash
   heroku open
   ```

### View Your Apps
```bash
heroku apps
```

### View Logs (if needed)
```bash
heroku logs --tail
```

## Technologies Used

- **Backend**: Flask (Python web framework)
- **Machine Learning**: scikit-learn
- **Data Processing**: pandas, numpy
- **Deployment**: Gunicorn, Heroku
- **Frontend**: HTML5, CSS3

## Assignment Requirements Completed

✅ 1. Dataset selection (UCI Wine Quality Dataset)
✅ 2. Data preprocessing (missing values, feature engineering)
✅ 3. Feature relevance analysis (correlation matrix)
✅ 4. Interactive prediction form with all relevant features
✅ 5. Multiple ML models comparison
✅ 6. Best model selection and deployment
✅ 7. Web application with prediction functionality

## Future Improvements

- Add data visualization on the web interface
- Implement model retraining functionality
- Add white wine quality prediction
- Include confidence intervals for predictions
- Add API endpoints for programmatic access

## Author

Created for CI3 - College of Computing, UM6P
Professor: Fahd KALLOUBI
Year: 2025/2026

## License

This project is for educational purposes.

## References

- UCI Machine Learning Repository: Wine Quality Dataset
- Flask Documentation: https://flask.palletsprojects.com/
- scikit-learn Documentation: https://scikit-learn.org/
- Heroku Python Documentation: https://devcenter.heroku.com/categories/python-support
