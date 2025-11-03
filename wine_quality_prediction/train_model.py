"""
Quick model training script for Wine Quality Prediction
Run this script to train the model without using Jupyter notebook
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def main():
    print("=" * 60)
    print("Wine Quality Prediction - Model Training")
    print("=" * 60)

    # Load dataset
    print("\n1. Loading dataset...")
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    df = pd.read_csv(url, sep=';')
    print(f"   Dataset shape: {df.shape}")
    print(f"   Features: {df.shape[1] - 1}")
    print(f"   Samples: {df.shape[0]}")

    # Check for missing values
    print("\n2. Checking data quality...")
    missing = df.isnull().sum().sum()
    print(f"   Missing values: {missing}")

    # Prepare data
    print("\n3. Preparing data...")
    X = df.drop('quality', axis=1)
    y = df['quality']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")

    # Define models
    print("\n4. Training models...")
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    results = []
    best_score = -np.inf
    best_model = None
    best_model_name = None

    # Train and evaluate each model
    for name, model in models.items():
        print(f"\n   Training {name}...")

        # Train
        model.fit(X_train, y_train)

        # Predictions
        y_pred_test = model.predict(X_test)

        # Metrics
        test_r2 = r2_score(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)

        results.append({
            'Model': name,
            'Test R²': test_r2,
            'Test RMSE': test_rmse,
            'Test MAE': test_mae
        })

        print(f"     Test R²: {test_r2:.4f}")
        print(f"     Test RMSE: {test_rmse:.4f}")
        print(f"     Test MAE: {test_mae:.4f}")

        # Track best model
        if test_r2 > best_score:
            best_score = test_r2
            best_model = model
            best_model_name = name

    # Display results
    print("\n" + "=" * 60)
    print("Model Comparison Results")
    print("=" * 60)
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Test R²', ascending=False)
    print(results_df.to_string(index=False))

    # Save best model
    print("\n" + "=" * 60)
    print(f"Best Model: {best_model_name}")
    print(f"Test R² Score: {best_score:.4f}")
    print("=" * 60)

    print("\n5. Saving model...")
    with open('wine_quality_model.pkl', 'wb') as file:
        pickle.dump(best_model, file)
    print("   Model saved as 'wine_quality_model.pkl'")

    # Test prediction
    print("\n6. Testing prediction...")
    sample = X_test.iloc[0:1]
    prediction = best_model.predict(sample)
    actual = y_test.iloc[0]
    print(f"   Sample prediction: {prediction[0]:.2f}")
    print(f"   Actual value: {actual}")
    print(f"   Difference: {abs(prediction[0] - actual):.2f}")

    print("\n" + "=" * 60)
    print("Training Complete! You can now run the Flask app.")
    print("Run: python app.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
