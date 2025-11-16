import json
import numpy as np
import pandas as pd
import joblib
import os
from azureml.core.model import Model

def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    global scaler

    try:
        # Get the path to the registered model file
        model_path = Model.get_model_path('wine_quality_model')
        print(f"Loading model from: {model_path}")

        # Load the model
        model = joblib.load(model_path)
        print("Model loaded successfully")

        # If you have a scaler, load it as well (optional)
        # scaler_path = Model.get_model_path('wine_quality_scaler')
        # scaler = joblib.load(scaler_path)

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    try:
        # Parse input data
        data = json.loads(raw_data)

        # Handle different input formats
        if isinstance(data, dict):
            if 'data' in data:
                # Format: {"data": [[...], [...]]}
                input_data = pd.DataFrame(data['data'])
            elif 'input' in data:
                # Format: {"input": [[...], [...]]}
                input_data = pd.DataFrame(data['input'])
            elif 'columns' in data and 'data' in data:
                # Pandas split format
                input_data = pd.DataFrame(data['data'], columns=data['columns'])
            else:
                # Assume it's a dict with feature names as keys
                input_data = pd.DataFrame([data])
        elif isinstance(data, list):
            # Format: [[...], [...]]
            input_data = pd.DataFrame(data)
        else:
            return json.dumps({"error": "Invalid input format"})

        print(f"Input shape: {input_data.shape}")

        # Make predictions
        predictions = model.predict(input_data)

        # If the model supports predict_proba
        try:
            probabilities = model.predict_proba(input_data)
            result = {
                'predictions': predictions.tolist(),
                'probabilities': probabilities.tolist()
            }
        except AttributeError:
            # Model doesn't support predict_proba (e.g., SVC without probability=True)
            result = {
                'predictions': predictions.tolist()
            }

        return json.dumps(result)

    except Exception as e:
        error_msg = str(e)
        print(f"Error during prediction: {error_msg}")
        return json.dumps({"error": error_msg})
