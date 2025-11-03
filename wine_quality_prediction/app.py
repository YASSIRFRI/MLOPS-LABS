import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the trained model
try:
    model = pickle.load(open('wine_quality_model.pkl', 'rb'))
except FileNotFoundError:
    model = None
    print("Warning: Model file not found. Please train the model first by running the notebook.")

@app.route('/')
def index():
    """Render the home page with the prediction form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests from the form
    """
    if model is None:
        return render_template('index.html',
                             prediction_text='Error: Model not loaded. Please train the model first.')

    try:
        # Get form data and convert to float
        features = request.form.to_dict()
        features_list = list(features.values())
        features_list = list(map(float, features_list))

        # Reshape for prediction (1 sample, 11 features)
        final_features = np.array(features_list).reshape(1, -1)

        # Make prediction
        prediction = model.predict(final_features)

        # Round the output to 1 decimal place
        output = round(prediction[0], 1)

        # Determine quality rating
        if output < 5:
            quality_rating = "Poor"
        elif output < 6:
            quality_rating = "Average"
        elif output < 7:
            quality_rating = "Good"
        else:
            quality_rating = "Excellent"

        return render_template('index.html',
                             prediction_text=f'Predicted Wine Quality: {output}/10 ({quality_rating})')

    except ValueError:
        return render_template('index.html',
                             prediction_text='Error: Please enter valid numbers for all fields.')
    except Exception as e:
        return render_template('index.html',
                             prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
