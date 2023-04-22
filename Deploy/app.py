from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("model.pkl")

# Set up the Flask route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json(force=True)

    # Convert the received data to a DataFrame
    input_data = pd.DataFrame(data)

    # Make predictions using the trained model
    predictions = model.predict(input_data)

    # Return the predictions as a JSON response
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(port=5000, debug=True)
