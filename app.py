from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle

from flask_cors import CORS

app = Flask(__name__)
CORS(app)


# Load the model and the mappings
with open('xgb_model.pkl', 'rb') as model_file:
    xgb_model = pickle.load(model_file)

mappings_df = pd.read_csv('encoded_values.csv')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input features from the request
    input_data = request.json

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Apply label encoding using mappings
    for i in range(0, mappings_df.shape[1], 2):
        column_mapping = dict(zip(mappings_df.iloc[:, i + 1], mappings_df.iloc[:, i]))
        column_name = mappings_df.columns[i + 1]
        input_df[column_name] = input_df[column_name].map(column_mapping)

    # Make predictions
    prediction = xgb_model.predict(input_df)

    # Return the result as a JSON object
    return jsonify({'prediction': f"Predicted Energy usage will be {float(prediction[0]):.2f} GWh"})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
