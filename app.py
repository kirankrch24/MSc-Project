from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the saved XGBoost model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input features from the request
    data = request.get_json(force=True)
    input_features = pd.DataFrame(data, index=[0])

    # Make predictions using the trained XGBoost model
    predictions = model.predict(input_features)

    # Return the predicted energy consumption value
    return jsonify(predictions[0])

if __name__ == '__main__':
    app.run(debug=True)
