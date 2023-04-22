from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the encoders
feature_name_encoder = pickle.load(open('feature_name_encoder.pkl', 'rb'))
energy_type_encoder = pickle.load(open('energy_type_encoder.pkl', 'rb'))
energy_consuming_sector_encoder = pickle.load(open('energy_consuming_sector_encoder.pkl', 'rb'))

# Load the trained model
model = pickle.load(open('linear_regression_model.pkl', 'rb'))


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    feature_name = feature_name_encoder.transform([data["FeatureName"]])[0]
    energy_type = energy_type_encoder.transform([data["Energy Type"]])[0]
    energy_consuming_sector = energy_consuming_sector_encoder.transform([data["Energy Consuming Sector"]])[0]
    
    input_data = np.array([[data['DateCode'], feature_name, energy_type, energy_consuming_sector]])
    
    prediction = model.predict(input_data)
    
    return jsonify({"Prediction": prediction[0]})


if __name__ == '__main__':
    app.run(debug=True)
