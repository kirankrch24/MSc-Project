from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the trained model
model = joblib.load("model.pkl")

# Initialize the encoders (use the same encoders from the training script)
energy_type_encoder = LabelEncoder()
energy_consuming_sector_encoder = LabelEncoder()

# Fit the encoders (use the same categories from the training script)
energy_type_encoder.fit(["Coal", "Electricity", "Gas", "Bioenergy & Wastes", "Petroleum Products", "Manufactured Fuels"])
energy_consuming_sector_encoder.fit(["Rail", "Domestic", "Public Sector", "Industrial & Commercial", "Agriculture", "Road Transport", "Commercial", "Industrial"])

@app.route("/predict", methods=["POST"])
def predict():
    input_data = request.get_json()

    # Preprocess the input features before making predictions
    input_data["Energy Type"] = energy_type_encoder.transform([input_data.get("Energy Type")])[0]
    input_data["Energy Consuming Sector"] = energy_consuming_sector_encoder.transform([input_data.get("Energy Consuming Sector")])[0]


    # Convert input data to a DataFrame
    input_df = pd.DataFrame([input_data])

    # Make the prediction
    prediction = model.predict(input_df)[0]

    # Return the prediction as JSON
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)
