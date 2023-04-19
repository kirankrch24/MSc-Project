import requests
import pandas as pd

data = {
    "DateCode": 2021,
    "FeatureName": "West Dunbartonshire",
    "Energy Type": "Coal",
    "Energy Consuming Sector": "Domestic",
}


df = pd.DataFrame([data])



# Perform One-Hot Encoding for 'FeatureName' (assuming you used it during training)
encoded_feature = pd.get_dummies(df["FeatureName"])
encoded_df = pd.concat([df.drop("FeatureName", axis=1), encoded_feature], axis=1)

response = requests.post("http://127.0.0.1:5000/predict", json=encoded_df.to_dict(orient="records"))


response_data = response.json()
print("Predicted energy consumption:", response_data.get("prediction", "Error: Could not parse prediction from server response"))


