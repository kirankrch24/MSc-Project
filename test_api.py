import requests
import json

url = "http://127.0.0.1:5000/predict"

data = {
    "FeatureName": "Total electricity supplied",
    "DateCode": 202005,
    "Energy Type": "Electricity",
    "Energy Consuming Sector": "Total final consumers",
}

headers = {'Content-type': 'application/json'}

response = requests.post(url, data=json.dumps(data), headers=headers)

print("Status code:", response.status_code)

if response.status_code == 200:
    print("Prediction:", response.json()['prediction'])
else:
    print("Error message:", response.text)
