from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
model = joblib.load('model/cancer_detection_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Convert data to DataFrame
    features = pd.DataFrame(data, index=[0])
    
    # Ensure all features are numeric
    features = features.apply(pd.to_numeric, errors='coerce')

    try:
        # Make prediction
        prediction = model.predict(features)
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
