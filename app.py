from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
CORS(app,origins="*")

# Load the dataset
df = pd.read_csv('co2 Emissions.csv')

# Load the model
model = pickle.load(open('CO2_model.pkl', 'rb'))

@app.route('/',methods = ['GET'])
def home():
    makes = sorted(df["Make"].unique())
    models = sorted(df["Model"].unique())
    vehicle_classes = sorted(df["Vehicle Class"].unique())
    transmissions = sorted(df["Transmission"].unique())
    fuel_types = sorted(df["Fuel Type"].unique())
    return jsonify({
        'makes': makes,
        'models': models,
        'vehicle_classes': vehicle_classes,
        'transmissions': transmissions,
        'fuel_types': fuel_types
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    engine_size = float(data['engine_size'])
    cylinders = int(data['cylinders'])
    fuel_consumption = float(data['fuel_consumption'])

    features = np.array([[fuel_consumption, fuel_consumption, engine_size, cylinders]])
    prediction = model.predict(features)[0]
    
    return jsonify({'prediction': f'Predicted CO2 Emissions: {prediction:.2f} g/km'})

if __name__ == '__main__':
    app.run()
