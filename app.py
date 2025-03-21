from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and scaler
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get sensor input data from request
    sensor_data = request.json
    features = np.array([sensor_data['pressure'], sensor_data['flow_rate'], sensor_data['temperature']])
    features_scaled = scaler.transform([features])

    # Make predictions
    prediction = model.predict(features_scaled)
    result = 'Leak Detected' if prediction[0] == 1 else 'No Leak'

    return jsonify({
        'status': result,
        'pipe_section': sensor_data.get('pipe_section', 'Unknown')
    })

if __name__ == '__main__':
    app.run(debug=True)
