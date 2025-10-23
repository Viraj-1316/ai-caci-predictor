from flask import Flask, request, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
import joblib
import numpy as np
import os
import requests
import time
import json 

# --- 1. CONFIGURATION (Set as Environment Variables on the Server) ---
# IMPORTANT: When running LOCALLY (testing), you MUST define these variables 
# in your terminal or replace the os.environ.get() with your actual key strings temporarily.
# We will use os.environ.get() for secure CLOUD DEPLOYMENT.

TS_READ_KEY = os.environ.get("TS_READ_KEY", "YOUR_THINGSPEAK_READ_KEY") 
TS_WRITE_KEY = os.environ.get("TS_WRITE_KEY", "YOUR_THINGSPEAK_WRITE_KEY")
CHANNEL_ID = os.environ.get("CHANNEL_ID", "YOUR_CHANNEL_ID")

PREDICTION_INTERVAL_MINUTES = 7 # The frequency the prediction job runs

# --- 2. Load Model and Scalers (Executed once when server starts) ---
try:
    # Files must be present in the same directory as app.py on the server
    model = joblib.load('rf_caci_model.pkl')
    scaler_X = joblib.load('scaler_X.pkl')
    scaler_Y = joblib.load('scaler_Y.pkl')
    print("✅ Models and Scalers Loaded.")
except Exception as e:
    print(f"❌ ERROR loading ML assets: {e}. Prediction functionality disabled.")
    model = None

# --- 3. Flask Server Setup ---
app = Flask(__name__)

# --- 4. The Scheduled Job (The Automation Engine) ---
def scheduled_prediction_job():
    """
    This function runs every 7 minutes automatically. It fetches data, 
    calls the prediction function, and writes the result to ThingSpeak Field 6.
    """
    # Define where the internal prediction function is running (local host)
    port = os.environ.get('PORT', 5000)
    internal_api_url = f"http://127.0.0.1:{port}/predict_caci_internal"
    
    # Fetch the latest live sensor data (Fields 1-5) from ThingSpeak
    read_url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds/last.json?api_key={TS_READ_KEY}"
    try:
        response = requests.get(read_url, timeout=10).json()
        
        # Structure data for the prediction API call
        current_data = {
            # Note: Using .get() ensures it handles cases where a field might be missing (defaulting to 0)
            "CO2": float(response.get('field1', 0)),
            "Temp": float(response.get('field2', 0)),
            "Hum": float(response.get('field3', 0)),
            "AQI": float(response.get('field4', 0)),
            "CACI": float(response.get('field5', 0))
        }
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] ERROR: Failed to fetch latest data from TS. {e}")
        return

    # Call the internal prediction endpoint
    try:
        # Send the current sensor data to the prediction function
        prediction_response = requests.post(internal_api_url, json=current_data, timeout=10)
        
        if prediction_response.status_code == 200:
            predicted_caci = prediction_response.json()['predicted_caci_1hr']
            
            # Write the Prediction back to ThingSpeak (Field 6)
            write_url = f"https://api.thingspeak.com/update?api_key={TS_WRITE_KEY}&field6={predicted_caci}"
            requests.get(write_url)
            print(f"[{time.strftime('%H:%M:%S')}] ✅ CACI Forecasted ({predicted_caci:.2f}) and written to Field 6.")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] ❌ Internal API Call Failed. Status: {prediction_response.status_code}")
            
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] ❌ Critical Error during writing or API call: {e}")


# --- 5. Internal Prediction Endpoint (Runs the ML math) ---
@app.route('/predict_caci_internal', methods=['POST'])
def predict_caci_internal():
    """ Runs the actual ML model prediction logic. """
    if model is None: return jsonify({"error": "Model not loaded"}), 500
    try:
        data = request.json
        # The order of features MUST match the training phase: CO2, Temp, Hum, AQI, CACI
        input_data = np.array([[data['CO2'], data['Temp'], data['Hum'], data['AQI'], data['CACI']]])
        
        # 1. Scale Input Data
        input_scaled = scaler_X.transform(input_data)
        
        # 2. Predict using Random Forest model
        pred_scaled = model.predict(input_scaled)
        
        # 3. Inverse Transform to get actual CACI value (0-100)
        pred_actual = scaler_Y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
        
        return jsonify({
            "status": "success",
            "predicted_caci_1hr": round(float(pred_actual), 2)
        })
    except Exception as e:
        return jsonify({"error": str(e), "message": "Input data format is wrong."}), 400


# --- 6. Main Execution (Starts the Server and the Scheduler) ---
if __name__ == '__main__':
    # Initialize the scheduler
    scheduler = BackgroundScheduler()
    # Add the job to run the prediction every PREDICTION_INTERVAL_MINUTES
    scheduler.add_job(scheduled_prediction_job, 'interval', minutes=PREDICTION_INTERVAL_MINUTES)
    scheduler.start()

    # Start the Flask Web Service (use_reloader=False is critical to prevent duplicate schedulers)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, use_reloader=False)