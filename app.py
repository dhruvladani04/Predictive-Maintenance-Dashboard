# from flask import Flask, jsonify, render_template
# import joblib
# import pandas as pd
# import numpy as np
# import time

# # Initialize the Flask application
# app = Flask(__name__)

# # --- Machine Simulator ---
# # This class mimics the logic from our data engineering script to generate data in real-time.
# class MachineSimulator:
#     def __init__(self):
#         self.cycle = 0
#         self.base_temp = 25.0
#         self.base_vibration = 10.0
#         self.base_pressure = 100.0
#         self.base_rpm = 1500.0
#         self.health_score = 100.0
#         self.max_cycles = np.random.randint(500, 1000)

#     def get_next_reading(self):
#         """Generates the next set of sensor readings."""
#         if self.health_score > 0:
#             self.cycle += 1
#             # Health score degrades faster as it gets lower
#             degradation = 100 / self.max_cycles + (1 - self.health_score / 100) * 0.1
#             self.health_score -= degradation
#             self.health_score = max(self.health_score, 0)
#         else:
#             # Reset after failure to simulate a new machine
#             self.__init__()

#         # Sensor values drift and become more noisy as health degrades
#         health_factor = self.health_score / 100.0
#         temp_noise = np.random.normal(0, 0.5) * (2 - health_factor)
#         vibration_noise = np.random.normal(0, 0.2) * (2 - health_factor)
#         pressure_noise = np.random.normal(0, 1.0) * (2 - health_factor)
#         rpm_noise = np.random.normal(0, 5.0) * (2 - health_factor)

#         data = {
#             'timestamp': time.time(),
#             'cycle': self.cycle,
#             'health_score': round(self.health_score, 2),
#             'temperature_celsius': round(self.base_temp + (100 - self.health_score) * 0.5 + temp_noise, 2),
#             'vibration_hz': round(self.base_vibration + (100 - self.health_score) * 0.2 + vibration_noise, 2),
#             'pressure_psi': round(self.base_pressure + (100 - self.health_score) * 0.1 + pressure_noise, 2),
#             'rotation_rpm': round(self.base_rpm - (100 - self.health_score) * 3 + rpm_noise, 2)
#         }
#         return data

# # --- Model and Simulator Loading ---
# try:
#     model = joblib.load('failure_classifier_model.joblib')
#     print("[SUCCESS] Machine learning model loaded successfully.")
# except FileNotFoundError:
#     print("[ERROR] Model file 'failure_classifier_model.joblib' not found. Please train the model first.")
#     model = None

# # Create a single, persistent instance of our simulator
# simulator = MachineSimulator()

# # --- Flask Routes ---
# @app.route('/')
# def index():
#     """Renders the main dashboard page."""
#     return render_template('index.html')

# @app.route('/data')
# def get_data():
#     """
#     This is the API endpoint our dashboard will fetch data from.
#     It gets the next sensor reading, makes a prediction, and returns the data as JSON.
#     """
#     if not model:
#         return jsonify({"error": "Model not loaded"}), 500

#     # 1. Get live data from the simulator
#     live_data = simulator.get_next_reading()
    
#     # 2. Prepare data for the model
#     # The model expects a DataFrame with specific column names in a specific order.
#     features_for_model = [
#         'temperature_celsius', 
#         'vibration_hz', 
#         'pressure_psi', 
#         'rotation_rpm'
#     ]
#     input_df = pd.DataFrame([live_data], columns=features_for_model)
    
#     # 3. Make a prediction
#     prediction = model.predict(input_df)[0]
#     prediction_proba = model.predict_proba(input_df)[0]

#     # Map prediction to a human-readable status
#     status = "FAILURE IMMINENT" if prediction == 1 else "HEALTHY"
#     failure_probability = round(prediction_proba[1] * 100, 2)

#     # 4. Add prediction to our data payload
#     live_data['status'] = status
#     live_data['failure_probability'] = failure_probability
    
#     return jsonify(live_data)

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import time
import logging

# Initialize the Flask application
app = Flask(__name__)

# --- Logging Setup ---
# This configures the logger to write to a file named 'maintenance_log.log'.
# It will record messages of level INFO and above, with a timestamp.
logging.basicConfig(
    filename='maintenance_log.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Machine Simulator ---
# This class mimics the logic from our data engineering script to generate data in real-time.
class MachineSimulator:
    def __init__(self):
        self.cycle = 0
        self.base_temp = 25.0
        self.base_vibration = 10.0
        self.base_pressure = 100.0
        self.base_rpm = 1500.0
        self.health_score = 100.0
        # --- CHANGE: ACCELERATED LIFECYCLE ---
        # Instead of 500-1000 cycles, we'll set the max to 100-150.
        # This will make the machine degrade much faster for demonstration purposes.
        self.max_cycles = np.random.randint(100, 150)
        logging.info(f"New machine simulation started. Expected lifespan: ~{self.max_cycles} cycles.")

    def get_next_reading(self):
        """Generates the next set of sensor readings."""
        if self.health_score > 0:
            self.cycle += 1
            # Health score degrades faster as it gets lower
            degradation = 100 / self.max_cycles + (1 - self.health_score / 100) * 0.1
            self.health_score -= degradation
            self.health_score = max(self.health_score, 0)
        else:
            # Log the failure event before resetting
            logging.info(f"Machine simulation reached end of life at cycle {self.cycle}. Resetting.")
            # Reset after failure to simulate a new machine
            self.__init__()

        # Sensor values drift and become more noisy as health degrades
        health_factor = self.health_score / 100.0
        temp_noise = np.random.normal(0, 0.5) * (2 - health_factor)
        vibration_noise = np.random.normal(0, 0.2) * (2 - health_factor)
        pressure_noise = np.random.normal(0, 1.0) * (2 - health_factor)
        rpm_noise = np.random.normal(0, 5.0) * (2 - health_factor)

        data = {
            'timestamp': time.time(),
            'cycle': self.cycle,
            'health_score': round(self.health_score, 2),
            'temperature_celsius': round(self.base_temp + (100 - self.health_score) * 0.5 + temp_noise, 2),
            'vibration_hz': round(self.base_vibration + (100 - self.health_score) * 0.2 + vibration_noise, 2),
            'pressure_psi': round(self.base_pressure + (100 - self.health_score) * 0.1 + pressure_noise, 2),
            'rotation_rpm': round(self.base_rpm - (100 - self.health_score) * 3 + rpm_noise, 2)
        }
        return data

# --- Model and Simulator Loading ---
try:
    model = joblib.load('failure_classifier_model.joblib')
    success_msg = "[SUCCESS] Machine learning model loaded successfully."
    print(success_msg)
    logging.info(success_msg)
except FileNotFoundError:
    error_msg = "[ERROR] Model file 'failure_classifier_model.joblib' not found. App will run without predictions."
    print(error_msg)
    logging.error(error_msg)
    model = None

# Create a single, persistent instance of our simulator
simulator = MachineSimulator()

# --- Flask Routes ---
@app.route('/')
def index():
    """Renders the main dashboard page."""
    return render_template('index.html')

@app.route('/data')
def get_data():
    """
    This is the API endpoint our dashboard will fetch data from.
    It gets the next sensor reading, makes a prediction, and returns the data as JSON.
    """
    live_data = simulator.get_next_reading()
    
    # If the model failed to load, we can't make predictions.
    if not model:
        live_data['status'] = "DISCONNECTED"
        live_data['failure_probability'] = "N/A"
        return jsonify(live_data)

    # 1. Get live data from the simulator is already done
    
    # 2. Prepare data for the model
    features_for_model = [
        'temperature_celsius', 'vibration_hz', 'pressure_psi', 'rotation_rpm'
    ]
    input_df = pd.DataFrame([live_data], columns=features_for_model)
    
    # 3. Make a prediction
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]

    # Map prediction to a human-readable status
    status = "FAILURE IMMINENT" if prediction == 1 else "HEALTHY"
    failure_probability = round(prediction_proba[1] * 100, 2)

    # 4. Add prediction to our data payload
    live_data['status'] = status
    live_data['failure_probability'] = failure_probability
    
    # --- ADD LOGGING FOR FAILURE WARNING ---
    if status == "FAILURE IMMINENT":
        log_message = (f"FAILURE IMMINENT DETECTED | Cycle: {live_data['cycle']}, "
                       f"Probability: {failure_probability}%, Health: {live_data['health_score']}%")
        logging.warning(log_message)
    
    return jsonify(live_data)

if __name__ == '__main__':
    logging.info("Flask application starting up.")
    app.run(debug=True)