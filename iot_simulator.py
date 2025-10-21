# This script simulates the data stream from our machine. It generates sensor readings that gradually "drift" and become more erratic as the machine's "health score" degrades over time.

# To get started, you can run this Python script on your local machine. You'll see JSON output printed to your console every two seconds, mimicking a real-time data feed.
import json
import random
import time
from datetime import datetime, timedelta

# --- Configuration ---
# Machine details
MACHINE_ID = "MACHINE-001"

# Sensor configurations (mean, standard deviation, drift per cycle)
SENSOR_CONFIG = {
    "temperature_celsius": {"mean": 80, "std_dev": 2, "drift": 0.01},
    "vibration_hz": {"mean": 50, "std_dev": 5, "drift": 0.05},
    "pressure_psi": {"mean": 100, "std_dev": 3, "drift": -0.02},
    "rotation_rpm": {"mean": 1500, "std_dev": 50, "drift": -0.1},
}

# Simulation parameters
INITIAL_HEALTH_SCORE = 100.0  # Starting health of the machine
HEALTH_DEGRADATION_RATE = 0.05 # How fast the machine degrades per cycle
FAILURE_THRESHOLD = 60       # Health score at which failure is imminent
SIMULATION_SPEED_SECONDS = 2 # Time between data points in seconds

# --- Data Simulation Functions ---

def get_initial_state():
    """Initializes the state of the machine simulation."""
    return {
        "machine_id": MACHINE_ID,
        "cycle": 0,
        "health_score": INITIAL_HEALTH_SCORE,
        "status": "Healthy",
        "sensors": {
            "temperature_celsius": SENSOR_CONFIG["temperature_celsius"]["mean"],
            "vibration_hz": SENSOR_CONFIG["vibration_hz"]["mean"],
            "pressure_psi": SENSOR_CONFIG["pressure_psi"]["mean"],
            "rotation_rpm": SENSOR_CONFIG["rotation_rpm"]["mean"],
        }
    }

def simulate_sensor_data(current_state):
    """
    Simulates the next set of sensor readings based on the current state.
    As the machine degrades (lower health score), the sensor readings become more erratic.
    """
    new_sensors = {}
    health_factor = (100 - current_state["health_score"]) / 100.0 # 0 is healthy, 1 is near failure

    for sensor, config in SENSOR_CONFIG.items():
        # Add random noise
        noise = random.normalvariate(0, config["std_dev"])

        # Add drift over time
        drift = config["drift"] * current_state["cycle"]

        # Add more significant noise as health degrades
        degradation_noise = random.normalvariate(0, config["std_dev"] * 2 * health_factor)

        # Calculate new value
        current_value = current_state["sensors"][sensor]
        new_value = current_value + drift + noise + degradation_noise
        new_sensors[sensor] = round(new_value, 2)

    return new_sensors


def update_machine_state(current_state):
    """Updates the overall state of the machine for the next cycle."""
    # Increment cycle
    current_state["cycle"] += 1

    # Update sensor data
    current_state["sensors"] = simulate_sensor_data(current_state)

    # Degrade health score (with some randomness)
    degradation = HEALTH_DEGRADATION_RATE * random.uniform(0.8, 1.2)
    current_state["health_score"] -= degradation
    current_state["health_score"] = max(current_state["health_score"], 0) # Cannot be less than 0

    # Update status based on health score
    if current_state["health_score"] < FAILURE_THRESHOLD:
        current_state["status"] = "Failure Imminent"
    elif current_state["health_score"] < (FAILURE_THRESHOLD + 20):
        current_state["status"] = "Warning"
    else:
        current_state["status"] = "Healthy"

    # Add timestamp
    current_state["timestamp"] = datetime.utcnow().isoformat() + "Z"

    return current_state


# --- Main Execution ---

if __name__ == "__main__":
    machine_state = get_initial_state()
    print("--- Starting Real-time Machine Data Simulation ---")
    print("Press Ctrl+C to stop.")
    print("-" * 50)

    try:
        while machine_state["health_score"] > 0:
            machine_state = update_machine_state(machine_state)

            # Convert to JSON to simulate a real-world data stream (e.g., to Kafka, MQTT, or a log file)
            data_json = json.dumps(machine_state, indent=4)
            print(data_json)
            print("-" * 50)

            # In a real system, you would send this JSON to a streaming platform.
            # Here, we just print it.

            # Stop if health is very low
            if machine_state["health_score"] <= (FAILURE_THRESHOLD - 10) and machine_state["status"] == "Failure Imminent":
                 print("Machine has failed. Simulation stopping.")
                 break

            time.sleep(SIMULATION_SPEED_SECONDS)

    except KeyboardInterrupt:
        print("\n--- Simulation stopped by user. ---")
