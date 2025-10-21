import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import random
from datetime import datetime

# --- Section 1: Data Generation ---
# We'll adapt the core logic from the iot_simulator.py to generate a complete
# historical dataset from a machine's start until failure.

# --- Configuration (from iot_simulator.py) ---
MACHINE_ID = "MACHINE-001"
SENSOR_CONFIG = {
    "temperature_celsius": {"mean": 80, "std_dev": 2, "drift": 0.01},
    "vibration_hz": {"mean": 50, "std_dev": 5, "drift": 0.05},
    "pressure_psi": {"mean": 100, "std_dev": 3, "drift": -0.02},
    "rotation_rpm": {"mean": 1500, "std_dev": 50, "drift": -0.1},
}
INITIAL_HEALTH_SCORE = 100.0
HEALTH_DEGRADATION_RATE = 0.05
FAILURE_THRESHOLD = 60

# --- Simulation Functions (adapted for batch generation) ---
def get_initial_state():
    """Initializes the state of the machine simulation."""
    return {
        "machine_id": MACHINE_ID,
        "cycle": 0,
        "health_score": INITIAL_HEALTH_SCORE,
        "status": "Healthy",
        "sensors": {
            key: config["mean"] for key, config in SENSOR_CONFIG.items()
        }
    }

def simulate_sensor_data(current_state):
    """Simulates the next set of sensor readings."""
    new_sensors = {}
    health_factor = (100 - current_state["health_score"]) / 100.0
    for sensor, config in SENSOR_CONFIG.items():
        noise = random.normalvariate(0, config["std_dev"])
        drift = config["drift"] * current_state["cycle"]
        degradation_noise = random.normalvariate(0, config["std_dev"] * 2 * health_factor)
        current_value = SENSOR_CONFIG[sensor]['mean'] # Start from mean to avoid compounding errors in batch generation
        new_value = current_value + drift + noise + degradation_noise
        new_sensors[sensor] = round(new_value, 2)
    return new_sensors

def update_machine_state(current_state):
    """Updates the machine state for the next cycle."""
    current_state["cycle"] += 1
    current_state["sensors"] = simulate_sensor_data(current_state)
    degradation = HEALTH_DEGRADATION_RATE * random.uniform(0.8, 1.2)
    current_state["health_score"] -= degradation
    current_state["health_score"] = max(current_state["health_score"], 0)
    if current_state["health_score"] < FAILURE_THRESHOLD:
        current_state["status"] = "Failure Imminent"
    elif current_state["health_score"] < (FAILURE_THRESHOLD + 20):
        current_state["status"] = "Warning"
    else:
        current_state["status"] = "Healthy"
    current_state["timestamp"] = datetime.utcnow().isoformat() + "Z"
    return current_state

def generate_historical_data(filepath="machine_data.csv"):
    """Generates and saves a complete historical dataset for one machine."""
    print("--- Generating historical machine data... ---")
    machine_state = get_initial_state()
    history = []

    while machine_state["health_score"] > 0:
        machine_state = update_machine_state(machine_state)
        # Flatten the dictionary for CSV
        flat_state = {
            'machine_id': machine_state['machine_id'],
            'cycle': machine_state['cycle'],
            'health_score': machine_state['health_score'],
            'status': machine_state['status'],
            'timestamp': machine_state['timestamp']
        }
        for sensor, value in machine_state['sensors'].items():
            flat_state[sensor] = value
        history.append(flat_state)
        
        if machine_state["health_score"] <= (FAILURE_THRESHOLD - 20):
            break # Stop well after failure point

    df = pd.DataFrame(history)
    df.to_csv(filepath, index=False)
    print(f"--- Data saved to {filepath} ---")
    return df

# --- Section 2: EDA and Preprocessing ---

def perform_eda(df):
    """Performs EDA on the machine dataset."""
    print("\n--- Starting Exploratory Data Analysis ---")
    
    # Create output directory for plots
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # 1. Basic Information
    print("\n[INFO] Dataset Information:")
    df.info()
    
    print("\n[INFO] Statistical Summary:")
    print(df.describe())
    
    # 2. Feature Engineering: Calculate Remaining Useful Life (RUL)
    # RUL is the number of cycles left before the next failure.
    # We define failure as the cycle where health_score first drops below the threshold.
    failure_cycle = df[df['health_score'] < FAILURE_THRESHOLD]['cycle'].min()
    if pd.isna(failure_cycle):
        print("[WARNING] No failure point found in the dataset. RUL cannot be calculated.")
        df['RUL'] = 0
    else:
        print(f"[INFO] Failure detected at cycle: {failure_cycle}")
        df['RUL'] = failure_cycle - df['cycle']
        # Set RUL for cycles after failure to 0
        df.loc[df['cycle'] >= failure_cycle, 'RUL'] = 0
    
    print("\n[INFO] Dataset with RUL (Target Variable):")
    print(df.head())
    
    # 3. Visualizations
    print("\n[INFO] Generating plots...")
    
    # Plot sensor data and health score over time (cycles)
    sensor_columns = list(SENSOR_CONFIG.keys())
    plot_columns = sensor_columns + ['health_score', 'RUL']
    
    fig, axes = plt.subplots(len(plot_columns), 1, figsize=(15, 5 * len(plot_columns)), sharex=True)
    for i, col in enumerate(plot_columns):
        sns.lineplot(x='cycle', y=col, data=df, ax=axes[i], color='royalblue' if col != 'RUL' else 'darkred')
        axes[i].set_title(f'{col} over Time (Cycles)', fontsize=14)
        axes[i].set_ylabel(col)
        if 'failure_cycle' in locals() and pd.notna(failure_cycle):
            axes[i].axvline(x=failure_cycle, color='r', linestyle='--', label=f'Failure Point ({failure_cycle} cycles)')
        axes[i].legend()
        axes[i].grid(True, linestyle='--', alpha=0.6)
    
    plt.xlabel('Cycle')
    plt.tight_layout()
    plt.savefig('plots/time_series_plots.png')
    print(" - Saved time_series_plots.png")
    
    # Correlation Heatmap
    plt.figure(figsize=(12, 8))
    corr = df[sensor_columns + ['health_score', 'RUL']].corr()
    sns.heatmap(corr, annot=True, cmap='viridis', fmt='.2f')
    plt.title('Correlation Matrix of Sensors, Health Score, and RUL', fontsize=16)
    plt.savefig('plots/correlation_heatmap.png')
    print(" - Saved correlation_heatmap.png")
    
    print("\n--- EDA Complete. Check the 'plots' directory for visualizations. ---")
    
    # Save the processed data with the RUL column
    df.to_csv('processed_machine_data.csv', index=False)
    print("--- Processed data with RUL saved to processed_machine_data.csv ---")


if __name__ == "__main__":
    # Check if data exists, otherwise generate it
    if os.path.exists('machine_data.csv'):
        print("--- Loading existing historical data. ---")
        main_df = pd.read_csv('machine_data.csv')
    else:
        main_df = generate_historical_data()
        
    perform_eda(main_df)
