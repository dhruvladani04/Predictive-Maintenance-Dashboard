# Real-time Predictive Maintenance Dashboard

This is a complete, end-to-end machine learning web application that predicts machine failure in real-time. It simulates live IoT sensor data, processes it, and uses a trained classification model to determine if a machine's status is "HEALTHY" or "FAILURE IMMINENT". The results are displayed on a live, dynamic web dashboard.

## Key Features

- Real-time Data Simulation: A Python script simulates a live stream of sensor data (temperature, vibration, etc.) from an industrial machine.
- End-to-End ML Pipeline: The project covers the full data science lifecycle: data engineering, exploratory data analysis (EDA), model training, and evaluation.
- Machine Learning Model: A LightGBM classifier is trained to predict the likelihood of machine failure based on sensor patterns.
- Dynamic Web Dashboard: A Flask-based web application with a modern, ambient UI built with Tailwind CSS and Chart.js to visualize live data and predictions.
- Event Logging: Critical events like model predictions and system resets are automatically logged to a file (maintenance_log.log) for monitoring and auditing.

## Project Lifecycle

The project follows a standard machine learning systems design flow:

- Data Engineering (data_engineering/): A simulator was built to generate a realistic time-series dataset of sensor readings from a machine as its health degrades over time.
- Analysis & Preprocessing (analysis/): The generated data was analyzed (EDA) to understand sensor trends. A target variable (failure_imminent) was engineered for the classification task.
- Model Training (model/): A classification model was trained on the preprocessed data to learn the relationship between sensor readings and impending failure. The final model is saved to failure_classifier_model.joblib.
- Web Application (app.py, templates/): A Flask server loads the trained model and runs the simulator in real-time. It serves the data to a frontend dashboard that updates dynamically.

## How to Run the Project

You must run the scripts in the following order to generate the data and train the model before launching the web application.

1. Run the Exploratory Data Analysis:
    ```
    python analysis/eda_and_preprocessing.py
    ```
2. Train the Machine Learning Model:
    ```
    python model/train_model.py
    ```
3. Launch the Web Application:
    ```
    python app.py
    ```
4. View the Dashboard:
    Open your web browser and navigate to: http://127.0.0.1:5000
    You should now see the live dashboard in action!

## Dependencies

- Flask
- LightGBM
- Pandas
- Numpy
- Matplotlib
- Seaborn
- Joblib

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.