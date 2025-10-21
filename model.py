# import pandas as pd
# import lightgbm as lgb
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import joblib
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import os

# def train_and_evaluate_model(data_path='processed_machine_data.csv'):
#     """
#     Loads data, trains an RUL prediction model, evaluates it, and saves the model.
#     """
#     print("--- Starting Model Training and Evaluation ---")
    
#     # 1. Load Data
#     try:
#         df = pd.read_csv(data_path)
#     except FileNotFoundError:
#         print(f"[ERROR] The data file was not found at {data_path}.")
#         print("Please run the 'analysis/eda_and_preprocessing.py' script first to generate it.")
#         return

#     print(f"[INFO] Successfully loaded data with {df.shape[0]} rows and {df.shape[1]} columns.")

#     # 2. Feature Selection
#     # We will use the sensor data and the operational cycle as features.
#     # 'health_score' is also a very strong predictor.
#     features = [
#         'cycle', 
#         'health_score', 
#         'temperature_celsius', 
#         'vibration_hz', 
#         'pressure_psi', 
#         'rotation_rpm'
#     ]
#     target = 'RUL'
    
#     X = df[features]
#     y = df[target]

#     print(f"[INFO] Features used for training: {features}")
#     print(f"[INFO] Target variable: {target}")

#     # 3. Data Splitting
#     # We split the data into 80% for training and 20% for testing.
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )
#     print(f"[INFO] Training data shape: {X_train.shape}")
#     print(f"[INFO] Testing data shape: {X_test.shape}")

#     # 4. Model Training
#     print("\n[INFO] Training the LightGBM Regressor model...")
    
#     # Initialize and train the model
#     # LightGBM is a gradient boosting framework that uses tree-based learning algorithms.
#     # It is known for its speed and high performance.
#     model = lgb.LGBMRegressor(random_state=42)
#     model.fit(X_train, y_train)
    
#     print("[INFO] Model training complete.")

#     # 5. Model Evaluation
#     print("\n[INFO] Evaluating the model on the test set...")
    
#     y_pred = model.predict(X_test)
    
#     # Calculate metrics
#     mae = mean_absolute_error(y_test, y_pred)
#     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#     r2 = r2_score(y_test, y_pred)
    
#     print("-" * 30)
#     print("Model Performance Metrics:")
#     print(f"  - Mean Absolute Error (MAE): {mae:.2f}")
#     print(f"  - Root Mean Squared Error (RMSE): {rmse:.2f}")
#     print(f"  - R-squared (R²): {r2:.2f}")
#     print("-" * 30)

#     # 6. Visualization of Results
#     # Create plots directory if it doesn't exist
#     if not os.path.exists('plots'):
#         os.makedirs('plots')
        
#     plt.figure(figsize=(12, 8))
#     sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, label='Predictions')
#     plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2, label='Perfect Prediction')
#     plt.title('Actual RUL vs. Predicted RUL', fontsize=16)
#     plt.xlabel('Actual RUL (Cycles)', fontsize=12)
#     plt.ylabel('Predicted RUL (Cycles)', fontsize=12)
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('plots/actual_vs_predicted_rul.png')
#     print("\n[INFO] Saved prediction performance plot to 'plots/actual_vs_predicted_rul.png'")
    
#     # 7. Save the Model
#     model_filename = 'rul_model.joblib'
#     joblib.dump(model, model_filename)
#     print(f"[INFO] Trained model saved as '{model_filename}'")
    
#     print("\n--- Model training process finished successfully! ---")


# if __name__ == "__main__":
#     train_and_evaluate_model()

# import pandas as pd
# import lightgbm as lgb
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import joblib
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import os

# def train_and_evaluate_model(data_path='processed_machine_data.csv'):
#     """
#     Loads data, trains an RUL prediction model, evaluates it, and saves the model.
#     """
#     print("--- Starting Model Training and Evaluation ---")
    
#     # 1. Load Data
#     try:
#         df = pd.read_csv(data_path)
#     except FileNotFoundError:
#         print(f"[ERROR] The data file was not found at {data_path}.")
#         print("Please run the 'analysis/eda_and_preprocessing.py' script first to generate it.")
#         return

#     print(f"[INFO] Successfully loaded data with {df.shape[0]} rows and {df.shape[1]} columns.")

#     # 2. Feature Selection
#     # --- CHANGE 1: REMOVED 'health_score' ---
#     # The 'health_score' feature was causing data leakage and leading to an overfit model.
#     # A real-world model would only have access to raw sensor data and operational cycles.
#     features = [
#         'cycle', 
#         # 'health_score', # This is a "leaky" feature, so we remove it.
#         'temperature_celsius', 
#         'vibration_hz', 
#         'pressure_psi', 
#         'rotation_rpm'
#     ]
#     target = 'RUL'
    
#     X = df[features]
#     y = df[target]

#     print(f"[INFO] Features used for training: {features}")
#     print(f"[INFO] Target variable: {target}")

#     # 3. Data Splitting
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )
#     print(f"[INFO] Training data shape: {X_train.shape}")
#     print(f"[INFO] Testing data shape: {X_test.shape}")

#     # 4. Model Training
#     print("\n[INFO] Training the LightGBM Regressor model with regularization...")
    
#     # --- CHANGE 2: ADDED HYPERPARAMETERS FOR REGULARIZATION ---
#     # These parameters help prevent the model from overfitting.
#     # n_estimators: Number of boosting rounds.
#     # learning_rate: Step size shrinkage.
#     # reg_alpha (L1): Can help in feature selection.
#     # reg_lambda (L2): Helps prevent the model from becoming too complex.
#     model = lgb.LGBMRegressor(
#         random_state=42,
#         n_estimators=150,
#         learning_rate=0.1,
#         reg_alpha=0.1,
#         reg_lambda=0.1
#     )
#     model.fit(X_train, y_train)
    
#     print("[INFO] Model training complete.")

#     # 5. Model Evaluation
#     print("\n[INFO] Evaluating the model on the test set...")
    
#     y_pred = model.predict(X_test)
    
#     # Calculate metrics
#     mae = mean_absolute_error(y_test, y_pred)
#     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#     r2 = r2_score(y_test, y_pred)
    
#     print("-" * 30)
#     print("Model Performance Metrics (More Realistic):")
#     print(f"  - Mean Absolute Error (MAE): {mae:.2f}")
#     print(f"  - Root Mean Squared Error (RMSE): {rmse:.2f}")
#     print(f"  - R-squared (R²): {r2:.2f}")
#     print("-" * 30)

#     # 6. Visualization of Results
#     if not os.path.exists('plots'):
#         os.makedirs('plots')
        
#     plt.figure(figsize=(12, 8))
#     sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, label='Predictions')
#     plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2, label='Perfect Prediction')
#     plt.title('Actual RUL vs. Predicted RUL (Robust Model)', fontsize=16)
#     plt.xlabel('Actual RUL (Cycles)', fontsize=12)
#     plt.ylabel('Predicted RUL (Cycles)', fontsize=12)
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('plots/actual_vs_predicted_rul.png')
#     print("\n[INFO] Saved prediction performance plot to 'plots/actual_vs_predicted_rul.png'")
    
#     # 7. Save the Model
#     model_filename = 'rul_model.joblib'
#     joblib.dump(model, model_filename)
#     print(f"[INFO] Trained model saved as '{model_filename}'")
    
#     print("\n--- Model training process finished successfully! ---")


# if __name__ == "__main__":
#     train_and_evaluate_model()

# import pandas as pd
# import lightgbm as lgb
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import joblib
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import os

# def train_and_evaluate_model(data_path='processed_machine_data.csv'):
#     """
#     Loads data, trains an RUL prediction model, evaluates it, and saves the model.
#     """
#     print("--- Starting Model Training and Evaluation ---")
    
#     # 1. Load Data
#     try:
#         df = pd.read_csv(data_path)
#     except FileNotFoundError:
#         print(f"[ERROR] The data file was not found at {data_path}.")
#         print("Please run the 'analysis/eda_and_preprocessing.py' script first to generate it.")
#         return

#     print(f"[INFO] Successfully loaded data with {df.shape[0]} rows and {df.shape[1]} columns.")

#     # 2. Feature Selection
#     # --- CHANGE 1: REMOVED 'health_score' ---
#     # The 'health_score' feature was causing data leakage and leading to an overfit model.
#     # A real-world model would only have access to raw sensor data and operational cycles.
#     features = [
#         'cycle', 
#         # 'health_score', # This is a "leaky" feature, so we remove it.
#         'temperature_celsius', 
#         'vibration_hz', 
#         'pressure_psi', 
#         'rotation_rpm'
#     ]
#     target = 'RUL'
    
#     X = df[features]
#     y = df[target]

#     print(f"[INFO] Features used for training: {features}")
#     print(f"[INFO] Target variable: {target}")

#     # 3. Data Splitting
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )
#     print(f"[INFO] Training data shape: {X_train.shape}")
#     print(f"[INFO] Testing data shape: {X_test.shape}")

#     # 4. Model Training
#     print("\n[INFO] Training the LightGBM Regressor model with regularization...")
    
#     # --- CHANGE 2: ADDED HYPERPARAMETERS FOR REGULARIZATION ---
#     # These parameters help prevent the model from overfitting.
#     # n_estimators: Number of boosting rounds.
#     # learning_rate: Step size shrinkage.
#     # reg_alpha (L1): Can help in feature selection.
#     # reg_lambda (L2): Helps prevent the model from becoming too complex.
#     model = lgb.LGBMRegressor(
#         random_state=42,
#         n_estimators=150,
#         learning_rate=0.1,
#         reg_alpha=0.1,
#         reg_lambda=0.1
#     )
#     model.fit(X_train, y_train)
    
#     print("[INFO] Model training complete.")

#     # 5. Model Evaluation
#     print("\n[INFO] Evaluating the model on the test set...")
    
#     y_pred = model.predict(X_test)
    
#     # Calculate metrics
#     mae = mean_absolute_error(y_test, y_pred)
#     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#     r2 = r2_score(y_test, y_pred)
    
#     print("-" * 30)
#     print("Model Performance Metrics (More Realistic):")
#     print(f"  - Mean Absolute Error (MAE): {mae:.2f}")
#     print(f"  - Root Mean Squared Error (RMSE): {rmse:.2f}")
#     print(f"  - R-squared (R²): {r2:.2f}")
#     print("-" * 30)

#     # 6. Visualization of Results
#     if not os.path.exists('plots'):
#         os.makedirs('plots')
        
#     plt.figure(figsize=(12, 8))
#     sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, label='Predictions')
#     plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2, label='Perfect Prediction')
#     plt.title('Actual RUL vs. Predicted RUL (Robust Model)', fontsize=16)
#     plt.xlabel('Actual RUL (Cycles)', fontsize=12)
#     plt.ylabel('Predicted RUL (Cycles)', fontsize=12)
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('plots/actual_vs_predicted_rul.png')
#     print("\n[INFO] Saved prediction performance plot to 'plots/actual_vs_predicted_rul.png'")
    
#     # 7. Save the Model
#     model_filename = 'rul_model.joblib'
#     joblib.dump(model, model_filename)
#     print(f"[INFO] Trained model saved as '{model_filename}'")
    
#     print("\n--- Model training process finished successfully! ---")


# if __name__ == "__main__":
#     train_and_evaluate_model()



import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def train_and_evaluate_classifier(data_path='processed_machine_data.csv'):
    """
    Loads data, frames the problem as a classification task, trains a classifier,
    evaluates it, and saves the final model.
    """
    print("--- Reframing as Classification: Starting Model Training ---")
    
    # 1. Load Data
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"[ERROR] The data file was not found at {data_path}.")
        print("Please run the 'analysis/eda_and_preprocessing.py' script first.")
        return

    # --- CHANGE 1: PROBLEM REFRAMING (CLASSIFICATION) ---
    # We define a 'warning period'. If the RUL is within this period,
    # we classify it as 'imminent failure' (1). Otherwise, 'healthy' (0).
    WARNING_PERIOD_CYCLES = 30
    df['failure_imminent'] = np.where(df['RUL'] <= WARNING_PERIOD_CYCLES, 1, 0)
    
    print(f"[INFO] Created classification target 'failure_imminent' with warning period of {WARNING_PERIOD_CYCLES} cycles.")
    print(df['failure_imminent'].value_counts())

    # --- CHANGE 2: FINAL FEATURE SELECTION (NO DATA LEAKAGE) ---
    # We REMOVE 'cycle' to force the model to learn from sensor patterns alone.
    # This is the most realistic scenario.
    features = [
        'temperature_celsius', 
        'vibration_hz', 
        'pressure_psi', 
        'rotation_rpm'
    ]
    target = 'failure_imminent'
    
    X = df[features]
    y = df[target]

    print(f"[INFO] Features used for training: {features}")
    print(f"[INFO] Target variable: {target}")

    # 3. Data Splitting
    # 'stratify=y' ensures the train/test split has a similar proportion of 0s and 1s as the original dataset.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Model Training
    print("\n[INFO] Training the LightGBM Classifier model...")
    
    # We now use LGBMClassifier for our binary (0/1) prediction task.
    model = lgb.LGBMClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    print("[INFO] Model training complete.")

    # --- CHANGE 3: EVALUATION WITH CLASSIFICATION METRICS ---
    print("\n[INFO] Evaluating the classifier on the test set...")
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print("-" * 30)
    print("Classifier Performance Metrics:")
    print(f"  - Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("-" * 30)

    # --- CHANGE 4: VISUALIZATION WITH CONFUSION MATRIX ---
    if not os.path.exists('plots'):
        os.makedirs('plots')
        
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Healthy', 'Predicted Failure'], 
                yticklabels=['Actual Healthy', 'Actual Failure'])
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig('plots/confusion_matrix.png')
    print("\n[INFO] Saved confusion matrix plot to 'plots/confusion_matrix.png'")
    
    # 7. Save the Final Model
    model_filename = 'failure_classifier_model.joblib'
    joblib.dump(model, model_filename)
    print(f"[INFO] Trained classifier model saved as '{model_filename}'")
    
    print("\n--- Model training process finished successfully! ---")


if __name__ == "__main__":
    train_and_evaluate_classifier()