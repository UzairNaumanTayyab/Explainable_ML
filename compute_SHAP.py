# -----------------------
# compute_shap.py
# -----------------------
# Purpose:
#   Compute SHAP values for a sample of 1000 rows from the diabetes dataset.
#   Save precomputed SHAP values, expected value, and sample rows to .pkl files.
#   These files will later be used in a Streamlit dashboard for fast visualization.
# -----------------------

import pandas as pd
import numpy as np
import joblib
import shap
import tensorflow as tf

# -----------------------
# STEP 1: LOAD MODEL
# -----------------------
# Use either .keras or .h5 model
# If both exist, choose the one you trained last
model_path = "diabetes_model.keras"  # change to .h5 if you prefer
model = tf.keras.models.load_model(model_path)

# -----------------------
# STEP 2: LOAD SCALER
# -----------------------
scaler = joblib.load("scaler.pkl")  # StandardScaler used during training

# -----------------------
# STEP 3: LOAD DATASET
# -----------------------
df = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2021.csv")

# Drop target column to get features
X = df.drop("Diabetes_binary", axis=1)

# -----------------------
# STEP 4: SAMPLE 1000 ROWS
# -----------------------
# For performance, we explain only 1000 rows
X_sample = X.sample(n=1000, random_state=42)

# -----------------------
# STEP 5: SCALE THE DATA
# -----------------------
scaled_sample = scaler.transform(X_sample.values)

# -----------------------
# STEP 6: CREATE BACKGROUND DATA FOR SHAP
# -----------------------
# Background is a small subset of rows used by KernelExplainer
# We use 100 rows randomly selected
background = scaled_sample[np.random.choice(scaled_sample.shape[0], 100, replace=False)]

# -----------------------
# STEP 7: CREATE SHAP EXPLAINER
# -----------------------
# KernelExplainer works for any model, including neural networks
explainer = shap.KernelExplainer(model.predict, background)

# -----------------------
# STEP 8: COMPUTE SHAP VALUES
# -----------------------
# This may take several minutes depending on CPU
# shap_values[i][j] = contribution of feature j for row i
print("Computing SHAP values. This may take a while...")
shap_values = explainer.shap_values(scaled_sample)
print("SHAP computation completed!")

# -----------------------
# STEP 9: SAVE RESULTS
# -----------------------
# Save SHAP values, expected value, and the sampled rows
joblib.dump(shap_values, "shap_values.pkl")
joblib.dump(explainer.expected_value, "shap_expected.pkl")
joblib.dump(X_sample, "shap_rows.pkl")

print("SHAP values and data saved successfully!")
print("Files created: shap_values.pkl, shap_expected.pkl, shap_rows.pkl")
