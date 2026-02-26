# -----------------------
# IMPORTS
# -----------------------
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from lime.lime_tabular import LimeTabularExplainer
import shap

# -----------------------
# LOAD MODEL + SCALER
# -----------------------
model = tf.keras.models.load_model("diabetes_model.keras")
scaler = joblib.load("scaler.pkl")

# -----------------------
# LOAD DATASET (FOR LIME)
# -----------------------
df = pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2021.csv")
X = df.drop("Diabetes_binary", axis=1)
feature_names = X.columns.tolist()

# -----------------------
# UI TITLE
# -----------------------
st.title("Diabetes Prediction Dashboard")

# -----------------------
# LIME EXPLAINER (CREATE ONCE)
# -----------------------
explainer = LimeTabularExplainer(
    training_data=scaler.transform(X.values),
    feature_names=feature_names,
    class_names=["Not Diabetic", "Diabetic"],
    mode="classification"
)

def predict_fn(x):
    preds = model.predict(x)
    return np.hstack((1 - preds, preds))

# -----------------------
# ROW SELECTOR
# -----------------------
st.subheader("Select Data Row to Explain")
index = st.slider("Row Index", 0, len(X)-1, 0)

sample = X.iloc[index].values.reshape(1, -1)
scaled_sample = scaler.transform(sample)

# -----------------------
# BUTTON ACTION
# -----------------------
if st.button("Predict & Explain"):
    prob = model.predict(scaled_sample)[0][0]

    st.write(f"Probability of Diabetes: {prob:.3f}")
    st.write("Diabetic" if prob >= 0.5 else "Not Diabetic")

    exp = explainer.explain_instance(
        scaled_sample[0],
        predict_fn,
        num_features=8
    )

    st.write(f"Explanation Fidelity (R²): {exp.score:.3f}")


    st.pyplot(exp.as_pyplot_figure())

# -----------------------
# LOAD SHAP FILES
# -----------------------
shap_values = joblib.load("shap_values.pkl")
expected_value = joblib.load("shap_expected.pkl")
shap_rows = joblib.load("shap_rows.pkl")

# For binary classification, select class 1 (Diabetic)
if isinstance(shap_values, list):
    shap_values = shap_values[1]


import matplotlib.pyplot as plt

# -----------------------
# SHAP EXPLANATION SECTION
# -----------------------
st.subheader("SHAP Explanation (Precomputed Sample)")

shap_index = st.slider("Select SHAP Sample Index", 0, len(shap_rows)-1, 0)

if st.button("Show SHAP Explanation"):

    base_val = float(np.array(expected_value).flatten()[0])
    shap_val = shap_values[shap_index].flatten()

    explanation = shap.Explanation(
        values=shap_val,
        base_values=base_val,
        data=shap_rows.iloc[shap_index].values,
        feature_names=shap_rows.columns
    )

    fig = plt.figure()
    shap.plots.waterfall(explanation)
    st.pyplot(fig)
    plt.close(fig)

    # Global Summary
    fig2 = plt.figure()
    shap.summary_plot(
        shap_values.squeeze(),
        shap_rows,
        show=False
    )
    st.pyplot(fig2)
    plt.close(fig2)
