import streamlit as st
import joblib
import pandas as pd

# Load the saved model and settings
model = joblib.load("sepsis_prediction_model.pkl")
feature_columns = joblib.load("sepsis_feature_columns.pkl")
threshold = joblib.load("sepsis_threshold.pkl")

st.title("Sepsis Prediction System")
st.write("Educational demonstration using a trained LightGBM model.")
st.warning(
    "This tool is for academic demonstration only, "
    "not for medical diagnosis or treatment."
)

st.subheader("Enter Patient Measurements")

with st.form("prediction_form"):
    values = {}

    for feature in feature_columns:
        values[feature] = st.number_input(
            feature,
            value=0.0,
            format="%.2f"
        )

    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = pd.DataFrame([values], columns=feature_columns)
    probability = model.predict_proba(input_data)[0][1]

    st.metric("Estimated model probability", f"{probability:.1%}")

    if probability >= threshold:
        st.error("Model prediction: Sepsis")
    else:
        st.success("Model prediction: No Sepsis")
