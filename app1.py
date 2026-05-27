# -*- coding: utf-8 -*-
"""
Created on Wed May 27 00:53:02 2026

@author: nk336
"""

import streamlit as st
import pandas as pd
import joblib
import lightgbm as lgb

# ======================
# LOAD MODEL
# ======================

model = joblib.load("clean_model2.pkl")
encoders = joblib.load("encoders2.pkl")
features = joblib.load("features2.pkl")

st.title("5-Year Cardiovascular Risk Predictor")

# ======================
# USER INPUTS
# ======================

age = st.number_input("Age", 18, 100, 50)
gender = st.selectbox("Gender", ["Male", "Female"])
imd = st.slider("IMD", 1, 10, 5)

bmi = st.number_input("BMI", 10.0, 60.0, 28.0)
sbp = st.number_input("Systolic BP", 80, 250, 130)
dbp = st.number_input("Diastolic BP", 40, 150, 80)

hba1c = st.number_input("HbA1c", 20.0, 150.0, 50.0)
hdl = st.number_input("HDL", 0.1, 5.0, 1.2)
ldl = st.number_input("LDL", 0.1, 10.0, 3.0)

triglyceride = st.number_input("Triglycerides", 0.1, 20.0, 2.0)
egfr = st.number_input("eGFR", 1.0, 150.0, 90.0)

ckd = st.checkbox("CKD")
htn = st.checkbox("Hypertension")
af = st.checkbox("Atrial Fibrillation")
copd = st.checkbox("COPD")
asthma = st.checkbox("Asthma")
hld = st.checkbox("Hyperlipidemia")

# ======================
# PREDICT
# ======================

if st.button("Calculate Risk"):

    patient = {
        "age": age,
        "gender": gender,
        "IMD": imd,
        "bmi": bmi,
        "sbp": sbp,
        "dbp": dbp,
        "hba1c": hba1c,
        "hdl": hdl,
        "ldl": ldl,
        "triglyceride": triglyceride,
        "egfr": egfr,
        "CKD": int(ckd),
        "Hypertension": int(htn),
        "Atrial_fibrillation": int(af),
        "COPD": int(copd),
        "Asthma": int(asthma),
        "Hyperlipidemia": int(hld)
    }

    df = pd.DataFrame([patient])

    df = df.reindex(columns=features, fill_value=0)

    for col, le in encoders.items():

        if col in df.columns:

            df[col] = [
                le.transform([str(v)])[0]
                if str(v) in le.classes_
                else -1
                for v in df[col]
            ]

    risk = model.predict_proba(df)[0][1]

    if risk < 0.1:
        band = "Low"
    elif risk < 0.2:
        band = "Moderate"
    else:
        band = "High"

    st.metric("5-Year CV Risk", f"{risk:.1%}")
    st.success(f"Risk Category: {band}")