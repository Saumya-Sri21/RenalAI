import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model
@st.cache_resource
def load_model():
    with open("ckd_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

st.markdown(
    """
    <style>
    .centered-title {
        text-align: center;
        line-height: 1.5;
    }
    .main-title {
        font-size: 42px;
        font-weight: 700;
    }
    .subtitle {
        font-size: 36px;
        font-weight: 400;
        color: #cccccc; /* optional: softer color */
        margin-bottom: 35px;
    }
    </style>

    <div class="centered-title">
        <div class="main-title">🧬RenalAI</div>
        <div class="subtitle">  Your Personal Kidney Health Assistant</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("Enter your medical details to check CKD probability and stage estimation")

# Input fields based on dataset features
age = st.number_input("Age", min_value=1, max_value=100, value=40)
bp = st.number_input("Blood Pressure (mmHg)", value=80)
sg = st.number_input("Specific Gravity", value=1.02, step=0.001)
al = st.number_input("Albumin", value=1)
su = st.number_input("Sugar", value=0)
bgr = st.number_input("Blood Glucose (mg/dL)", value=100)
bu = st.number_input("Blood Urea (mg/dL)", value=30)
sc = st.number_input("Serum Creatinine (mg/dL)", value=1.2)
sod = st.number_input("Sodium (mEq/L)", value=135)
pot = st.number_input("Potassium (mEq/L)", value=4.5)
hemo = st.number_input("Hemoglobin (g/dL)", value=13.5)
pcv = st.number_input("Packed Cell Volume", value=40)
wbcc = st.number_input("White Blood Cell Count", value=8000)
rbcc = st.number_input("Red Blood Cell Count", value=4.5)
htn = st.selectbox("Hypertension", ["no", "yes"])
dm = st.selectbox("Diabetes Mellitus", ["no", "yes"])
cad = st.selectbox("Coronary Artery Disease", ["no", "yes"])
appet = st.selectbox("Appetite", ["good", "poor"])
pe = st.selectbox("Pedal Edema", ["no", "yes"])
ane = st.selectbox("Anemia", ["no", "yes"])

# Convert categorical to numeric (same encoding as training)
mapping = {"no": 0, "yes": 1, "good": 0, "poor": 1}
input_data = {
    'age': age, 'bp': bp, 'sg': sg, 'al': al, 'su': su, 'bgr': bgr, 'bu': bu,
    'sc': sc, 'sod': sod, 'pot': pot, 'hemo': hemo, 'pcv': pcv, 'wbcc': wbcc,
    'rbcc': rbcc, 'htn': mapping[htn], 'dm': mapping[dm], 'cad': mapping[cad],
    'appet': mapping[appet], 'pe': mapping[pe], 'ane': mapping[ane]
}

input_df = pd.DataFrame([input_data])

# Helper functions
def compute_egfr_mdrd(creatinine_mg_dl, age, female=True):
    try:
        val = 175 * (float(creatinine_mg_dl) ** -1.154) * (float(age) ** -0.203)
        if female: val *= 0.742
        return round(val, 1)
    except:
        return np.nan

def stage_from_egfr(egfr):
    if np.isnan(egfr): return "Unknown"
    if egfr >= 90: return "Stage 1 (Normal)"
    elif egfr >= 60: return "Stage 2 (Mild)"
    elif egfr >= 30: return "Stage 3 (Moderate)"
    elif egfr >= 15: return "Stage 4 (Severe)"
    else: return "Stage 5 (Failure)"

# Prediction
if st.button("Predict CKD Stage"):
    proba = model.predict_proba(input_df)[0, 1]
    egfr = compute_egfr_mdrd(sc, age)
    stage = stage_from_egfr(egfr)

    st.subheader("Results:")
    st.write(f"**CKD Probability:** {proba*100:.2f}%")
    st.write(f"**Estimated eGFR:** {egfr} ml/min/1.73m²")
    st.write(f"**CKD Stage:** {stage}")

    if proba < 0.3:
        st.success("Low risk — Kidney function likely normal.")
    elif proba < 0.6:
        st.warning("Moderate risk — Recommend regular monitoring.")
    else:
        st.error("High risk — Consult a nephrologist immediately.")
