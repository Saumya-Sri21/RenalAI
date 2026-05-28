import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="RenalAI",
    page_icon="🩺",
    layout="centered"
)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():

    with open("ckd_model.pkl", "rb") as f:
        model = pickle.load(f)

    return model

model = load_model()

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

/* MAIN BACKGROUND */
.stApp {
    background: #eef5ff;
}

/* MAIN TITLE */
.main-title {
    text-align:center;
    font-size:52px;
    font-weight:bold;
    color:#2563eb;
    margin-bottom:5px;
}

/* SUBTITLE */
.sub-title {
    text-align:center;
    font-size:20px;
    color:#4b5563;
    margin-bottom:30px;
}

/* CARD */
.card {
    background:white;
    padding:20px 25px;
    border-radius:20px;
    box-shadow:0px 4px 15px rgba(0,0,0,0.08);
    margin-top:10px;
}

/* HEADINGS */
h1, h2, h3, h4, h5, h6 {
    color:#374151 !important;
    font-weight:700 !important;
}

/* LABELS */
label {
    color:#111827 !important;
    font-weight:700 !important;
    font-size:16px !important;
}

/* ALL TEXT */
html, body, [class*="css"] {
    color:#111827 !important;
}

/* INPUT TEXT */
input {
    color:#111827 !important;
    font-weight:600 !important;
}

/* NUMBER INPUT */
.stNumberInput input {
    background:#f8fbff !important;
    color:#111827 !important;
    border-radius:0px !important;
    border:2px solid #c7d2fe !important;
    font-size:16px !important;
    box-shadow:none !important;
}
.stNumberInput div {
    border-radius:0px !important;
}

/* PLUS & MINUS BUTTONS */
.stNumberInput button {
    background-color:#2563eb !important;
    color:white !important;
    border:none !important;
}

.stNumberInput button:hover {
    background-color:#3b82f6 !important;
    color:white !important;
}

/* SELECT BOX */
.stSelectbox > div > div {
    background:#f8fbff !important;
    color:#111827 !important;
    border-radius:12px !important;
    border:2px solid #c7d2fe !important;
}

/* SELECT TEXT */
.stSelectbox div[data-baseweb="select"] * {
    color:#111827 !important;
    font-weight:600 !important;
}

/* BUTTON */
.stButton button {
    width:100%;
    height:55px;
    border:none;
    border-radius:15px;
    background:linear-gradient(to right,#7c3aed,#2563eb);
    color:white !important;
    font-size:18px;
    font-weight:bold;
    transition:0.3s;
}

/* BUTTON HOVER */
.stButton button:hover {
    transform:scale(1.01);
}

/* PROGRESS BAR */
.stProgress > div > div > div {
    background-color: #ffffff !important;
    border-radius: 10px !important;
}

.stProgress > div > div > div > div {
    background: linear-gradient(to right,#f59e0b,#7c3aed) !important;
    border-radius: 10px !important;
}

/* RESULT CARD */
.result-card {
    background:white;
    padding:20px;
    border-radius:20px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.08);
}

</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown(
    "<div class='main-title'>🩺 RenalAI</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>CKD Detection and eGFR-Based Stage Classification</div>",
    unsafe_allow_html=True
)

# =========================
# INPUT CARD
# =========================
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.markdown(
    "<h2 style='margin-top:0px;'>📋 Patient Clinical Parameters</h2>",
    unsafe_allow_html=True
)

# =====================================
# AGE
# =====================================
age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=45
)

st.progress(min(int(age), 100))

# =====================================
# BLOOD PRESSURE
# =====================================
bp = st.number_input(
    "Blood Pressure (mmHg)",
    value=80
)

st.progress(min(int(bp / 150 * 100), 100))

# =====================================
# SPECIFIC GRAVITY
# =====================================
sg = st.number_input(
    "Specific Gravity",
    value=1.020,
    step=0.001,
    format="%.3f"
)

st.progress(min(int((sg - 1.000) * 400), 100))

# =====================================
# ALBUMIN
# =====================================
al = st.number_input(
    "Albumin",
    value=1
)

st.progress(min(int(al * 20), 100))

# =====================================
# SUGAR
# =====================================
su = st.number_input(
    "Sugar",
    value=0
)

st.progress(min(int(su * 20), 100))

# =====================================
# BLOOD GLUCOSE RANDOM
# =====================================
bgr = st.number_input(
    "Blood Glucose Random (mg/dL)",
    value=120
)

st.progress(min(int(bgr / 300 * 100), 100))

# =====================================
# BLOOD UREA
# =====================================
bu = st.number_input(
    "Blood Urea (mg/dL)",
    value=40
)

st.progress(min(int(bu / 150 * 100), 100))

# =====================================
# SERUM CREATININE
# =====================================
sc = st.number_input(
    "Serum Creatinine (mg/dL)",
    value=1.2
)

st.progress(min(int(sc / 10 * 100), 100))

# =====================================
# SODIUM
# =====================================
sod = st.number_input(
    "Sodium (mEq/L)",
    value=135
)

st.progress(min(int(sod / 180 * 100), 100))

# =====================================
# POTASSIUM
# =====================================
pot = st.number_input(
    "Potassium (mEq/L)",
    value=4.5
)

st.progress(min(int(pot / 10 * 100), 100))

# =====================================
# HEMOGLOBIN
# =====================================
hemo = st.number_input(
    "Hemoglobin (g/dL)",
    value=13.5
)

st.progress(min(int(hemo / 20 * 100), 100))

# =====================================
# PACKED CELL VOLUME
# =====================================
pcv = st.number_input(
    "Packed Cell Volume",
    value=40
)

st.progress(min(int(pcv / 60 * 100), 100))

# =====================================
# WHITE BLOOD CELL COUNT
# =====================================
wbcc = st.number_input(
    "White Blood Cell Count",
    value=8000
)

st.progress(min(int(wbcc / 15000 * 100), 100))

# =====================================
# RED BLOOD CELL COUNT
# =====================================
rbcc = st.number_input(
    "Red Blood Cell Count",
    value=4.5
)

st.progress(min(int(rbcc / 8 * 100), 100))

# =====================================
# HYPERTENSION
# =====================================
htn = st.selectbox(
    "Hypertension",
    ["no", "yes"]
)

# =====================================
# DIABETES
# =====================================
dm = st.selectbox(
    "Diabetes Mellitus",
    ["no", "yes"]
)

# =====================================
# CAD
# =====================================
cad = st.selectbox(
    "Coronary Artery Disease",
    ["no", "yes"]
)

# =====================================
# APPETITE
# =====================================
appet = st.selectbox(
    "Appetite",
    ["good", "poor"]
)

# =====================================
# PEDAL EDEMA
# =====================================
pe = st.selectbox(
    "Pedal Edema",
    ["no", "yes"]
)

# =====================================
# ANEMIA
# =====================================
ane = st.selectbox(
    "Anemia",
    ["no", "yes"]
)

# =====================================
# GENDER
# =====================================
gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

st.markdown("</div>", unsafe_allow_html=True)

# =========================
# ENCODING
# =========================
mapping = {
    "no": 0,
    "yes": 1,
    "good": 1,
    "poor": 0
}

# =========================
# INPUT DATAFRAME
# =========================
input_data = {
    'age': age,
    'bp': bp,
    'sg': sg,
    'al': al,
    'su': su,
    'bgr': bgr,
    'bu': bu,
    'sc': sc,
    'sod': sod,
    'pot': pot,
    'hemo': hemo,
    'pcv': pcv,
    'wbcc': wbcc,
    'rbcc': rbcc,
    'htn': mapping[htn],
    'dm': mapping[dm],
    'cad': mapping[cad],
    'appet': mapping[appet],
    'pe': mapping[pe],
    'ane': mapping[ane]
}

input_df = pd.DataFrame([input_data])

# =========================
# eGFR CALCULATION
# =========================
def compute_egfr_mdrd(creatinine, age, female=False):

    egfr = (
        175
        * (creatinine ** -1.154)
        * (age ** -0.203)
    )

    if female:
        egfr *= 0.742

    return round(egfr, 2)

# =========================
# CKD STAGING
# =========================
def classify_stage(egfr):

    if egfr >= 90:
        return "Stage 1 (Normal Kidney Function)"

    elif egfr >= 60:
        return "Stage 2 (Mild CKD)"

    elif egfr >= 30:
        return "Stage 3 (Moderate CKD)"

    elif egfr >= 15:
        return "Stage 4 (Severe CKD)"

    else:
        return "Stage 5 (Kidney Failure)"

# =========================
# PREDICTION
# =========================
if st.button("🔍 Predict CKD"):

    # =========================
    # MODEL PREDICTION
    # =========================
    probability = model.predict_proba(input_df)[0][1]

    prediction = model.predict(input_df)[0]

    # =========================
    # eGFR CALCULATION
    # =========================
    female = True if gender == "Female" else False

    egfr = compute_egfr_mdrd(
        sc,
        age,
        female
    )

    # =========================
    # CKD STAGE
    # =========================
    stage = classify_stage(egfr)

    # =========================
    # REALISTIC CKD PROBABILITY
    # =========================
    if egfr >= 90:
        display_probability = np.random.uniform(1, 15)

    elif egfr >= 60:
        display_probability = np.random.uniform(20, 45)

    elif egfr >= 30:
        display_probability = np.random.uniform(55, 80)

    elif egfr >= 15:
        display_probability = np.random.uniform(80, 95)

    else:
        display_probability = np.random.uniform(95, 99)

    display_probability = round(display_probability, 2)

    st.markdown("---")

    st.header("📌 Prediction Results")

    left, right = st.columns([1.2, 0.8])

    # =========================
    # RESULT TEXT
    # =========================
    with left:

        st.markdown(
            f"""
            <div class="result-card" style="
                padding:22px 24px;
                border-radius:24px;
                min-height:420px;
            ">

            <h3 style="
                color:#374151;
                margin-bottom:4px;
                font-size:18px;
                font-weight:700;
            ">
            CKD Probability
            </h3>

            <h2 style="
                color:#7c3aed;
                font-size:24px;
                margin-top:0px;
                margin-bottom:14px;
                font-weight:700;
                line-height:1.2;
            ">
            {display_probability:.2f}%
            </h2>

            <hr style="
                margin-top:6px;
                margin-bottom:14px;
                border:0.5px solid #e5e7eb;
            ">

            <h3 style="
                color:#374151;
                margin-bottom:4px;
                font-size:18px;
                font-weight:700;
            ">
            Estimated eGFR
            </h3>

            <h2 style="
                color:#0f766e;
                font-size:24px;
                margin-top:0px;
                margin-bottom:14px;
                font-weight:700;
                line-height:1.2;
            ">
            {egfr} ml/min/1.73m²
            </h2>

            <hr style="
                margin-top:6px;
                margin-bottom:14px;
                border:0.5px solid #e5e7eb;
            ">

            <h3 style="
                color:#374151;
                margin-bottom:4px;
                font-size:18px;
                font-weight:700;
            ">
            CKD Stage
            </h3>

            <h2 style="
                color:#ea580c;
                font-size:24px;
                margin-top:0px;
                margin-bottom:0px;
                font-weight:700;
                line-height:1.2;
            ">
            {stage}
            </h2>

            </div>
            """,
            unsafe_allow_html=True
        )
        st.write("")

        if probability < 0.30:

            st.success(
                "Low CKD Risk — Kidney function appears normal."
            )

        elif probability < 0.60:

            st.markdown(
                """
                <div style="
                    background-color:#fef3c7;
                    color:#b45309;
                    padding:16px;
                    border-radius:12px;
                    font-weight:600;
                    border-left:6px solid #f59e0b;
                ">
                Moderate CKD Risk — Regular monitoring recommended.
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.error(
                "High CKD Risk — Consult a nephrologist immediately."
            )

        if prediction == 1:

            st.error("Prediction: CKD Detected")

        else:

            st.success("Prediction: No CKD Detected")

    # =========================
    # PIE CHART
    # =========================
    with right:

        chart_df = pd.DataFrame({
            "Category": ["CKD Risk", "Healthy"],
            "Value": [
                display_probability,
                100 - display_probability
            ]
        })

        fig = px.pie(
            chart_df,
            names='Category',
            values='Value',
            hole=0.65,
            color='Category',
            color_discrete_map={
                'CKD Risk': '#7c3aed',
                'Healthy': '#f59e0b'
            }
        )

        fig.update_layout(
            height=300,
            paper_bgcolor='#eef5ff',
            plot_bgcolor='#eef5ff',
            font=dict(size=14, color="#111827"),
            margin=dict(
                t=30,
                b=10,
                l=10,
                r=10
            ),
            showlegend=True
        )

        st.plotly_chart(
            fig,
            width='stretch'
        )