# ============================================
# STREAMLIT APP: Student Performance Predictor
# Author: Edwin Muoki
# ID: SCT213-C002-0062/2021
# JKUAT — BSc. Data Science and Analytics
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# LOAD MODELS AND ARTIFACTS
# ============================================
@st.cache_resource
def load_artifacts():
    """Load all saved model artifacts."""
    base_path = 'saved_models'

    model     = joblib.load(
        os.path.join(base_path, 'xgboost_model.pkl')
    )
    scaler    = joblib.load(
        os.path.join(base_path, 'scaler.pkl')
    )
    features  = pd.read_csv(
        os.path.join(base_path, 'selected_features.csv')
    )['selected_features'].tolist()

    with open(os.path.join(base_path, 'encoding_map.json')) as f:
        encoding_map = json.load(f)

    return model, scaler, features, encoding_map

model, scaler, selected_features, encoding_map = load_artifacts()

# ============================================
# HEADER
# ============================================
st.title("🎓 Student Performance Predictor")
st.markdown("""
**Predicting Student Performance Using Machine Learning**
*BSc. Data Science and Analytics — JKUAT*
""")

st.markdown("---")

# ============================================
# SIDEBAR — PROJECT INFO
# ============================================
with st.sidebar:
    st.header("📊 Model Information")
    st.success("🏆 Best Model: XGBoost")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", "93.67%")
        st.metric("Recall", "93.67%")
    with col2:
        st.metric("F1-Score", "93.69%")
        st.metric("ROC-AUC", "98.73%")

    st.markdown("---")
    st.header("📋 Performance Classes")
    st.error("🔴 Fail — G3 < 10")
    st.warning("🟡 Average — G3: 10–14")
    st.success("🟢 Pass — G3 > 14")

    st.markdown("---")
    st.header("ℹ️ About")
    st.info(
        "This model predicts student final "
        "performance based on academic history "
        "and background factors. It uses "
        "XGBoost trained on the UCI Student "
        "Performance dataset."
    )

# ============================================
# INPUT FORM
# ============================================
st.header("📝 Enter Student Information")
st.markdown(
    "Fill in the student details below and "
    "click **Predict** to get a performance prediction."
)

with st.form("prediction_form"):

    # ---- Academic History ----
    st.subheader("📚 Academic History")
    col1, col2, col3 = st.columns(3)

    with col1:
        G1 = st.slider(
            "First Term Grade (G1)",
            min_value=0, max_value=20, value=10,
            help="Student's first term grade (0-20)"
        )
    with col2:
        G2 = st.slider(
            "Second Term Grade (G2)",
            min_value=0, max_value=20, value=10,
            help="Student's second term grade (0-20)"
        )
    with col3:
        failures = st.selectbox(
            "Past Class Failures",
            options=[0, 1, 2, 3],
            help="Number of past class failures"
        )

    # ---- School Information ----
    st.subheader("🏫 School Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        school = st.selectbox(
            "School",
            options=["GP", "MS"],
            help="GP = Gabriel Pereira, MS = Mousinho"
        )
    with col2:
        schoolsup = st.selectbox(
            "School Educational Support",
            options=["yes", "no"],
            help="Extra educational support from school"
        )
    with col3:
        paid = st.selectbox(
            "Extra Paid Classes",
            options=["yes", "no"],
            help="Extra paid classes within subject"
        )

    # ---- Family Background ----
    st.subheader("👨‍👩‍👧 Family Background")
    col1, col2, col3 = st.columns(3)

    with col1:
        Medu = st.selectbox(
            "Mother's Education",
            options=[0, 1, 2, 3, 4],
            index=2,
            help="0=none, 1=primary, 2=middle, "
                 "3=secondary, 4=higher"
        )
    with col2:
        Fedu = st.selectbox(
            "Father's Education",
            options=[0, 1, 2, 3, 4],
            index=2,
            help="0=none, 1=primary, 2=middle, "
                 "3=secondary, 4=higher"
        )
    with col3:
        famrel = st.slider(
            "Family Relationship Quality",
            min_value=1, max_value=5, value=3,
            help="1=very bad, 5=excellent"
        )

    col1, col2 = st.columns(2)
    with col1:
        Mjob = st.selectbox(
            "Mother's Job",
            options=[
                "at_home", "health",
                "other", "services", "teacher"
            ],
            help="Mother's occupation"
        )
    with col2:
        Fjob = st.selectbox(
            "Father's Job",
            options=[
                "at_home", "health",
                "other", "services", "teacher"
            ],
            help="Father's occupation"
        )

    col1, col2 = st.columns(2)
    with col1:
        famsup = st.selectbox(
            "Family Educational Support",
            options=["yes", "no"],
            help="Family provides educational support"
        )
    with col2:
        reason = st.selectbox(
            "Reason for Choosing School",
            options=[
                "course", "home",
                "other", "reputation"
            ],
            help="Why student chose this school"
        )

    # ---- Lifestyle & Social ----
    st.subheader("🎯 Lifestyle & Social Factors")
    col1, col2, col3 = st.columns(3)

    with col1:
        goout = st.slider(
            "Going Out With Friends",
            min_value=1, max_value=5, value=3,
            help="1=very low, 5=very high"
        )
    with col2:
        Dalc = st.slider(
            "Weekday Alcohol Consumption",
            min_value=1, max_value=5, value=1,
            help="1=very low, 5=very high"
        )
    with col3:
        health = st.slider(
            "Current Health Status",
            min_value=1, max_value=5, value=3,
            help="1=very bad, 5=very good"
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        activities = st.selectbox(
            "Extra-Curricular Activities",
            options=["yes", "no"],
            help="Participates in activities"
        )
    with col2:
        romantic = st.selectbox(
            "In Romantic Relationship",
            options=["yes", "no"],
            help="Currently in a relationship"
        )
    with col3:
        absences = st.slider(
            "Number of Absences",
            min_value=0, max_value=93, value=0,
            help="Number of school absences"
        )

    # ---- Submit Button ----
    st.markdown("---")
    submitted = st.form_submit_button(
        "🔮 Predict Performance",
        use_container_width=True
    )

# ============================================
# PREDICTION LOGIC
# ============================================
if submitted:

    # ---- Encode binary inputs ----
    school_enc     = 0 if school == "GP" else 1
    schoolsup_enc  = 1 if schoolsup == "yes" else 0
    paid_enc       = 1 if paid == "yes" else 0
    famsup_enc     = 1 if famsup == "yes" else 0
    activities_enc = 1 if activities == "yes" else 0
    romantic_enc   = 1 if romantic == "yes" else 0

    # ---- Calculate engineered features ----
    family_edu_score    = (Medu + Fedu) / 2
    academic_risk_score = (failures * 3) + (absences * 0.5)
    social_engagement   = goout + activities_enc + romantic_enc
    parental_support_idx = famrel + famsup_enc

    # ---- One-hot encode Mjob ----
    Mjob_health    = 1 if Mjob == "health" else 0
    Mjob_other     = 1 if Mjob == "other" else 0
    Mjob_services  = 1 if Mjob == "services" else 0
    Mjob_teacher   = 1 if Mjob == "teacher" else 0
    # at_home is the dropped reference category

    # ---- One-hot encode Fjob ----
    Fjob_health    = 1 if Fjob == "health" else 0
    Fjob_other     = 1 if Fjob == "other" else 0
    Fjob_services  = 1 if Fjob == "services" else 0
    Fjob_teacher   = 1 if Fjob == "teacher" else 0

    # ---- One-hot encode reason ----
    reason_home       = 1 if reason == "home" else 0
    reason_other      = 1 if reason == "other" else 0
    reason_reputation = 1 if reason == "reputation" else 0

    # ---- Build input dictionary ----
    # Must match selected_features order exactly
    input_dict = {
        'G2'                 : G2,
        'G1'                 : G1,
        'failures'           : failures,
        'activities'         : activities_enc,
        'Medu'               : Medu,
        'family_edu_score'   : family_edu_score,
        'reason_reputation'  : reason_reputation,
        'Mjob_services'      : Mjob_services,
        'academic_risk_score': academic_risk_score,
        'social_engagement'  : social_engagement,
        'school'             : school_enc,
        'Dalc'               : Dalc,
        'health'             : health,
        'paid'               : paid_enc,
        'reason_home'        : reason_home,
        'Fjob_other'         : Fjob_other,
        'famrel'             : famrel,
        'parental_support_idx': parental_support_idx,
        'Mjob_health'        : Mjob_health,
        'schoolsup'          : schoolsup_enc,
        'Mjob_other'         : Mjob_other
    }

    # ---- Create DataFrame in correct feature order ----
    input_df = pd.DataFrame([input_dict])[selected_features]

    # ---- Scale input ----
    input_scaled = scaler.transform(input_df)

    # ---- Predict ----
    prediction      = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]

    # ---- Display Results ----
    st.markdown("---")
    st.header("🎯 Prediction Results")

    # Map prediction to label
    label_map = {0: 'Fail', 1: 'Average', 2: 'Pass'}
    predicted_label = label_map[prediction]

    # Display main prediction
    col1, col2 = st.columns([1, 2])

    with col1:
        if prediction == 0:
            st.error(f"## 🔴 {predicted_label}")
            st.error(
                "This student is **at risk of failing.** "
                "Immediate intervention recommended."
            )
        elif prediction == 1:
            st.warning(f"## 🟡 {predicted_label}")
            st.warning(
                "This student is performing **adequately.** "
                "Monitor and provide support where needed."
            )
        else:
            st.success(f"## 🟢 {predicted_label}")
            st.success(
                "This student is **on track to pass.** "
                "Continue current approach."
            )

    with col2:
        st.subheader("Prediction Probabilities")
        prob_df = pd.DataFrame({
            'Performance Class': ['Fail', 'Average', 'Pass'],
            'Probability'      : prediction_proba,
            'Percentage'       : [
                f"{p*100:.1f}%" for p in prediction_proba
            ]
        })

        # Display probability bars
        for i, row in prob_df.iterrows():
            label = row['Performance Class']
            prob  = row['Probability']
            pct   = row['Percentage']

            if label == 'Fail':
                color = 'red'
            elif label == 'Average':
                color = 'orange'
            else:
                color = 'green'

            st.markdown(f"**{label}:** {pct}")
            st.progress(float(prob))

    # ---- Input Summary ----
    st.markdown("---")
    st.subheader("📋 Input Summary")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Academic:**")
        st.write(f"G1: {G1} | G2: {G2}")
        st.write(f"Past failures: {failures}")
        st.write(f"Absences: {absences}")
        st.write(f"Academic risk score: "
                 f"{academic_risk_score:.1f}")

    with col2:
        st.markdown("**Family:**")
        st.write(f"Mother education: {Medu}")
        st.write(f"Father education: {Fedu}")
        st.write(f"Family edu score: "
                 f"{family_edu_score:.1f}")
        st.write(f"Family support: {famsup}")

    with col3:
        st.markdown("**Lifestyle:**")
        st.write(f"Going out: {goout}")
        st.write(f"Alcohol (weekday): {Dalc}")
        st.write(f"Health: {health}")
        st.write(f"Social engagement: "
                 f"{social_engagement}")

    # ---- Recommendation ----
    st.markdown("---")
    st.subheader("💡 Recommendation")

    if prediction == 0:
        st.error("""
        **Immediate Actions Recommended:**
        - Schedule urgent academic counseling session
        - Review attendance and address absence patterns
        - Connect student with tutoring resources
        - Notify parent or guardian
        - Create a structured study plan
        """)
    elif prediction == 1:
        st.warning("""
        **Monitoring Recommended:**
        - Schedule check-in with academic advisor
        - Encourage consistent study habits
        - Monitor attendance going forward
        - Consider peer study groups
        """)
    else:
        st.success("""
        **Student On Track:**
        - Maintain current study habits
        - Continue extra-curricular engagement
        - Consider mentoring struggling peers
        """)

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
Student Performance Prediction System |
Edwin Muoki | SCT213-C002-0062/2021 |
JKUAT BSc. Data Science and Analytics |
UCI Student Performance Dataset — Cortez & Silva (2008)
</div>
""", unsafe_allow_html=True)