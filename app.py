import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Cervical Cancer Early-Risk Prediction",
    page_icon="🩺",
    layout="wide"
)

# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model_and_scaler():

    model = None
    scaler = None
    model_error = None
    scaler_error = None

    # Load model
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
        else:
            model_error = "model.pkl not found"
    except Exception as e:
        model_error = str(e)

    # Load scaler
    try:
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
        else:
            scaler_error = "scaler.pkl not found"
    except Exception as e:
        scaler_error = str(e)

    return model, scaler, model_error, scaler_error


model, scaler, model_error, scaler_error = load_model_and_scaler()


# ============================================================
# TITLE
# ============================================================

st.title("🩺 Cervical Cancer Early-Risk Prediction System")

st.markdown(
    """
    This application uses a Machine Learning model to generate a
    **model-estimated cervical cancer risk score** from the information
    entered by the user.
    """
)

st.warning(
    "⚠️ Educational/project tool only. The percentage shown here is "
    "a machine-learning model score and is NOT a medical diagnosis."
)


# ============================================================
# FILE STATUS
# ============================================================

if model is None:

    st.error(
        "❌ model.pkl could not be loaded."
    )

    if model_error:
        st.code(model_error)

    st.info(
        "Make sure model.pkl is inside the same folder as app.py."
    )

    st.stop()


if scaler is None:

    st.warning(
        "⚠️ scaler.pkl could not be loaded. "
        "The application will try to use the model directly."
    )

    if scaler_error:
        st.caption(scaler_error)


# ============================================================
# PATIENT INFORMATION
# ============================================================

st.header("👩 Patient Information")

col1, col2, col3 = st.columns(3)


# ---------------- COLUMN 1 ----------------

with col1:

    age = st.number_input(
        "Age",
        min_value=10.0,
        max_value=100.0,
        value=25.0,
        step=1.0
    )

    sexual_partners = st.number_input(
        "Number of sexual partners",
        min_value=0.0,
        max_value=50.0,
        value=1.0,
        step=1.0
    )

    first_sexual_intercourse = st.number_input(
        "Age at first sexual intercourse",
        min_value=5.0,
        max_value=60.0,
        value=18.0,
        step=1.0
    )

    pregnancies = st.number_input(
        "Number of pregnancies",
        min_value=0.0,
        max_value=20.0,
        value=0.0,
        step=1.0
    )


# ---------------- COLUMN 2 ----------------

with col2:

    smokes = st.selectbox(
        "Smoking history?",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    smokes_years = st.number_input(
        "Smoking years",
        min_value=0.0,
        max_value=80.0,
        value=0.0,
        step=1.0
    )

    smokes_packs = st.number_input(
        "Smoking packs/year",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=1.0
    )

    hormonal = st.selectbox(
        "Hormonal contraceptive history?",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    hormonal_years = st.number_input(
        "Hormonal contraceptive years",
        min_value=0.0,
        max_value=80.0,
        value=0.0,
        step=1.0
    )


# ---------------- COLUMN 3 ----------------

with col3:

    iud = st.selectbox(
        "IUD history?",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    iud_years = st.number_input(
        "IUD years",
        min_value=0.0,
        max_value=80.0,
        value=0.0,
        step=1.0
    )

    stds = st.selectbox(
        "History of STDs?",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    std_number = st.number_input(
        "Number of STDs",
        min_value=0.0,
        max_value=20.0,
        value=0.0,
        step=1.0
    )


# ============================================================
# STD INFORMATION
# ============================================================

st.header("🦠 STD Information")

std_features = [
    "STDs:condylomatosis",
    "STDs:cervical condylomatosis",
    "STDs:vaginal condylomatosis",
    "STDs:vulvo-perineal condylomatosis",
    "STDs:syphilis",
    "STDs:pelvic inflammatory disease",
    "STDs:genital herpes",
    "STDs:molluscum contagiosum",
    "STDs:AIDS",
    "STDs:HIV",
    "STDs:Hepatitis B",
    "STDs:HPV"
]

std_values = {}

std_cols = st.columns(3)

for i, feature in enumerate(std_features):

    with std_cols[i % 3]:

        std_values[feature] = st.selectbox(
            feature,
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            key="std_" + feature
        )


# ============================================================
# STD DIAGNOSIS
# ============================================================

st.subheader("STD Diagnosis Information")

col4, col5, col6 = st.columns(3)

with col4:

    std_diagnosis = st.number_input(
        "STDs: Number of diagnosis",
        min_value=0.0,
        max_value=20.0,
        value=0.0,
        step=1.0
    )

with col5:

    std_first = st.number_input(
        "Time since first STD diagnosis",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=1.0
    )

with col6:

    std_last = st.number_input(
        "Time since last STD diagnosis",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=1.0
    )


# ============================================================
# CREATE PATIENT DATA
# ============================================================

input_data = {

    "Age": age,

    "Number of sexual partners":
        sexual_partners,

    "First sexual intercourse":
        first_sexual_intercourse,

    "Num of pregnancies":
        pregnancies,

    "Smokes":
        smokes,

    "Smokes (years)":
        smokes_years,

    "Smokes (packs/year)":
        smokes_packs,

    "Hormonal Contraceptives":
        hormonal,

    "Hormonal Contraceptives (years)":
        hormonal_years,

    "IUD":
        iud,

    "IUD (years)":
        iud_years,

    "STDs":
        stds,

    "STDs (number)":
        std_number,

    "STDs: Number of diagnosis":
        std_diagnosis,

    "STDs: Time since first diagnosis":
        std_first,

    "STDs: Time since last diagnosis":
        std_last
}


# Add STD features

input_data.update(std_values)


# ============================================================
# DIAGNOSIS FEATURES
# ============================================================

# These are added because some versions of the Kaggle dataset
# contain diagnosis columns.

diagnosis_features = {

    "Dx": 0,
    "Dx:CIN": 0,
    "Dx:Cancer": 0,
    "Dx:HPV": 0
}

input_data.update(diagnosis_features)


# ============================================================
# CREATE DATAFRAME
# ============================================================

input_df = pd.DataFrame([input_data])


# ============================================================
# MODEL FEATURE ALIGNMENT
# ============================================================

def prepare_model_input(model, scaler, dataframe):

    df = dataframe.copy()

    # --------------------------------------------------------
    # Find features used during model training
    # --------------------------------------------------------

    model_features = None

    if hasattr(model, "feature_names_in_"):

        model_features = list(model.feature_names_in_)

    # --------------------------------------------------------
    # If model does not have feature names
    # --------------------------------------------------------

    if model_features is None:

        if hasattr(model, "n_features_in_"):

            expected = model.n_features_in_

            if df.shape[1] != expected:

                raise ValueError(
                    f"Model expects {expected} features, "
                    f"but application generated {df.shape[1]} features."
                )

        return df, None


    # --------------------------------------------------------
    # Add missing features automatically
    # --------------------------------------------------------

    for feature in model_features:

        if feature not in df.columns:

            df[feature] = 0


    # --------------------------------------------------------
    # Remove extra columns
    # --------------------------------------------------------

    df = df[model_features]


    # --------------------------------------------------------
    # First try raw model input
    # --------------------------------------------------------

    raw_input = df.copy()


    # --------------------------------------------------------
    # Scaled input
    # --------------------------------------------------------

    scaled_input = None

    if scaler is not None:

        try:

            if hasattr(scaler, "feature_names_in_"):

                scaler_features = list(
                    scaler.feature_names_in_
                )

                scaler_df = df.copy()

                for feature in scaler_features:

                    if feature not in scaler_df.columns:

                        scaler_df[feature] = 0

                scaler_df = scaler_df[scaler_features]

                scaled_values = scaler.transform(
                    scaler_df
                )

                scaled_input = scaled_values

            else:

                if hasattr(scaler, "n_features_in_"):

                    if scaler.n_features_in_ == df.shape[1]:

                        scaled_input = scaler.transform(
                            df
                        )

        except Exception:

            scaled_input = None


    return raw_input, scaled_input


# ============================================================
# PERSONALIZED GUIDANCE
# ============================================================

def generate_patient_guidance(patient_data, risk_percentage):

    guidance = []


    # --------------------------------------------------------
    # RISK CATEGORY
    # --------------------------------------------------------

    if risk_percentage < 20:

        guidance.append(
            "The model-estimated score is in the lower range. "
            "Continue recommended preventive healthcare and "
            "routine cervical-health screening."
        )

    elif risk_percentage < 50:

        guidance.append(
            "The model-estimated score is in the intermediate range. "
            "Discuss your individual risk factors and appropriate "
            "screening with a qualified healthcare professional."
        )

    else:

        guidance.append(
            "The model-estimated score is in the higher range. "
            "Please discuss the result and appropriate follow-up "
            "with a qualified healthcare professional."
        )


    # --------------------------------------------------------
    # SMOKING
    # --------------------------------------------------------

    if patient_data["Smokes"] == 1:

        guidance.append(
            "Smoking history was reported. Avoiding tobacco and "
            "discussing smoking-cessation support with a healthcare "
            "professional may be beneficial."
        )


    # --------------------------------------------------------
    # HPV
    # --------------------------------------------------------

    if patient_data["STDs:HPV"] == 1:

        guidance.append(
            "HPV history was reported. Discuss appropriate HPV-related "
            "screening and follow-up with a healthcare professional."
        )


    # --------------------------------------------------------
    # STD
    # --------------------------------------------------------

    if patient_data["STDs"] == 1:

        guidance.append(
            "An STD history was reported. Discuss appropriate "
            "sexual-health screening and follow-up."
        )


    # --------------------------------------------------------
    # MULTIPLE PREGNANCIES
    # --------------------------------------------------------

    if patient_data["Num of pregnancies"] >= 3:

        guidance.append(
            "Multiple pregnancies were reported. Include this history "
            "when discussing your overall cervical-health care."
        )


    # --------------------------------------------------------
    # HORMONAL CONTRACEPTIVE
    # --------------------------------------------------------

    if patient_data["Hormonal Contraceptives"] == 1:

        guidance.append(
            "Hormonal contraceptive use was reported. Share this "
            "history with your healthcare professional when discussing "
            "your individual screening needs."
        )


    # --------------------------------------------------------
    # IUD
    # --------------------------------------------------------

    if patient_data["IUD"] == 1:

        guidance.append(
            "IUD history was reported. Include this information in "
            "your reproductive-health history."
        )


    # --------------------------------------------------------
    # STD DIAGNOSIS
    # --------------------------------------------------------

    if patient_data["STDs: Number of diagnosis"] > 0:

        guidance.append(
            "Previous STD diagnoses were reported. Appropriate "
            "follow-up can be discussed with a healthcare professional."
        )


    # --------------------------------------------------------
    # SMOKING YEARS
    # --------------------------------------------------------

    if patient_data["Smokes (years)"] > 0:

        guidance.append(
            "A smoking duration was entered. Smoking cessation support "
            "may be discussed with a healthcare professional."
        )


    # --------------------------------------------------------
    # NO EXTRA FACTORS
    # --------------------------------------------------------

    if len(guidance) == 1:

        guidance.append(
            "No additional factor-specific guidance was triggered "
            "by the information entered."
        )


    return guidance


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.header("🔍 Prediction")

predict_button = st.button(
    "🔎 Predict Cervical Cancer Risk",
    type="primary",
    use_container_width=True
)


if predict_button:

    try:

        # ====================================================
        # PREPARE INPUT
        # ====================================================

        raw_input, scaled_input = prepare_model_input(
            model,
            scaler,
            input_df
        )


        prediction = None
        probability = None
        used_input = None


        # ====================================================
        # TRY RAW INPUT
        # ====================================================

        try:

            if hasattr(model, "predict_proba"):

                probability_array = model.predict_proba(
                    raw_input
                )

                if probability_array.shape[1] >= 2:

                    probability = float(
                        probability_array[0][1]
                    )

            prediction = model.predict(
                raw_input
            )[0]

            used_input = raw_input


        except Exception as raw_error:

            # =================================================
            # TRY SCALED INPUT
            # =================================================

            if scaled_input is None:

                raise raw_error

            if hasattr(model, "predict_proba"):

                probability_array = model.predict_proba(
                    scaled_input
                )

                if probability_array.shape[1] >= 2:

                    probability = float(
                        probability_array[0][1]
                    )

            prediction = model.predict(
                scaled_input
            )[0]

            used_input = scaled_input


        # ====================================================
        # RISK SCORE
        # ====================================================

        if probability is None:

            if prediction == 1:

                risk_percentage = 100.0

            else:

                risk_percentage = 0.0

        else:

            risk_percentage = probability * 100.0


        # Keep score between 0 and 100

        risk_percentage = max(
            0.0,
            min(
                100.0,
                risk_percentage
            )
        )


        # ====================================================
        # RISK CATEGORY
        # ====================================================

        st.subheader("📊 Patient Result")

        st.metric(
            "Model-Estimated Risk Score",
            f"{risk_percentage:.2f}%"
        )


        if risk_percentage < 20:

            risk_category = (
                "Lower model-estimated risk"
            )

            st.success(
                f"🟢 {risk_category}"
            )


        elif risk_percentage < 50:

            risk_category = (
                "Intermediate model-estimated risk"
            )

            st.warning(
                f"🟡 {risk_category}"
            )


        else:

            risk_category = (
                "Higher model-estimated risk"
            )

            st.error(
                f"🔴 {risk_category}"
            )


        # ====================================================
        # EXPLANATION
        # ====================================================

        st.info(
            "This percentage is a machine-learning model score. "
            "It is not a clinically validated cancer probability "
            "and does not confirm or rule out cancer."
        )


        # ====================================================
        # PERSONALIZED GUIDANCE
        # ====================================================

        st.divider()

        st.subheader(
            "🌿 Personalized Patient Guidance"
        )

        guidance = generate_patient_guidance(
            input_data,
            risk_percentage
        )

        for item in guidance:

            st.write(
                "• " + item
            )


        # ====================================================
        # PATIENT FACTOR SUMMARY
        # ====================================================

        st.divider()

        st.subheader(
            "📋 Patient Factors"
        )

        summary_col1, summary_col2 = st.columns(2)


        with summary_col1:

            st.write(
                f"**Age:** {age:.0f}"
            )

            st.write(
                f"**Sexual partners:** "
                f"{sexual_partners:.0f}"
            )

            st.write(
                f"**Pregnancies:** "
                f"{pregnancies:.0f}"
            )

            st.write(
                f"**Smoking:** "
                f"{'Yes' if smokes else 'No'}"
            )

            st.write(
                f"**STD history:** "
                f"{'Yes' if stds else 'No'}"
            )


        with summary_col2:

            st.write(
                f"**HPV history:** "
                f"{'Yes' if std_values['STDs:HPV'] else 'No'}"
            )

            st.write(
                f"**Hormonal contraceptive:** "
                f"{'Yes' if hormonal else 'No'}"
            )

            st.write(
                f"**IUD history:** "
                f"{'Yes' if iud else 'No'}"
            )

            st.write(
                f"**STD diagnoses:** "
                f"{std_diagnosis:.0f}"
            )

            st.write(
                f"**Risk category:** "
                f"{risk_category}"
            )


        # ====================================================
        # MODEL INPUT INFORMATION
        # ====================================================

        with st.expander(
            "🔧 View model input features"
        ):

            st.dataframe(
                raw_input,
                use_container_width=True
            )


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        st.error(
            "❌ Prediction error occurred."
        )

        st.warning(
            "The model and application feature configuration "
            "do not appear to match."
        )

        st.code(
            str(e)
        )

        st.info(
            "Please check that model.pkl was trained using the "
            "same dataset/features as this application."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🩺 Cervical Cancer Early-Risk Prediction System | "
    "Machine Learning + Personalized Guidance"
)

st.caption(
    "Educational/project demonstration only. "
    "Medical decisions should be made with a qualified healthcare professional."
)