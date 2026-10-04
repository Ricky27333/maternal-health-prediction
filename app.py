import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# FILES
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "maternal_health_decision_tree.pkl"
MAPPING_FILE = BASE_DIR / "complete_mapping.csv"

# These are the EXACT 27 features used by your trained model.
FEATURES = [
    "Age",
    "Gravida",
    "TT_Vaccination_Dose",
    "Gestational_Age_Weeks",
    "Weight_kg",
    "Height",
    "Anemia_Status",
    "Jaundice_Status",
    "Fetal_Position",
    "Fetal_Movement",
    "Fetal_Heart_Rate",
    "Urine_Albumin",
    "Urine_Sugar",
    "VDRL",
    "HRsAG",
    "Gravida_n",
    "TT_Vaccination_Dose_n",
    "Anemia_Status_n",
    "Jaundice_Status_n",
    "Fetal_Position_n",
    "Fetal_Movement_n",
    "Urine_Albumin_n",
    "Urine_Sugar_n",
    "VDRL_n",
    "HRsAG_n",
    "Systolic_BP",
    "Diastolic_BP",
]

CATEGORICAL_COLUMNS = [
    "Gravida",
    "TT_Vaccination_Dose",
    "Anemia_Status",
    "Jaundice_Status",
    "Fetal_Position",
    "Fetal_Movement",
    "Urine_Albumin",
    "Urine_Sugar",
    "VDRL",
    "HRsAG",
]


# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)


# ------------------------------------------------------------
# LOAD YOUR MAPPING CSV
# ------------------------------------------------------------
@st.cache_data
def load_mappings():
    mapping_df = pd.read_csv(MAPPING_FILE)

    mappings = {}

    for col in CATEGORICAL_COLUMNS:
        encoded_col = col + "_n"

        pairs = (
            mapping_df[[col, encoded_col]]
            .dropna()
            .drop_duplicates()
        )

        mappings[col] = {
            str(row[col]): int(row[encoded_col])
            for _, row in pairs.iterrows()
        }

    return mappings


def get_options(mapping, column):
    """Return the human-readable values for a dropdown."""
    return list(mapping[column].keys())


# ------------------------------------------------------------
# CREATE MODEL INPUT
# ------------------------------------------------------------
def create_input(
    age,
    gravida,
    tt_dose,
    gestational_age,
    weight,
    height,
    systolic_bp,
    diastolic_bp,
    anemia,
    jaundice,
    fetal_position,
    fetal_movement,
    fetal_heart_rate,
    urine_albumin,
    urine_sugar,
    vdrl,
    hrsag,
    mappings,
):
    # Convert UI values to the exact numeric mappings used in training.
    encoded = {
        "Gravida": mappings["Gravida"][gravida],
        "TT_Vaccination_Dose": mappings["TT_Vaccination_Dose"][tt_dose],
        "Anemia_Status": mappings["Anemia_Status"][anemia],
        "Jaundice_Status": mappings["Jaundice_Status"][jaundice],
        "Fetal_Position": mappings["Fetal_Position"][fetal_position],
        "Fetal_Movement": mappings["Fetal_Movement"][fetal_movement],
        "Urine_Albumin": mappings["Urine_Albumin"][urine_albumin],
        "Urine_Sugar": mappings["Urine_Sugar"][urine_sugar],
        "VDRL": mappings["VDRL"][vdrl],
        "HRsAG": mappings["HRsAG"][hrsag],
    }

    row = {
        "Age": float(age),
        "Gravida": float(encoded["Gravida"]),
        "TT_Vaccination_Dose": float(encoded["TT_Vaccination_Dose"]),
        "Gestational_Age_Weeks": float(gestational_age),
        "Weight_kg": float(weight),
        "Height": float(height),
        "Anemia_Status": float(encoded["Anemia_Status"]),
        "Jaundice_Status": float(encoded["Jaundice_Status"]),
        "Fetal_Position": float(encoded["Fetal_Position"]),
        "Fetal_Movement": float(encoded["Fetal_Movement"]),
        "Fetal_Heart_Rate": float(fetal_heart_rate),
        "Urine_Albumin": float(encoded["Urine_Albumin"]),
        "Urine_Sugar": float(encoded["Urine_Sugar"]),
        "VDRL": float(encoded["VDRL"]),
        "HRsAG": float(encoded["HRsAG"]),

        # Your trained model expects BOTH the original encoded
        # categorical columns and the *_n columns.
        "Gravida_n": float(encoded["Gravida"]),
        "TT_Vaccination_Dose_n": float(encoded["TT_Vaccination_Dose"]),
        "Anemia_Status_n": float(encoded["Anemia_Status"]),
        "Jaundice_Status_n": float(encoded["Jaundice_Status"]),
        "Fetal_Position_n": float(encoded["Fetal_Position"]),
        "Fetal_Movement_n": float(encoded["Fetal_Movement"]),
        "Urine_Albumin_n": float(encoded["Urine_Albumin"]),
        "Urine_Sugar_n": float(encoded["Urine_Sugar"]),
        "VDRL_n": float(encoded["VDRL"]),
        "HRsAG_n": float(encoded["HRsAG"]),

        "Systolic_BP": float(systolic_bp),
        "Diastolic_BP": float(diastolic_bp),
    }

    # IMPORTANT: exact same feature order as model training.
    return pd.DataFrame(
        [[row[column] for column in FEATURES]],
        columns=FEATURES,
    )


# ------------------------------------------------------------
# STREAMLIT PAGE
# ------------------------------------------------------------
st.set_page_config(
    page_title="Maternal Health Risk Prediction",
    page_icon="🩺",
    layout="centered",
)

st.title("🩺 Maternal Health Risk Prediction")
st.write(
    "Enter the maternal and fetal parameters and click Predict."
)

st.info(
    "Research/demo use only. This prediction is not a medical "
    "diagnosis and should not replace advice from a qualified healthcare professional."
)

# Load files
try:
    model = load_model()
    mappings = load_mappings()
except Exception as e:
    st.error("Model or mapping file could not be loaded.")
    st.exception(e)
    st.stop()


# ------------------------------------------------------------
# USER INPUT
# ------------------------------------------------------------
with st.form("maternal_health_form"):

    st.subheader("Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age (years)",
            min_value=10.0,
            max_value=60.0,
            value=25.0,
            step=1.0,
        )

        weight = st.number_input(
            "Weight (kg)",
            min_value=20.0,
            max_value=200.0,
            value=60.0,
            step=0.5,
        )

        height = st.number_input(
            "Height (feet)",
            min_value=3.0,
            max_value=8.0,
            value=5.2,
            step=0.1,
        )

        gestational_age = st.number_input(
            "Gestational Age (weeks)",
            min_value=1.0,
            max_value=45.0,
            value=38.0,
            step=1.0,
        )

        fetal_heart_rate = st.number_input(
            "Fetal Heart Rate",
            min_value=50.0,
            max_value=250.0,
            value=140.0,
            step=1.0,
        )

    with col2:
        gravida = st.selectbox(
            "Gravida",
            get_options(mappings, "Gravida"),
        )

        tt_dose = st.selectbox(
            "TT Vaccination Dose",
            get_options(mappings, "TT_Vaccination_Dose"),
        )

        systolic_bp = st.number_input(
            "Systolic BP",
            min_value=50.0,
            max_value=250.0,
            value=100.0,
            step=1.0,
        )

        diastolic_bp = st.number_input(
            "Diastolic BP",
            min_value=30.0,
            max_value=180.0,
            value=60.0,
            step=1.0,
        )

    st.subheader("Clinical Parameters")

    col1, col2 = st.columns(2)

    with col1:
        anemia = st.selectbox(
            "Anemia Status",
            get_options(mappings, "Anemia_Status"),
        )

        jaundice = st.selectbox(
            "Jaundice Status",
            get_options(mappings, "Jaundice_Status"),
        )

        fetal_position = st.selectbox(
            "Fetal Position",
            get_options(mappings, "Fetal_Position"),
        )

        fetal_movement = st.selectbox(
            "Fetal Movement",
            get_options(mappings, "Fetal_Movement"),
        )

        urine_albumin = st.selectbox(
            "Urine Albumin",
            get_options(mappings, "Urine_Albumin"),
        )

    with col2:
        urine_sugar = st.selectbox(
            "Urine Sugar",
            get_options(mappings, "Urine_Sugar"),
        )

        vdrl = st.selectbox(
            "VDRL",
            get_options(mappings, "VDRL"),
        )

        hrsag = st.selectbox(
            "HRsAG",
            get_options(mappings, "HRsAG"),
        )

    predict_button = st.form_submit_button(
        "🔍 Predict",
        use_container_width=True,
    )


# ------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------
if predict_button:

    try:
        input_data = create_input(
            age=age,
            gravida=gravida,
            tt_dose=tt_dose,
            gestational_age=gestational_age,
            weight=weight,
            height=height,
            systolic_bp=systolic_bp,
            diastolic_bp=diastolic_bp,
            anemia=anemia,
            jaundice=jaundice,
            fetal_position=fetal_position,
            fetal_movement=fetal_movement,
            fetal_heart_rate=fetal_heart_rate,
            urine_albumin=urine_albumin,
            urine_sugar=urine_sugar,
            vdrl=vdrl,
            hrsag=hrsag,
            mappings=mappings,
        )

        prediction = int(model.predict(input_data)[0])

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("⚠️ High-Risk Pregnancy: YES")
        else:
            st.success("✅ High-Risk Pregnancy: NO")

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)
