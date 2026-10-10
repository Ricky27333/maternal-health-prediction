from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd
from pathlib import Path

# Setup FastAPI
app = FastAPI(title="Maternal Health API")

# Allow CORS for frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Files
BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "maternal_health_decision_tree.pkl"
MAPPING_FILE = BASE_DIR / "complete_mapping.csv"

# Load Model
with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

# Load Mappings
CATEGORICAL_COLUMNS = [
    "Gravida", "TT_Vaccination_Dose", "Anemia_Status", "Jaundice_Status",
    "Fetal_Position", "Fetal_Movement", "Urine_Albumin", "Urine_Sugar",
    "VDRL", "HRsAG",
]

mapping_df = pd.read_csv(MAPPING_FILE)
mappings = {}
for col in CATEGORICAL_COLUMNS:
    encoded_col = col + "_n"
    pairs = mapping_df[[col, encoded_col]].dropna().drop_duplicates()
    mappings[col] = {str(row[col]): int(row[encoded_col]) for _, row in pairs.iterrows()}

FEATURES = [
    "Age", "Gravida", "TT_Vaccination_Dose", "Gestational_Age_Weeks",
    "Weight_kg", "Height", "Anemia_Status", "Jaundice_Status",
    "Fetal_Position", "Fetal_Movement", "Fetal_Heart_Rate",
    "Urine_Albumin", "Urine_Sugar", "VDRL", "HRsAG",
    "Gravida_n", "TT_Vaccination_Dose_n", "Anemia_Status_n",
    "Jaundice_Status_n", "Fetal_Position_n", "Fetal_Movement_n",
    "Urine_Albumin_n", "Urine_Sugar_n", "VDRL_n", "HRsAG_n",
    "Systolic_BP", "Diastolic_BP",
]

# Define Input Data Model
class MaternalInput(BaseModel):
    age: float
    gravida: str
    tt_dose: str
    gestational_age: float
    weight: float
    height: float
    systolic_bp: float
    diastolic_bp: float
    anemia: str
    jaundice: str
    fetal_position: str
    fetal_movement: str
    fetal_heart_rate: float
    urine_albumin: str
    urine_sugar: str
    vdrl: str
    hrsag: str

@app.get("/")
def read_root():
    return {"status": "Maternal Health API is Live!"}

@app.post("/predict")
def predict_risk(data: MaternalInput):
    try:
        encoded = {
            "Gravida": mappings["Gravida"][data.gravida],
            "TT_Vaccination_Dose": mappings["TT_Vaccination_Dose"][data.tt_dose],
            "Anemia_Status": mappings["Anemia_Status"][data.anemia],
            "Jaundice_Status": mappings["Jaundice_Status"][data.jaundice],
            "Fetal_Position": mappings["Fetal_Position"][data.fetal_position],
            "Fetal_Movement": mappings["Fetal_Movement"][data.fetal_movement],
            "Urine_Albumin": mappings["Urine_Albumin"][data.urine_albumin],
            "Urine_Sugar": mappings["Urine_Sugar"][data.urine_sugar],
            "VDRL": mappings["VDRL"][data.vdrl],
            "HRsAG": mappings["HRsAG"][data.hrsag],
        }

        row = {
            "Age": float(data.age),
            "Gravida": float(encoded["Gravida"]),
            "TT_Vaccination_Dose": float(encoded["TT_Vaccination_Dose"]),
            "Gestational_Age_Weeks": float(data.gestational_age),
            "Weight_kg": float(data.weight),
            "Height": float(data.height),
            "Anemia_Status": float(encoded["Anemia_Status"]),
            "Jaundice_Status": float(encoded["Jaundice_Status"]),
            "Fetal_Position": float(encoded["Fetal_Position"]),
            "Fetal_Movement": float(encoded["Fetal_Movement"]),
            "Fetal_Heart_Rate": float(data.fetal_heart_rate),
            "Urine_Albumin": float(encoded["Urine_Albumin"]),
            "Urine_Sugar": float(encoded["Urine_Sugar"]),
            "VDRL": float(encoded["VDRL"]),
            "HRsAG": float(encoded["HRsAG"]),
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
            "Systolic_BP": float(data.systolic_bp),
            "Diastolic_BP": float(data.diastolic_bp),
        }

        df = pd.DataFrame([[row[c] for c in FEATURES]], columns=FEATURES)
        pred = int(model.predict(df)[0])
        
        return {
            "success": True,
            "prediction_value": pred,
            "prediction_text": "High-Risk Pregnancy" if pred == 1 else "Normal (Low-Risk)"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
