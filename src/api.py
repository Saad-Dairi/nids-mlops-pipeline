from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="NIDS ML API", description="API de Détection d'Intrusion Réseau")

# 1. Gestion robuste des chemins (indépendant du dossier depuis lequel on lance uvicorn)
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "xgboost_nids_model.json"
ENCODER_PATH = BASE_DIR / "models" / "label_encoder.pkl"

# 2. Chargement du modèle et récupération des colonnes exactes attendues
try:
    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))
    model.set_params(device="cpu")
    
    encoder = joblib.load(ENCODER_PATH)
    # XGBoost garde en mémoire le nom et l'ordre exact des 52 colonnes d'entraînement !
    EXPECTED_FEATURES = model.get_booster().feature_names
    print(f"Modèle chargé ({len(EXPECTED_FEATURES)} features attendues).")
except Exception as e:
    raise RuntimeError(f"Impossible de démarrer l'API, erreur modèle : {e}")

class NetworkFlow(BaseModel):
    features: Dict[str, float]

@app.get("/")
def read_root():
    return {"message": "NIDS API est en ligne et prête à prédire."}

@app.post("/predict")
def predict_traffic(flow: NetworkFlow):
    # 1. Vérifier s'il manque des colonnes réseau obligatoires
    missing_cols = set(EXPECTED_FEATURES) - set(flow.features.keys())
    if missing_cols:
        raise HTTPException(
            status_code=422,
            detail=f"Colonnes manquantes ({len(missing_cols)}) : {list(missing_cols)[:5]}..."
        )
    
    # 2. Construire le DataFrame dans l'ordre EXACT attendu par XGBoost
    df_input = pd.DataFrame([flow.features])[EXPECTED_FEATURES]
    
    # 3. Vérifier qu'aucune valeur n'est NaN ou Infinie
    if df_input.isin([np.inf, -np.inf, np.nan]).any().any():
        raise HTTPException(
            status_code=422,
            detail="Les données contiennent des valeurs invalides (NaN ou Infini)."
        )

    # 4. Prédiction + Probabilité de confiance
    prediction_encoded = model.predict(df_input)
    probabilities = model.predict_proba(df_input)[0]
    confidence = float(np.max(probabilities))
    
    prediction_label = encoder.inverse_transform(prediction_encoded)[0]
    
    return {
        "status": "success",
        "prediction": prediction_label,
        "confidence": round(confidence, 4)
    }

