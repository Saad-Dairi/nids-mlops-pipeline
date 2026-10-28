from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram

app = FastAPI(title="NIDS ML API", description="API de Détection d'Intrusion Réseau")

# 1. Métriques HTTP standards (exposées sur la route /metrics)
Instrumentator().instrument(app).expose(app)

# 2. Métriques ML personnalisées
PREDICTION_COUNTER = Counter(
    "nids_predictions_total",
    "Nombre total de prédictions par classe de trafic réseau",
    ["predicted_class"]
)

CONFIDENCE_HISTOGRAM = Histogram(
    "nids_prediction_confidence",
    "Distribution des scores de confiance du modèle XGBoost",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "xgboost_nids_model.json"
ENCODER_PATH = BASE_DIR / "models" / "label_encoder.pkl"

try:
    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))
    model.set_params(device="cpu")
    
    encoder = joblib.load(ENCODER_PATH)
    EXPECTED_FEATURES = model.get_booster().feature_names
    print(f"✅ Modèle chargé ({len(EXPECTED_FEATURES)} features attendues).")
except Exception as e:
    raise RuntimeError(f"❌ Impossible de démarrer l'API, erreur modèle : {e}")

class NetworkFlow(BaseModel):
    features: Dict[str, float]

@app.get("/")
def read_root():
    return {"message": "NIDS API est en ligne et prête à prédire."}

@app.post("/predict")
def predict_traffic(flow: NetworkFlow):
    missing_cols = set(EXPECTED_FEATURES) - set(flow.features.keys())
    if missing_cols:
        raise HTTPException(
            status_code=422,
            detail=f"Colonnes manquantes ({len(missing_cols)}) : {list(missing_cols)[:5]}..."
        )
    
    df_input = pd.DataFrame([flow.features])[EXPECTED_FEATURES]
    
    if df_input.isin([np.inf, -np.inf, np.nan]).any().any():
        raise HTTPException(
            status_code=422,
            detail="Les données contiennent des valeurs invalides (NaN ou Infini)."
        )

    prediction_encoded = model.predict(df_input)
    probabilities = model.predict_proba(df_input)[0]
    confidence = float(np.max(probabilities))
    
    prediction_label = str(encoder.inverse_transform(prediction_encoded)[0])
    
    # Enregistrement des métriques dans Prometheus
    PREDICTION_COUNTER.labels(predicted_class=prediction_label).inc()
    CONFIDENCE_HISTOGRAM.observe(confidence)
    
    return {
        "status": "success",
        "prediction": prediction_label,
        "confidence": round(confidence, 4)
    }