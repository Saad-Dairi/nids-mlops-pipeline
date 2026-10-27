from fastapi.testclient import TestClient
from src.api import app, EXPECTED_FEATURES

client = TestClient(app)

def test_health_check():
    """Vérifie que l'API répond bien sur la route principale."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "NIDS API est en ligne et prête à prédire."}

def test_predict_valid_flow():
    """Vérifie qu'un flux réseau complet (52 features à 0.0) renvoie une prédiction valide."""
    dummy_features = {feature: 0.0 for feature in EXPECTED_FEATURES}
    response = client.post("/predict", json={"features": dummy_features})
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "prediction" in data
    assert 0.0 <= data["confidence"] <= 1.0

def test_predict_missing_features():
    """Vérifie que l'API rejette proprement (422) une requête incomplète."""
    incomplete_features = {"Destination Port": 80.0, "Flow Duration": 100.0}
    response = client.post("/predict", json={"features": incomplete_features})
    
    assert response.status_code == 422
    assert "Colonnes manquantes" in response.json()["detail"]