import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings
import os

client = TestClient(app)


# 1. Alap online állapot ellenőrzése
def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"


# 2. Biztonsági teszt: Modell feltöltése kulcs nélkül (Hiba várt)
def test_upload_model_no_key():
    response = client.post("/upload-model")
    assert response.status_code == 403  # Forbidden


# 3. Biztonsági teszt: Modell feltöltése rossz kulccsal
def test_upload_model_invalid_key():
    response = client.post(
        "/upload-model",
        headers={"X-API-KEY": "rossz-kulcs"},
        files={"file": ("model.pkl", b"dummy_content")}
    )
    assert response.status_code == 403
    assert "Érvénytelen API kulcs" in response.json()["detail"]


# 4. Funkcionális teszt: Sikeres modell feltöltés és logolás
def test_upload_model_success(tmp_path):
    # Ideiglenesen átállítjuk a modell útvonalát a teszthez
    test_model_path = tmp_path / "test_model.pkl"
    settings.MODEL_PATH = str(test_model_path)

    test_content = b"fake-xgboost-model-data"

    response = client.post(
        "/upload-model",
        headers={"X-API-KEY": settings.API_KEY},
        files={"file": ("model.pkl", test_content)}
    )

    assert response.status_code == 200
    assert os.path.exists(test_model_path)

    # Ellenőrizzük, hogy belekerült-e a logba (a logged.txt-t is nézzük)
    with open("logged.txt", "r") as f:
        logs = f.read()
        assert "SIKERES MODELL FRISSÍTÉS" in logs


# 5. Predikció teszt: Fájlok hiánya esetén
def test_predict_no_files():
    response = client.post("/predict")
    assert response.status_code == 422  # Validation Error (FastAPI alapból dobja)