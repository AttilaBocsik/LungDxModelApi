from fastapi.testclient import TestClient
from app.main import app
import pytest

client = TestClient(app)

def test_root():
    """Ellenőrzi, hogy az API elérhető-e."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_predict_validation():
    """Ellenőrzi, hogy fájlok nélkül hibaüzenetet kapunk-e."""
    response = client.post("/predict")
    # A FastAPI automatikusan 422-es kódot ad, ha hiányoznak a kötelező paraméterek
    assert response.status_code == 422