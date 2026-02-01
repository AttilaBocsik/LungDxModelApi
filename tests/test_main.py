import pytest
import os
from fastapi.testclient import TestClient
from app.core.config import settings

# Biztosítjuk, hogy legyen category.txt a tesztelés idejére a várt helyen
if not os.path.exists(settings.CATEGORY_FILE):
    with open(settings.CATEGORY_FILE, "w") as f:
        f.write("A\nB\nD\nG")

from app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "online" in response.json()["status"]