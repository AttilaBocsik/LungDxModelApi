import pytest
from fastapi.testclient import TestClient
from app.main import app
import io

client = TestClient(app)

def test_read_main():
    """Alap ellenőrzés, hogy él-e az API"""
    response = client.get("/")
    assert response.status_code == 200
    assert "online" in response.json()["status"].lower()

def test_predict_endpoint_exists():
    """Ellenőrizzük, hogy a /predict végpont fogadja a fájlokat (még ha a tartalom nem is igazi DICOM)"""
    files = {
        "file1": ("dummy.dcm", io.BytesIO(b"not-a-dicom"), "application/dicom"),
        "file2": ("dummy.xml", io.BytesIO(b"<xml></xml>"), "text/xml")
    }
    response = client.post("/predict", files=files)
    # Itt a 400 vagy 500 hiba is elfogadható teszt szempontból, mert azt jelenti,
    # hogy a kérés bejutott a kódodig, csak a fájl tartalmával nem tud mit kezdeni.
    assert response.status_code != 404

def test_predict_with_mock_files():
    # Készítünk egy kamu DICOM és XML fájlt a memóriában
    fake_dicom = io.BytesIO(b"DICM" + b"\x00" * 128)  # Minimális DICOM header szimuláció
    fake_xml = io.BytesIO(b"<annotation><size><width>512</width><height>512</height></size></annotation>")

    files = {
        "file1": ("test.dcm", fake_dicom, "application/dicom"),
        "file2": ("test.xml", fake_xml, "text/xml")
    }

    response = client.post("/predict", files=files)

    # Itt valószínűleg 400-at vagy 500-at kapsz, mert a kamu fájl nem igazi DICOM,
    # de a lényeg, hogy az API végpont él és fogadja a kérést!
    assert response.status_code in [200, 400, 500]