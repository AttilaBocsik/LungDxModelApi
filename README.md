# LungDx Model API - CT Képfeldolgozó és Predikciós Rendszer
Ez az API egy előre betanított XGBoost modellt használ tüdő CT felvételek diagnosztikai elemzéséhez. 
A rendszer Dask alapú párhuzamosítást és FastAPI keretrendszert használ a nagy adatmennyiségek hatékony kezeléséhez.


### Mappa struktúra
```text
ct-prediction-service/
├── .github/
│   └── workflows/
│       └── main.yml           # GitHub Actions CI/CD konfiguráció
├── app/
│   ├── __init__.py
│   ├── main.py                # FastAPI alkalmazás és végpontok
│   ├── core/                  # Alapvető konfigurációk (pl. útvonalak, model betöltő)
│   │   ├── __init__.py
│   │   └── config.py
│   ├── services/              # A meglévő logikád (szolgáltatás réteg)
│   │   ├── __init__.py
│   │   ├── dicom_manager.py
│   │   ├── patient_model_predictor.py
│   │   ├── XML_preprocessor.py
│   │   ├── images_to_df.py
│   │   └── auxiliary.py
│   ├── utils/                 # Segédfájlok
│   │   ├── __init__.py
│   │   ├── file_manager.py
│   │   └── directory_manager.py
│   └── models/
│       └── xgboost_model.pkl  # Ide kerül a bináris modell fájl
├── tests/                     # Tesztelési könyvtár
│   ├── __init__.py
│   ├── conftest.py            # Pytest fixture-ök
│   └── test_predict.py        # API és logika tesztek
├── data/                      # Ideiglenes mappa (pl. teszt DICOM-oknak)
├── Dockerfile                 # Docker leíró fájl
├── requirements.txt           # Python függőségek listája
├── category.txt               # A kategóriákat tartalmazó fájl
└── .dockerignore              # Amit ne másoljon bele az image-be (pl. venv, .git)
```

### Szoftver dokumentációja
- Telepítés: 
```bash 
pip install sphinx sphinx-rtd-theme
```
- Sphinx inicializálása
  - A projekt gyökérmappájában (ahol a main.py is van) hozz létre egy docs mappát
```bash
mkdir docs
cd docs
sphinx-quickstart
```

#### Manuális dokumentáció készítés
- A conf.py pontos beállítása
  - A docs/source/conf.py fájlban a sys.path sorát módosítsd az alábbira. Ez biztosítja, hogy a src mappa legyen a csomagok gyökere:
```python
import os
import sys

# Nagyon fontos: a 'src' mappát kell megadni, hogy a 'dicom_labeler' látható legyen
sys.path.insert(0, os.path.abspath('../../src'))

extensions = [
    'sphinx.ext.autodoc',  # Kinyeri a docstringeket a kódból
    'sphinx.ext.napoleon', # Ez kezeli a Google-stílusú kommenteket
    'sphinx.ext.viewcode', # Linket tesz a forráskódhoz
    'sphinx_rtd_theme',    # A modern téma
    'myst_parser',         # README.md integrálása
]

# Téma beállítása
html_theme = 'sphinx_rtd_theme'

# Ha a Sphinx nem találja a PyQt6-ot a gépén, add hozzá ezt:
autodoc_mock_imports = ["PyQt6", "requests", "pydicom"]
```

- Az index.rst frissítése
```text
.. automodule:: dicom_labeler.main
   :members:

.. automodule:: dicom_labeler.ui.main_window
   :members:

.. automodule:: dicom_labeler.ui.viewer_widget
   :members:
```

- Dokumentáció generálása (Build)/ Újra generálás (Windows alatt a docs mappában:)
  - Töröld a régit: 
```bash 
.\make.bat clean
```
  - Futtasd az újat: 
```bash
.\make.bat html
```

#### Élő nézet (sphinx-autobuild)
Prezentáció közben (vagy fejlesztés alatt) nagyon hasznos, ha nem kell folyton lefuttatnod a make html-t.
- Telepítsd: 
```bash
pip install sphinx-autobuild
```
- Futtasd: 
```bash
sphinx-autobuild docs/source docs/build/html
```
- Ez elindít egy helyi szervert (általában a http://127.0.0.1:8000 címen), ami azonnal frissül, amint elmented a kódban a docstringet.

#### API indítás
0. Előfeltétel: Uvicorn telepítése
- A terminálban (vagy a PyCharm Terminal fülén) futtasd:
```bash
pip install uvicorn
````
1. Módszer: Indítás a Terminálból (A leggyorsabb)
- Nyisd meg a PyCharm alján a Terminal fület, és írd be a következőt:
```bash
uvicorn app.main:app --reload
````
- app.main: Ez a D:\GitProjects\LungDxModelApi\app\main.py fájlra mutat.
- :app: Ez a main.py-ban létrehozott app = FastAPI() objektum neve.
- --reload: Automatikusan újraindul a szerver, ha módosítod a kódot (fejlesztés közben életmentő).

#### Gyorsindítás (Docker) (docker-compose.yml).
A legegyszerűbb módja a futtatásnak a Docker Compose használata, amely automatikusan kezeli a függőségeket és a mappák csatolását.
1. Indítás (háttérben):
```bash
docker-compose up -d
```
Az API a http://localhost:8000 címen lesz elérhető.
2. Logok figyelése valós időben:
```bash
docker-compose logs -f
```
3. Leállítás:
```bash
docker-compose down
```

### 🔐 Biztonság és Konfiguráció
A modell frissítése védett végponton keresztül történik. Az API kulcsot a környezeti változók között állíthatod be:
- API_KEY: A modell feltöltéséhez szükséges titkos kulcs (alapértelmezett: titkos-kulcs-123).
- DASK_MEMORY_LIMIT: A feldolgozáshoz használt memória korlátja (pl.: 4GB).
Ezeket a docker-compose.yml fájlban vagy egy .env fájlban módosíthatod.

### 🛠 API Végpontok
1. Diagnosztikai Predikció
POST /predict
Feltöltendő fájlok:
- file1: DICOM képfájl (.dcm)
- file2: Annotációs XML fájl (.xml)
```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'X-API-KEY: titkos-kulcs-123' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file1=@1-08.dcm' \
  -F 'file2=@1-08.xml'
```

2. Modell Frissítése (Védett)
POST /upload-model
- Fejléc: X-API-KEY: <a_te_kulcsod>
- Body: file: Az új lung_dx_model_final.pkl fájl.
```bash
curl -X 'POST' \
  'http://localhost:8000/upload-model' \
  -H 'accept: application/json' \
  -H 'X-API-KEY: titkos-kulcs-123' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@lung_dx_model_final.pkl'
```

#### Swagger
http://localhost:8000/docs

### 📂 Naplózás és Audit
Minden modellfrissítés automatikusan rögzítésre kerül a logged.txt fájlban az orvosi szoftverekre vonatkozó audit követelményeknek megfelelően.
A logokat a konténeren kívül is eléred a projekt gyökérkönyvtárában.

### 🤖 CI/CD Pipeline
A projekt GitHub Actions-t használ:
- Test: Minden push/PR esetén lefutnak a Pytest tesztek.
- Build & Push: A main ágra való push esetén az új Docker image automatikusan elkészül és feltöltődik a GitHub Container Registry-be (GHCR).

### Teszt futtatása
A terminálban (vagy a Docker konténeren belül) egyszerűen add ki ezt a parancsot:
```bash
pytest tests/
```


