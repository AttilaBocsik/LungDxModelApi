# Tüdő daganat XGBoost model kiszolgáló API
- API elérés: http://localhost:8000/predict
- Másik gépről: http://IP-cim:8000/predict

## Mappa struktúra
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

## Directory haszmálat
- python
```python
from directory_tools import DirectoryManager

dm = DirectoryManager("teszt_mappa")
dm.create_directory()
print(dm.is_directory())  # True
print(dm.is_empty())      # True
T�lts�k meg valamivel, majd t�r�lj�k
with open("teszt_mappa/sample.txt", "w") as f:
    f.write("Hell� vil�g!")

print(dm.is_empty())      # False

dm.clear_directory()
print(dm.is_empty())      # True

dm.delete_directory()
```
## FileManager haszn�lat
```python
from directory_tools import FileManager

fm = FileManager("teszt_fajl.txt")
fm.create_file("Ez egy tesztf�jl.\n")
fm.append_to_file("Hozz�f?z�tt sor.\n")

if fm.file_exists():
    content = fm.read_file()
    print("F�jl tartalma:\n", content)

fm.delete_file()
```
### predict végpont hívás
```python
import requests
url = "http://127.0.0.1:5000/predict"
payload = {}
files=[
  ('file1',('1-13.dcm',open('/D:/GitProjects/DicamManagerProject/Data/Train/DICOM/Lung_Dx-A0001/04-04-2007-NA-Chest-07990/2.000000-5mm-40805/1-13.dcm','rb'),'application/octet-stream')),
  ('file2',('1.3.6.1.4.1.14519.5.2.1.6655.2359.217649008723153656849717286154.xml',open('/D:/GitProjects/DicamManagerProject/Data/Train/ANNOTATION/A0001/1.3.6.1.4.1.14519.5.2.1.6655.2359.217649008723153656849717286154.xml','rb'),'text/xml'))
]
headers = {}
response = requests.request("POST", url, headers=headers, data=payload, files=files)
print(response.text)

```
