# LungDx Model API - CT Image Processing and Prediction System
[Magyar nyelvű leírás itt / Hungarian version here](README.hu.md)

This API utilizes a pre-trained XGBoost model for the diagnostic analysis of lung CT scans. 
The system leverages Dask-based parallelization and the FastAPI framework to efficiently handle large datasets.

### Project Structure
```text
ct-prediction-service/
├── .github/
│   └── workflows/
│       └── main.yml           # GitHub Actions CI/CD configuration
├── app/
│   ├── __init__.py
│   ├── main.py                # FastAPI application and endpoints
│   ├── core/                  # Core configurations (e.g., routing, model loader)
│   │   ├── __init__.py
│   │   └── config.py
│   ├── services/              # Logic layer (services)
│   │   ├── __init__.py
│   │   ├── dicom_manager.py
│   │   ├── patient_model_predictor.py
│   │   ├── XML_preprocessor.py
│   │   ├── images_to_df.py
│   │   └── auxiliary.py
│   ├── utils/                 # Utility files
│   │   ├── __init__.py
│   │   ├── file_manager.py
│   │   └── directory_manager.py
│   └── models/
│       └── xgboost_model.pkl  # Binary model file location
├── tests/                     # Testing directory
│   ├── __init__.py
│   ├── conftest.py            # Pytest fixtures
│   └── test_predict.py        # API and logic tests
├── data/                      # Temporary folder (e.g., for test DICOMs)
├── Dockerfile                 # Docker descriptor file
├── requirements.txt           # Python dependency list
├── category.txt               # Category definition file
└── .dockerignore              # Files to exclude from the image (e.g., venv, .git)
```
### Documentation
- Installation: 
```bash
pip install sphinx sphinx-rtd-theme
```
- Initializing Sphinx:
    In the project root (where `main.py` is located), create a `docs` folder:
```bash
mkdir docs
cd docs
sphinx-quickstart
```
### Manual Documentation Build
- Configure conf.py:
    In docs/source/conf.py, modify the sys.path to ensure the src folder is visible:

```python
import os
import sys

sys.path.insert(0, os.path.abspath('../../src'))

extensions = [
    'sphinx.ext.autodoc',  # Extracts docstrings from code
    'sphinx.ext.napoleon', # Handles Google-style comments
    'sphinx.ext.viewcode', # Adds links to source code
    'sphinx_rtd_theme',    # Modern theme
    'myst_parser',         # README.md integration
]

html_theme = 'sphinx_rtd_theme'
autodoc_mock_imports = ["PyQt6", "requests", "pydicom"]
```
### Live Preview (sphinx-autobuild)
- Install: 
```bash
pip install sphinx-autobuild
```
- **Run:** 
```bash
sphinx-autobuild docs/source docs/build/html
```
The server will be available at http://127.0.0.1:8000, updating automatically upon saving docstrings.

### Starting the API
1. Method: Terminal (Fastest)
- Prerequisite: Install Uvicorn
```bash
pip install uvicorn
```
- Run:
```bash
uvicorn app.main:app --reload
```
2. Method: Quickstart with Docker
- Start (Background):
```bash
docker-compose up -d
```
The API will be available at http://localhost:8000.

### Security and Configuration
Model updates are handled via protected endpoints. You can configure the following environment variables:
- API_KEY: Secret key required for model uploads (default: titkos-kulcs-123).
- DASK_MEMORY_LIMIT: Memory limit for processing (e.g., 4GB).

### API Endpoints
1. Diagnostic Prediction
POST /predict
Files to upload:
- file1: DICOM image file (.dcm)
- file2: Annotation XML file (.xml)
```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'X-API-KEY: titkos-kulcs-123' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file1=@sample.dcm' \
  -F 'file2=@sample.xml'
```
2. Update Model (Protected)
POST /upload-model

### Logging and Audit
All model updates are automatically recorded in the logged.txt file, complying with medical software audit requirements.

### CI/CD Pipeline
The project utilizes GitHub Actions:
- Test: Runs Pytest on every push/PR.
- Build & Push: Automatically builds and pushes a Docker image to the GitHub Container Registry (GHCR) upon pushing to the main branch.

### Running Tests
```bash
pytest tests/
```