import os
import tempfile
import joblib
import shutil
import dask.dataframe as dd
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Security
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from starlette import status
from dask.distributed import Client, LocalCluster

from app.services.dicom_manager import DicomManager
from app.services.patient_model_predictor import PatientModelPredictor
from app.utils.directory_manager import DirectoryManager
from app.services.auxiliary import Auxiliary
from app.core.config import settings

# API kulcs definíció a biztonságos feltöltéshez
API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

app = FastAPI(title="CT Prediction API")
dm = DicomManager()
auxiliary = Auxiliary()
predictor = PatientModelPredictor()

_model_cache = {"model": None, "last_modified": 0}


async def get_api_key(header_key: str = Security(api_key_header)):
    """
    Ellenőrzi az X-API-KEY fejlécet a konfigurációban megadott kulccsal szemben.
    """
    if header_key == settings.API_KEY:
        return header_key
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Hozzáférés megtagadva: Érvénytelen API kulcs."
    )


def load_model():
    """
    Betölti az XGBoost modellt és figyeli a fájl módosítási idejét az automatikus frissítéshez.
    """
    path = settings.MODEL_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(f"Modell fájl nem található: {path}")

    current_modified = os.path.getmtime(path)

    if _model_cache["model"] is None or current_modified != _model_cache["last_modified"]:
        _model_cache["model"] = joblib.load(path)
        _model_cache["last_modified"] = current_modified

    return _model_cache["model"]


@app.get("/")
async def root():
    return {"status": "online", "message": "CT Image Predictor API"}


@app.post("/predict")
async def predict(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    """
    Diagnosztikai predikció végrehajtása DICOM és XML adatok alapján.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        dicom_path = os.path.join(temp_dir, file1.filename)
        annotation_path = os.path.join(temp_dir, file2.filename)

        with open(dicom_path, "wb") as buffer:
            buffer.write(await file1.read())
        with open(annotation_path, "wb") as buffer:
            buffer.write(await file2.read())

        try:
            xgboost_model = load_model()
            df = dm.preprocessing_dicom(dicom_path, annotation_path)

            if df is None:
                raise HTTPException(status_code=400, detail="Hiba: Nem megfelelő DICOM pozíció (HFS).")

            ddf = dd.from_pandas(df, npartitions=2)

            with LocalCluster(processes=False, memory_limit=settings.DASK_MEMORY_LIMIT) as cluster:
                with Client(cluster) as client:
                    predictor.set_client_and_model(client, xgboost_model)
                    metrics = predictor.predict_and_evaluate(ddf)

                    serializable_metrics = auxiliary.convert_ndarray_to_list(metrics)
                    return JSONResponse(content={"results": serializable_metrics})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-model")
async def upload_model(
        file: UploadFile = File(...),
        api_key: str = Depends(get_api_key)
):
    """
    Védett végpont a modell frissítéséhez. Minden sikeres feltöltést naplóz a logged.txt fájlba.
    """
    target_path = settings.MODEL_PATH

    try:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        # Fájl mentése
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # NAPLÓZÁS: Az auxiliary osztály log metódusát használva rögzítjük az eseményt
        log_msg = f"SIKERES MODELL FRISSÍTÉS - Fájlnév: {file.filename} -> Cél: {target_path}"
        auxiliary.log(log_msg)

        return {
            "status": "success",
            "message": "Modell frissítve és esemény naplózva.",
            "timestamp": auxiliary.log  # közvetve jelzi a logolás tényét
        }
    except Exception as e:
        error_msg = f"SIKERTELEN MODELL FRISSÍTÉS - Hiba: {str(e)}"
        auxiliary.log(error_msg)
        raise HTTPException(status_code=500, detail=str(e))