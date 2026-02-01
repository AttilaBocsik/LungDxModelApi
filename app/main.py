import os
import tempfile
import joblib
import dask.dataframe as dd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from dask.distributed import Client, LocalCluster

from app.services.dicom_manager import DicomManager
from app.services.patient_model_predictor import PatientModelPredictor
from app.utils.directory_manager import DirectoryManager
from app.services.auxiliary import Auxiliary
from app.core.config import settings

app = FastAPI(title="CT Prediction API")
dm = DicomManager()
auxiliary = Auxiliary()
predictor = PatientModelPredictor()

model_path = joblib.load(settings.MODEL_PATH)
_model_cache = {"model": None, "last_modified": 0}


def load_model():
    """Modell betöltése és frissítése, ha a fájl módosult."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modell fájl nem található: {model_path}")

    current_modified = os.path.getmtime(Mmodel_path)
    if _model_cache["model"] is None or current_modified != _model_cache["last_modified"]:
        _model_cache["model"] = joblib.load(model_path)
        _model_cache["last_modified"] = current_modified
    return _model_cache["model"]


@app.get("/")
async def root():
    return {"status": "online", "message": "CT Image Predictor API"}


@app.post("/predict")
async def predict(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    """
    DICOM és annotációs fájl fogadása, predikció futtatása.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        dicom_path = os.path.join(temp_dir, file1.filename)
        annotation_path = os.path.join(temp_dir, file2.filename)

        # Fájlok mentése a diszkre a feldolgozáshoz
        with open(dicom_path, "wb") as buffer:
            buffer.write(await file1.read())
        with open(annotation_path, "wb") as buffer:
            buffer.write(await file2.read())

        try:
            xgboost_model = load_model()

            # Képfeldolgozás (Preprocessing)
            df = dm.preprocessing_dicom(dicom_path, annotation_path)
            if df is None:
                raise HTTPException(status_code=400, detail="Hiba: Nem megfelelő DICOM pozíció (HFS szükséges).")

            ddf = dd.from_pandas(df, npartitions=2)

            # Dask alapú predikció
            with LocalCluster(processes=False, memory_limit='4GB') as cluster:
                with Client(cluster) as client:
                    predictor.set_client_and_model(client, xgboost_model)
                    metrics = predictor.predict_and_evaluate(ddf)

                    if not metrics:
                        raise HTTPException(status_code=500, detail="Üres eredmények a predikció után.")

                    # JSON kompatibilis formátumra alakítás
                    serializable_metrics = auxiliary.convert_ndarray_to_list(metrics)
                    return JSONResponse(content={"results": serializable_metrics})

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))