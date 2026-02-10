import os
import shutil
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

_model_cache = {"model": None, "last_modified": 0}


def load_model():
    """
    Betölti az XGBoost modellt a lemezről, és gyorsítótárazza azt a memóriába.
    Figyeli a fájl módosítási idejét, és automatikusan újratölti, ha a modellfájl frissült.

    :return: A betöltött joblib modell objektum.
    :rtype: xgboost.XGBClassifier or similar
    :raises FileNotFoundError: Ha a konfigurációban megadott elérési úton nem található a modell.
    """
    path = settings.MODEL_PATH

    # DEBUG: Ha a hiba jelentkezik, ez kiírja, hogy mi van benne valójában
    if not isinstance(path, str):
        raise TypeError(f"HIBA: A settings.MODEL_PATH nem string, hanem {type(path)}! Értéke: {path}")

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
    Fogadja a diagnosztikai fájlokat és végrehajtja a predikciós munkafolyamatot.

    A folyamat lépései:
    1. Ideiglenes könyvtárba menti a feltöltött DICOM és XML fájlokat.
    2. A DicomManager segítségével elvégzi a képi előfeldolgozást és jellemző kinyerést.
    3. Dask LocalCluster-t indít a párhuzamosított XGBoost predikcióhoz.
    4. Az eredményeket JSON formátumban adja vissza.

    :param file1: A vizsgálandó szelet DICOM (.dcm) fájlja.
    :param file2: Az orvosi annotációkat tartalmazó XML fájl.
    :return: A modell által becsült valószínűségek és osztályozási eredmények.
    :rtype: fastapi.responses.JSONResponse
    :raises HTTPException: 400-as hiba nem megfelelő DICOM pozíció (HFS) esetén, 500-as hiba feldolgozási hiba esetén.
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


@app.post("/upload-model")
async def upload_model(file: UploadFile = File(...)):
    """
    Lehetővé teszi az új XGBoost modell (.pkl) feltöltését az API-hoz.
    A fájlt a megadott app/models/lung_dx_model_final.pkl útvonalra menti el.

    :param file: A feltöltendő .pkl kiterjesztésű modell fájl.
    :return: Visszaigazolás a sikeres feltöltésről.
    :rtype: dict
    """
    target_path = settings.MODEL_PATH

    try:
        # Biztosítjuk, hogy a célkönyvtár létezik
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        # A feltöltött fájl mentése
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {"status": "success", "message": f"Modell sikeresen frissítve: {target_path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hiba a modell mentése során: {str(e)}")