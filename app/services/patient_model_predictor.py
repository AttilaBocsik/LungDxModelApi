import os

import numpy as np
import dask.dataframe as dd
from xgboost import dask as dxgb
from dask.distributed import Client
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import recall_score, f1_score, jaccard_score, confusion_matrix, accuracy_score


class PatientModelPredictor:
    def __init__(self):
        self.client = None
        self.model = None

    def set_client_and_model(self, client: Client, model):
        """
        Beállítjuk a Dask kliens és Dask XGBoost modell.
        :param client: Dask Client objektum.
        :param model: A Dask XGBoost modell, amelyet előrejelzésre használunk.
        :return None:
        """
        self.client = client
        self.model = model

    def load_csv_files(self, file_path):
        """
        Betölt egy CSV fájlt Dask DataFrame-be.

        :param file_path: A CSV fájl elérési útja.
        :return: A betöltött Dask DataFrame.
        """
        return dd.read_csv(file_path)

    def predict_and_evaluate(self, data):
        """
        Előrejelzés és metrikák számítása az adatokra.

        :param data: Dask DataFrame az előrejelzésre szánt adatokkal.
        :param target_column: Az oszlop neve, amely a valódi értékeket tartalmazza.
        :return: Számított metrikák (Recall (visszahívás), F1 Score (F1-érték), Jaccard Index,
                 Confusion Matrix (összezavarodási mátrix), Accuracy (pontosság)).
        """

        y_test = data['Label'].astype('int')
        X_test = data.drop(['Label', 'patient_id'], axis=1)
        if X_test.compute().empty:
            return {"rmse": None, "recall": None, "f1": None, "jaccard": None, "conf_matrix": None,
                    "accuracy": None}

        dtest = dxgb.DaskDMatrix(self.client, X_test, y_test)
        # Előrejelzés
        y_pred = dxgb.predict(self.client, self.model, dtest)
        # A dask.array -> numpy.array átalakítás
        y_pred = y_pred.compute()
        # Osztály címkék kiválasztása (argmax az egyes sorokon a legnagyobb valószínűségű címkéért)
        y_pred = np.argmax(y_pred, axis=1)
        y_true = y_test.compute()  # tényleges értékek
        # y_true átalakítása numpy tömbbé
        y_true = y_true.to_numpy()
        # Most már y_true és y_pred is numpy array, így használható a kiértékelési metrikákhoz
        # Metrikák számítása
        rmse = root_mean_squared_error(y_true, y_pred)
        # recall = recall_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro')
        jaccard = jaccard_score(y_true, y_pred, average=None)
        conf_matrix = confusion_matrix(y_true, y_pred)
        # Pontosság kiértékelése
        accuracy = accuracy_score(y_true, y_pred)
        return {"rmse": rmse, "recall": recall, "f1": f1, "jaccard": jaccard, "conf_matrix": conf_matrix,
                "accuracy": accuracy}

    def process_folder(self, folder_path):
        """
       Feldolgozza a mappában lévő összes CSV fájlt, egyenként előrejelzést végez és kiértékel.

       :param folder_path: A mappa elérési útja, ahol a CSV fájlok találhatók.
       :param target_column: Az oszlop neve, amely a valódi értékeket tartalmazza.
       :return: A fájlokra vonatkozó metrikák listája.
       """
        csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
        if not csv_files:
            raise ValueError("A megadott mappában nincsenek CSV fájlok.")

        results = {}
        for file in csv_files:
            data = self.load_csv_files(file)
            metrics = self.predict_and_evaluate(data)
            results[file] = metrics

        return results
