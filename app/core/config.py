import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Az alkalmazás globális konfigurációs osztálya, amely a Pydantic BaseSettings-re épül.

    Ez az osztály kezeli a környezeti változókat, az elérési utakat és a modell specifikus
    paramétereit a LungDx Model API-hoz. A beállítások automatikusan felülírhatóak
    környezeti változókkal.

    :ivar PROJECT_NAME: Az API megnevezése.
    :vartype PROJECT_NAME: str
    :ivar BASE_DIR: A projekt abszolút gyökérkönyvtára a fájlrendszerben.
    :vartype BASE_DIR: str
    :ivar CATEGORY_FILE: A CT kép feldolgozásához használt kategória fájl elérési útja.
    :vartype CATEGORY_FILE: str
    :ivar MODEL_PATH: Az előre betanított XGBoost modell (.pkl) elérési útja.
    :vartype MODEL_PATH: str
    :ivar DASK_MEMORY_LIMIT: A Dask scheduler számára fenntartott memória limit.
    :vartype DASK_MEMORY_LIMIT: str
    """
    PROJECT_NAME: str = "LungDx Model API"

    # A projekt gyökérkönyvtára (visszalépünk az app/core mappából)
    # app/core/config.py -> app/core -> app -> gyökér
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Abszolút útvonalak a gyökérhez képest
    CATEGORY_FILE: str = os.path.join(BASE_DIR, "category.txt")
    MODEL_PATH: str = os.path.join(BASE_DIR, "app/models/lung_dx_model_final.pkl")

    DASK_MEMORY_LIMIT: str = "4GB"

    model_config = SettingsConfigDict(case_sensitive=True)


settings = Settings()