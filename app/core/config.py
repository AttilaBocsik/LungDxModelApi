import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Projekt alapinformációk
    PROJECT_NAME: str = "CT Image XGBoost Predictor"
    API_V1_STR: str = "/api/v1"

    # Elérési utak (a Docker struktúrához igazítva)
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Modell beállítások
    # A gyökérkönyvtárhoz képest nézzük (app/models/...)
    MODEL_PATH: str = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "models", "xgboost_model.pkl"))
    CATEGORY_FILE: str = os.getenv("CATEGORY_FILE", "category.txt")

    # Dask / Erőforrás beállítások
    DASK_MEMORY_LIMIT: str = os.getenv("DASK_MEMORY_LIMIT", "4GB")
    DASK_PROCESSES: bool = False  # Orvosi képfeldolgozásnál a False gyakran stabilabb szálkezelést ad

    # Szerver beállítások
    HOST: str = os.getenv("SERVER_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("SERVER_PORT", 8000))

    class Config:
        case_sensitive = True


# Példányosítjuk, hogy bárhonnan importálható legyen
settings = Settings()