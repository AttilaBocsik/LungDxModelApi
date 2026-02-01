import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "LungDx Model API"

    # A projekt gyökérkönyvtára (visszalépünk az app/core mappából)
    # app/core/config.py -> app/core -> app -> gyökér
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Modell és kategória fájlok abszolút útvonala
    # Így a Dockerben és a Tesztben is ugyanazt az abszolút utat kapjuk meg
    MODEL_PATH: str = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "app", "models", "xgboost_model.pkl"))
    CATEGORY_FILE: str = os.getenv("CATEGORY_FILE", os.path.join(BASE_DIR, "category.txt"))

    DASK_MEMORY_LIMIT: str = "4GB"

    model_config = SettingsConfigDict(case_sensitive=True)


settings = Settings()