import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "LungDx Model API"

    # A projekt gyökérkönyvtára (visszalépünk az app/core mappából)
    # app/core/config.py -> app/core -> app -> gyökér
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Abszolút útvonalak a gyökérhez képest
    CATEGORY_FILE: str = os.path.join(BASE_DIR, "category.txt")
    MODEL_PATH: str = os.path.join(BASE_DIR, "xgboost_model.pkl")

    DASK_MEMORY_LIMIT: str = "4GB"

    model_config = SettingsConfigDict(case_sensitive=True)


settings = Settings()