# Karcsú, de stabil Debian-alapú Python kép
FROM python:3.11-slim

# Környezeti változók a Python optimalizáláshoz
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Rendszerfüggőségek telepítése (OpenCV és orvosi képfeldolgozás igényei)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Függőségek másolása és telepítése
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    # FastAPI specifikus kiegészítők
    pip install --no-cache-dir uvicorn python-multipart pytest httpx pydantic-settings

# Teljes forráskód másolása (az új app/ struktúra szerint)
COPY . .

# Alapértelmezett FastAPI port
EXPOSE 8000

# Indítás Uvicorn-nal
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]