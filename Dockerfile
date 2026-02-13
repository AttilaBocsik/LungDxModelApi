FROM python:3.11-slim

# Környezeti változók
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Rendszerfüggőségek (OpenCV-hez és alapvető fordításhoz szükségesek)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Függőségek telepítése külön rétegként a cache-elés miatt
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Forráskód másolása
COPY . .

# Biztonság: létrehozunk egy nem-root felhasználót
RUN useradd -m medicaluser && chown -R medicaluser:medicaluser /app
USER medicaluser

EXPOSE 8000

# Healthcheck hozzáadása (fontos orvosi rendszereknél)
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost:8000/ || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]