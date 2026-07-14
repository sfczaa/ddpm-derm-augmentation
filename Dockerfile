FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    DDPM_DERM_MODEL_PATH=/models/best.pt \
    DDPM_DERM_MODEL_MANIFEST=/app/deploy/model_manifest.json \
    DDPM_DERM_CLASS_MAP_PATH=/app/deploy/class_to_idx.json \
    DDPM_DERM_GALLERY_DIR=/gallery/epoch0100_seed0

WORKDIR /app

COPY requirements-deploy.txt ./
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision && \
    python -m pip install --no-cache-dir -r requirements-deploy.txt

COPY src ./src
COPY app ./app
COPY deploy ./deploy
# config.py requires only this sentinel at import; the HAM10000 dataset is not shipped.
COPY deploy/class_to_idx.json ./data/manifests/class_to_idx.json

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--no-access-log"]
