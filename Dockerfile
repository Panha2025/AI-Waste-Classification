FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_NUM_INTRAOP_THREADS=1 \
    TF_NUM_INTEROP_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MODEL_PATH=/app/artifacts/models/binary_mobilenetv2_finetuned.keras

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 bot

COPY requirements-deploy.txt ./requirements-deploy.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements-deploy.txt

COPY waste_ai/__init__.py waste_ai/config.py waste_ai/metrics.py waste_ai/inference.py waste_ai/bot.py ./waste_ai/
COPY artifacts/model_manifest.json ./artifacts/model_manifest.json
COPY artifacts/models/binary_mobilenetv2_finetuned.keras ./artifacts/models/binary_mobilenetv2_finetuned.keras

USER 10001:10001
CMD ["python", "-m", "waste_ai.bot"]
