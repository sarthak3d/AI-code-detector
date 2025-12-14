FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    TRANSFORMERS_CACHE=/opt/hf-cache \
    HUGGINGFACE_HUB_CACHE=/opt/hf-cache

WORKDIR /build

RUN python -m venv /opt/venv
RUN mkdir -p /opt/hf-cache
ENV PATH=/opt/venv/bin:${PATH}

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY api /app/api
COPY model /app/model
COPY scripts/precache_models.py /app/scripts/precache_models.py

ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}
ARG PRECACHE_MODELS=1

RUN if [ "${PRECACHE_MODELS}" = "1" ]; then \
        HF_LOCAL_FILES_ONLY=0 TRANSFORMERS_OFFLINE=0 python /app/scripts/precache_models.py \
            --model starcoder2-3b \
            --model deepseek-1.3b \
            --include-t5; \
    else \
        echo "Skipping model pre-cache at build time."; \
    fi

FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=model \
    HF_HOME=/opt/hf-cache \
    TRANSFORMERS_CACHE=/opt/hf-cache \
    HUGGINGFACE_HUB_CACHE=/opt/hf-cache \
    HF_LOCAL_FILES_ONLY=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_USE_V1=0 \
    VLLM_ENFORCE_EAGER=1 \
    PATH=/opt/venv/bin:${PATH}

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/hf-cache /opt/hf-cache
COPY --from=builder /app/api /app/api
COPY --from=builder /app/model /app/model

RUN find /app -type d -name "__pycache__" -prune -exec rm -rf {} + \
    && find /opt/venv -type d -name "__pycache__" -prune -exec rm -rf {} + \
    && find /opt/venv -type f -name "*.pyc" -delete

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
