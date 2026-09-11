FROM python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    FLUX_MODEL_DIR=/app/models FLUX_DATA_DIR=/app/data
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock \
    && groupadd -g 10001 glyph && useradd -r -u 10001 -g glyph glyph
COPY --chown=glyph:glyph src ./src
COPY --chown=glyph:glyph web ./web
COPY --chown=glyph:glyph assets ./assets
COPY --chown=glyph:glyph models ./models
RUN mkdir -p /app/data && chown glyph:glyph /app/data
USER glyph
EXPOSE 9000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:9000/api/health',timeout=4)"
CMD ["uvicorn", "flux_glyph.api:app", "--host", "0.0.0.0", "--port", "9000", "--workers", "1", "--limit-concurrency", "24", "--timeout-keep-alive", "5"]
