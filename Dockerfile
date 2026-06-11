FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System dependencies for chromadb / numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml .
COPY src/ ./src/
RUN pip install -e .

COPY data/ ./data/
COPY scripts/ ./scripts/
COPY evals/ ./evals/

# Non-root user for runtime
RUN useradd --create-home --shell /bin/bash anchor \
    && chown -R anchor:anchor /app
USER anchor

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "anchor.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
