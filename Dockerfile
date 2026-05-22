FROM python:3.12-slim

WORKDIR /app

# Install system deps required for some Python packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy pyproject.toml and uv.lock
COPY pyproject.toml uv.lock ./

# Install dependencies using uv sync
RUN uv sync --frozen --no-dev
# The --frozen flag ensures exact reproducibility across builds, and --no-dev skips development dependencies so the image is lean.

# Copy application code and data
COPY app/ ./app/
COPY data/ ./data/
COPY main.py ./

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

CMD ["uv", "run", "litestar", "--app", "main:app", "run", "--host", "0.0.0.0", "--port", "8000"]
