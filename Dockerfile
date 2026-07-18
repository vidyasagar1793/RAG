FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

# copy dependency manifests first for layer caching
COPY pyproject.toml poetry.toml /app/

RUN python -m pip install --upgrade pip \
    && pip install poetry \
    && poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction

# copy app sources
COPY . /app

EXPOSE 8000

# Adjust module path if your ASGI app is in a different module
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
