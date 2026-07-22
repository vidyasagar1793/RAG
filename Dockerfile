FROM python:3.12-slim

# Install system dependencies needed to compile PostgreSQL drivers
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Poetry
RUN pip install --no-cache-dir poetry

# Tell Poetry not to create a virtual environment (Docker is already isolated)
RUN poetry config virtualenvs.create false

# Copy dependency files FIRST to leverage Docker layer caching
COPY pyproject.toml poetry.lock* ./

# Install the dependencies
RUN poetry install --only main --no-interaction --no-root

# Copy the rest of your application code
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# The command to start the server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]