FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e ".[dev]" --no-cache-dir

EXPOSE 8000

CMD ["uvicorn", "src.neuroforge.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
