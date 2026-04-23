FROM python:3.11-slim

ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libpq-dev ffmpeg wget gcc build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    fastapi "uvicorn[standard]" psycopg2 requests python-jose[cryptography] \
    opencv-python-headless numpy==1.26.4 psutil pandas

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
