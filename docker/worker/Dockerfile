FROM python:3.11-slim

ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libpq-dev ffmpeg wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    mediapipe==0.10.14 opencv-python-headless requests numpy==1.26.4 \
    fastapi "uvicorn[standard]" pydantic

RUN mkdir /worker_models \
    && wget -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task \
        -O /worker_models/pose_landmarker_heavy.task \
    && wget -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task \
        -O /worker_models/face_landmarker.task \
    && wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task \
        -O /worker_models/hand_landmarker.task
