FROM python:3.11-slim

ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libpq-dev ffmpeg wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# CPU torch + HF Transformers for EdgeTAM-image precision mode.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.4.1 torchvision==0.19.1 \
  && pip install --no-cache-dir \
    mediapipe==0.10.14 opencv-python-headless requests numpy==1.26.4 \
    fastapi "uvicorn[standard]" pydantic \
    transformers==4.57.6 timm==1.0.26 pillow

RUN mkdir /worker_models \
    && wget -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task \
        -O /worker_models/pose_landmarker_heavy.task \
    && wget -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task \
        -O /worker_models/face_landmarker.task \
    && wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task \
        -O /worker_models/hand_landmarker.task

# Pre-cache EdgeTAM weights so first /mask?mode=precision request doesn't
# pay the download latency.
RUN python -c "from transformers import EdgeTamModel, Sam2Processor; \
    Sam2Processor.from_pretrained('yonigozlan/EdgeTAM-hf'); \
    EdgeTamModel.from_pretrained('yonigozlan/EdgeTAM-hf')"
