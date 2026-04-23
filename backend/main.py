"""maskanyone-lite backend.

Synchronous upload-and-mask endpoint. Accepts a video, writes it to a shared
volume, asks the worker to mask it, streams the result back.

A proper job queue will replace this — same /mask contract, different internals.
"""
from __future__ import annotations

import os
import uuid

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

app = FastAPI()

SHARED_DIR = "/var/lib/maskanyone/shared"
WORKER_URL = os.environ.get("WORKER_URL", "http://worker:8000")

os.makedirs(os.path.join(SHARED_DIR, "in"), exist_ok=True)
os.makedirs(os.path.join(SHARED_DIR, "out"), exist_ok=True)


@app.get("/platform/mode")
def platform_mode() -> dict:
    return {"mode": "local"}


@app.post("/mask")
async def mask(
    video: UploadFile = File(...),
    strategy: str = Form("blur"),
) -> FileResponse:
    if strategy not in {"blur", "solid", "pixelate"}:
        raise HTTPException(status_code=400, detail=f"unknown strategy: {strategy}")

    job_id = uuid.uuid4().hex
    in_path = os.path.join(SHARED_DIR, "in", f"{job_id}.mp4")
    out_path = os.path.join(SHARED_DIR, "out", f"{job_id}.mp4")

    with open(in_path, "wb") as f:
        f.write(await video.read())

    try:
        r = requests.post(
            f"{WORKER_URL}/mask",
            json={"input_path": in_path, "output_path": out_path, "strategy": strategy},
            timeout=600,
        )
        r.raise_for_status()
    except requests.RequestException as e:
        os.unlink(in_path)
        raise HTTPException(status_code=502, detail=f"worker error: {e}") from e

    os.unlink(in_path)
    return FileResponse(
        out_path,
        media_type="video/mp4",
        filename=f"masked_{strategy}_{video.filename or 'output.mp4'}",
        background=None,
    )
