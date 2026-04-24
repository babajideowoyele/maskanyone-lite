"""maskanyone-lite backend.

Synchronous upload-and-mask endpoint. Accepts a video, writes it to a shared
volume, asks the worker to mask it, streams the result back.

A proper job queue will replace this — same /mask contract, different internals.
"""
from __future__ import annotations

import io
import os
import uuid
import zipfile

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

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
    mode: str = Form("quick"),
    prompt_x: int | None = Form(None),
    prompt_y: int | None = Form(None),
) -> FileResponse:
    if strategy not in {"blur", "solid", "pixelate"}:
        raise HTTPException(status_code=400, detail=f"unknown strategy: {strategy}")
    if mode not in {"quick", "precision"}:
        raise HTTPException(status_code=400, detail=f"unknown mode: {mode}")

    job_id = uuid.uuid4().hex
    in_path = os.path.join(SHARED_DIR, "in", f"{job_id}.mp4")
    out_path = os.path.join(SHARED_DIR, "out", f"{job_id}.mp4")

    with open(in_path, "wb") as f:
        f.write(await video.read())

    try:
        r = requests.post(
            f"{WORKER_URL}/mask",
            json={
                "input_path": in_path,
                "output_path": out_path,
                "strategy": strategy,
                "mode": mode,
                "prompt_x": prompt_x,
                "prompt_y": prompt_y,
            },
            timeout=1800,
        )
        r.raise_for_status()
    except requests.RequestException as e:
        os.unlink(in_path)
        raise HTTPException(status_code=502, detail=f"worker error: {e}") from e

    os.unlink(in_path)

    manifest_path = out_path + ".manifest.json"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        z.write(out_path, arcname="masked.mp4")
        if os.path.exists(manifest_path):
            z.write(manifest_path, arcname="manifest.json")
    buf.seek(0)
    fname = f"masked_{mode}_{strategy}_{video.filename or 'output'}.zip"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )
