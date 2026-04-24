"""maskanyone-lite backend.

Async-ish job queue: POST /mask enqueues and returns immediately with
{job_id}. GET /mask/{id} returns status JSON. GET /mask/{id}/result
returns the zip (404 until done).
"""
from __future__ import annotations

import io
import os
import uuid
import zipfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

import db

app = FastAPI()

SHARED_DIR = "/var/lib/maskanyone/shared"

os.makedirs(os.path.join(SHARED_DIR, "in"), exist_ok=True)
os.makedirs(os.path.join(SHARED_DIR, "out"), exist_ok=True)


@app.on_event("startup")
def _startup() -> None:
    db.init_schema()


@app.get("/platform/mode")
def platform_mode() -> dict:
    return {"mode": "local"}


@app.post("/mask", status_code=202)
async def mask(
    video: UploadFile = File(...),
    strategy: str = Form("blur"),
    mode: str = Form("quick"),
    prompt_x: int | None = Form(None),
    prompt_y: int | None = Form(None),
    downsample: float = Form(1.0),
) -> dict:
    if strategy not in {"blur", "solid", "pixelate", "skeleton"}:
        raise HTTPException(status_code=400, detail=f"unknown strategy: {strategy}")
    if mode not in {"quick", "precision"}:
        raise HTTPException(status_code=400, detail=f"unknown mode: {mode}")
    if not 0.1 <= downsample <= 1.0:
        raise HTTPException(status_code=400, detail="downsample must be in [0.1, 1.0]")

    job_id = uuid.uuid4().hex
    in_path = os.path.join(SHARED_DIR, "in", f"{job_id}.mp4")
    out_path = os.path.join(SHARED_DIR, "out", f"{job_id}.mp4")

    with open(in_path, "wb") as f:
        f.write(await video.read())

    db.enqueue(
        job_id=job_id,
        mode=mode,
        strategy=strategy,
        prompt_x=prompt_x,
        prompt_y=prompt_y,
        downsample=downsample,
        original_filename=video.filename,
        input_path=in_path,
        output_path=out_path,
    )
    return {"job_id": job_id, "status": "pending"}


@app.get("/mask/{job_id}")
def job_status(job_id: str) -> dict:
    job = db.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return {
        "job_id": job["id"],
        "status": job["status"],
        "mode": job["mode"],
        "strategy": job["strategy"],
        "original_filename": job["original_filename"],
        "created_at": job["created_at"].isoformat() if job["created_at"] else None,
        "started_at": job["started_at"].isoformat() if job["started_at"] else None,
        "finished_at": job["finished_at"].isoformat() if job["finished_at"] else None,
        "error": job["error"],
    }


@app.get("/mask/{job_id}/result")
def job_result(job_id: str) -> StreamingResponse:
    job = db.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=f"job status: {job['status']}")

    out_path = job["output_path"]
    manifest_path = out_path + ".manifest.json"
    if not os.path.exists(out_path):
        # Either never produced (shouldn't happen at status='done'), or
        # already served once and cleaned up. 410 Gone tells the client
        # to stop retrying.
        raise HTTPException(status_code=410, detail="output already served or gone")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        z.write(out_path, arcname="masked.mp4")
        if os.path.exists(manifest_path):
            z.write(manifest_path, arcname="manifest.json")
    buf.seek(0)

    # Clean up the on-disk artifacts now that they're in the response buffer.
    # DB row stays for audit trail; subsequent GETs return 410.
    for p in (out_path, manifest_path):
        try:
            os.unlink(p)
        except FileNotFoundError:
            pass

    base = job["original_filename"] or "output"
    fname = f"masked_{job['mode']}_{job['strategy']}_{base}.zip"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )
