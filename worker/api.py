"""Worker HTTP API — thin wrapper around masker.mask_video.

POST /mask with JSON body: {input_path, output_path, strategy}
The paths must be accessible to the worker container (use a shared volume).
"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from masker import mask_video

app = FastAPI()


class MaskRequest(BaseModel):
    input_path: str
    output_path: str
    strategy: str = "blur"


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/mask")
def mask(req: MaskRequest) -> dict:
    if not os.path.exists(req.input_path):
        raise HTTPException(status_code=404, detail=f"input not found: {req.input_path}")
    n = mask_video(req.input_path, req.output_path, req.strategy)
    return {"frames": n, "output_path": req.output_path, "strategy": req.strategy}
