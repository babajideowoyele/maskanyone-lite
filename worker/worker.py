import os
import time

BACKEND = os.environ.get("WORKER_BACKEND_BASE_PATH", "http://python:8000/_worker/")

print(f"worker online (backend={BACKEND}) — no jobs yet", flush=True)

while True:
    time.sleep(60)
