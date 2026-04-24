"""Polling worker.

Claims one pending job at a time via Postgres row-locking, masks the video,
marks done/failed. No HTTP — the backend writes jobs, we read them.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import RealDictCursor

from masker import mask_video


DSN = (
    f"host={os.environ['BACKEND_PG_HOST']} "
    f"port={os.environ['BACKEND_PG_PORT']} "
    f"user={os.environ['BACKEND_PG_USER']} "
    f"password={os.environ['BACKEND_PG_PASSWORD']} "
    f"dbname={os.environ['BACKEND_PG_DATABASE']}"
)

POLL_INTERVAL_S = 2.0


def _connect():
    for _ in range(60):
        try:
            conn = psycopg2.connect(DSN)
            conn.autocommit = False
            return conn
        except psycopg2.OperationalError as e:
            print(f"[worker] waiting for postgres: {e}", flush=True)
            time.sleep(1.0)
    raise RuntimeError("postgres unreachable")


@contextmanager
def _cursor(conn):
    try:
        yield conn.cursor(cursor_factory=RealDictCursor)
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def _claim_next(conn) -> dict | None:
    with _cursor(conn) as cur:
        cur.execute(
            """UPDATE jobs SET status='running', started_at=now()
               WHERE id = (
                 SELECT id FROM jobs
                 WHERE status='pending'
                 ORDER BY created_at
                 FOR UPDATE SKIP LOCKED
                 LIMIT 1
               )
               RETURNING *"""
        )
        return cur.fetchone()


def _mark_done(conn, job_id: str) -> None:
    with _cursor(conn) as cur:
        cur.execute(
            "UPDATE jobs SET status='done', finished_at=now() WHERE id=%s",
            (job_id,),
        )


def _mark_failed(conn, job_id: str, error: str) -> None:
    with _cursor(conn) as cur:
        cur.execute(
            "UPDATE jobs SET status='failed', finished_at=now(), error=%s WHERE id=%s",
            (error, job_id),
        )


def _run_job(job: dict) -> None:
    prompt = (
        (job["prompt_x"], job["prompt_y"])
        if job["prompt_x"] is not None and job["prompt_y"] is not None
        else None
    )
    mask_video(
        job["input_path"],
        job["output_path"],
        strategy=job["strategy"],
        mode=job["mode"],
        prompt_xy=prompt,
        downsample=float(job["downsample"]),
        original_filename=job["original_filename"],
    )


def _cleanup_input(job: dict) -> None:
    """Remove the uploaded file after processing. Output stays until
    served and GC'd by the backend (future work)."""
    try:
        os.unlink(job["input_path"])
    except FileNotFoundError:
        pass


def main() -> None:
    print("[worker] starting", flush=True)
    conn = _connect()
    print("[worker] postgres ready, polling", flush=True)

    while True:
        job = _claim_next(conn)
        if job is None:
            time.sleep(POLL_INTERVAL_S)
            continue

        job_id = job["id"]
        print(f"[worker] claimed job {job_id} mode={job['mode']} strategy={job['strategy']}", flush=True)
        t0 = time.perf_counter()
        try:
            _run_job(job)
            _mark_done(conn, job_id)
            print(f"[worker] done {job_id} in {time.perf_counter() - t0:.1f}s", flush=True)
        except Exception:
            err = traceback.format_exc()
            print(f"[worker] job {job_id} failed:\n{err}", flush=True)
            _mark_failed(conn, job_id, err[:4000])
        finally:
            _cleanup_input(job)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[worker] shutting down", flush=True)
        sys.exit(0)
