"""Postgres connection + schema + job-queue ops.

Single table, single worker assumed. FOR UPDATE SKIP LOCKED means this
scales to N workers without changing the code.
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import RealDictCursor

DSN = (
    f"host={os.environ['BACKEND_PG_HOST']} "
    f"port={os.environ['BACKEND_PG_PORT']} "
    f"user={os.environ['BACKEND_PG_USER']} "
    f"password={os.environ['BACKEND_PG_PASSWORD']} "
    f"dbname={os.environ['BACKEND_PG_DATABASE']}"
)


SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'done', 'failed')),
    mode TEXT NOT NULL,
    strategy TEXT NOT NULL,
    prompt_x INTEGER,
    prompt_y INTEGER,
    downsample REAL NOT NULL DEFAULT 1.0,
    original_filename TEXT,
    input_path TEXT NOT NULL,
    output_path TEXT NOT NULL,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS jobs_status_created_idx ON jobs (status, created_at);
"""


def connect(retries: int = 30, delay_s: float = 1.0) -> psycopg2.extensions.connection:
    last: Exception | None = None
    for _ in range(retries):
        try:
            conn = psycopg2.connect(DSN)
            conn.autocommit = False
            return conn
        except psycopg2.OperationalError as e:
            last = e
            time.sleep(delay_s)
    raise RuntimeError(f"postgres unreachable after {retries} tries: {last}")


def init_schema() -> None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(SCHEMA)
        conn.commit()


@contextmanager
def cursor(dict_rows: bool = True):
    conn = connect()
    try:
        yield conn.cursor(cursor_factory=RealDictCursor) if dict_rows else conn.cursor()
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def enqueue(
    *,
    job_id: str,
    mode: str,
    strategy: str,
    prompt_x: int | None,
    prompt_y: int | None,
    downsample: float,
    original_filename: str | None,
    input_path: str,
    output_path: str,
) -> None:
    with cursor() as cur:
        cur.execute(
            """INSERT INTO jobs (id, status, mode, strategy, prompt_x, prompt_y,
                downsample, original_filename, input_path, output_path)
               VALUES (%s, 'pending', %s, %s, %s, %s, %s, %s, %s, %s)""",
            (job_id, mode, strategy, prompt_x, prompt_y, downsample,
             original_filename, input_path, output_path),
        )


def get(job_id: str) -> dict | None:
    with cursor() as cur:
        cur.execute("SELECT * FROM jobs WHERE id = %s", (job_id,))
        return cur.fetchone()


def claim_next() -> dict | None:
    """Atomically grab one pending job, mark it running. Returns None if queue empty."""
    with cursor() as cur:
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


def mark_done(job_id: str) -> None:
    with cursor() as cur:
        cur.execute(
            "UPDATE jobs SET status='done', finished_at=now() WHERE id=%s",
            (job_id,),
        )


def mark_failed(job_id: str, error: str) -> None:
    with cursor() as cur:
        cur.execute(
            "UPDATE jobs SET status='failed', finished_at=now(), error=%s WHERE id=%s",
            (error, job_id),
        )
