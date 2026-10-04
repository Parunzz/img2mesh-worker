"""Worker loop: claim a job, run it, report. Only outbound HTTPS calls."""

from __future__ import annotations

import io
import logging
import os
import threading
import time

from PIL import Image

from .pipeline import GenerateOptions
from .printable import PrintOptions
from .protocol import Client, Job, ProtocolError

log = logging.getLogger("img2mesh")

HEARTBEAT_SECONDS = 15
IDLE_POLL_SECONDS = float(os.environ.get("POLL_SECONDS", "5"))
MAX_BACKOFF_SECONDS = 300


def options_from(job: Job) -> tuple[GenerateOptions, PrintOptions]:
    o = job.options
    if "height_mm" not in o:
        raise ValueError("job options must include height_mm")
    generate = GenerateOptions(
        remove_background=o.get("remove_background", "auto"),
        seed=int(o.get("seed", 1234)),
        steps=int(o.get("steps", 50)),
        octree_resolution=int(o.get("octree_resolution", 384)),
    )
    printable = PrintOptions(
        height_mm=float(o["height_mm"]),
        flat_back=float(o.get("flat_back", 0.04)),
        flat_bottom=float(o.get("flat_bottom", 0.02)),
        up=o.get("up", "+y"),
        front=o.get("front", "+z"),
    )
    generate.validate()
    printable.validate()
    return generate, printable


class Heartbeat:
    """Reports progress every HEARTBEAT_SECONDS while a job runs."""

    def __init__(self, client: Client, job_id: str) -> None:
        self.client, self.job_id = client, job_id
        self.progress: float | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(HEARTBEAT_SECONDS):
            try:
                self.client.heartbeat(self.job_id, self.progress)
            except ProtocolError as error:
                log.warning("heartbeat failed: %s", error)

    def __enter__(self) -> "Heartbeat":
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def run_job(client: Client, engine, job: Job) -> None:
    log.info("job %s claimed", job.id)
    try:
        generate, printable = options_from(job)
        image = Image.open(io.BytesIO(client.download(job.image_url)))
        with Heartbeat(client, job.id) as beat:
            def progress(step: int, total: int) -> None:
                beat.progress = round(step / total, 2)

            stl, stats = engine.to_stl(image, generate, printable, on_progress=progress)
        client.upload(job.upload_url, stl)
        client.succeeded(job.id, stats)
        log.info("job %s done: %s", job.id, stats)
    except Exception as error:  # noqa: BLE001 - every failure is reported, then the loop continues
        log.exception("job %s failed", job.id)
        try:
            client.failed(job.id, f"{type(error).__name__}: {error}")
        except ProtocolError as report_error:
            log.warning("could not report failure: %s", report_error)


def serve(engine_factory, client: Client, once: bool = False) -> None:
    """Poll forever. The model loads before the first claim, so a claimed job
    never waits (without heartbeats) for weights to download."""
    log.info("loading model (the first start downloads several GB)")
    engine = engine_factory()
    backoff = IDLE_POLL_SECONDS
    log.info("worker %s polling %s", client.worker, client.base)
    while True:
        try:
            job = client.claim()
            backoff = IDLE_POLL_SECONDS
        except ProtocolError as error:
            log.warning("claim failed: %s (retry in %.0fs)", error, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)
            continue
        if job is None:
            if once:
                return
            time.sleep(IDLE_POLL_SECONDS)
            continue
        run_job(client, engine, job)
        if once:
            return
