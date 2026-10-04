"""Client for pull protocol v1 (see PROTOCOL.md). Standard library only."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

PROTOCOL = 1


class ProtocolError(Exception):
    pass


@dataclass(frozen=True)
class Job:
    id: str
    image_url: str
    upload_url: str
    options: dict


class Client:
    def __init__(self, site_url: str, secret: str, worker: str, timeout: float = 30) -> None:
        if not site_url or not secret or not worker:
            raise ValueError("SITE_URL, WORKER_SECRET and WORKER_NAME are all required")
        self.base = site_url.rstrip("/") + "/api/worker"
        self.secret = secret
        self.worker = worker
        self.timeout = timeout

    def _post(self, path: str, body: dict) -> tuple[int, dict | None]:
        data = json.dumps({"protocol": PROTOCOL, "worker": self.worker, **body}).encode()
        request = urllib.request.Request(
            self.base + path,
            data=data,
            method="POST",
            headers={"Authorization": f"Bearer {self.secret}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
                return response.status, (json.loads(raw) if raw else None)
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:300]
            raise ProtocolError(f"{path} -> HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise ProtocolError(f"{path} -> {error.reason}") from error

    def claim(self) -> Job | None:
        status, body = self._post("/claim", {})
        if status == 204 or not body or not body.get("job"):
            return None
        job = body["job"]
        try:
            return Job(id=str(job["id"]), image_url=job["image_url"], upload_url=job["upload_url"], options=job.get("options") or {})
        except KeyError as error:
            raise ProtocolError(f"claim response is missing {error}") from error

    def heartbeat(self, job_id: str, progress: float | None = None) -> None:
        self._post(f"/jobs/{job_id}", {"event": "heartbeat", "progress": progress})

    def succeeded(self, job_id: str, stats: dict) -> None:
        self._post(f"/jobs/{job_id}", {"event": "succeeded", "stats": stats})

    def failed(self, job_id: str, error: str) -> None:
        self._post(f"/jobs/{job_id}", {"event": "failed", "error": error[:500]})

    def download(self, url: str, max_bytes: int = 20 * 1024 * 1024) -> bytes:
        with urllib.request.urlopen(url, timeout=self.timeout) as response:
            data = response.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ProtocolError("image is larger than 20 MB")
        return data

    def upload(self, url: str, stl: bytes) -> None:
        request = urllib.request.Request(url, data=stl, method="PUT", headers={"Content-Type": "model/stl"})
        try:
            with urllib.request.urlopen(request, timeout=max(self.timeout, 120)):
                pass
        except urllib.error.HTTPError as error:
            raise ProtocolError(f"upload -> HTTP {error.code}") from error
