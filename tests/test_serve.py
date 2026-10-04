"""The serve loop against a fake site speaking protocol v1."""

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from PIL import Image

from img2mesh import serve as serve_module
from img2mesh.protocol import Client, ProtocolError
from img2mesh.serve import options_from, serve

SECRET = "s3cret"


class FakeSite:
    def __init__(self, options):
        self.events = []
        self.uploads = []
        self.jobs = [{"id": "gen-1", "options": options}]
        png = io.BytesIO()
        Image.new("RGBA", (8, 8), (255, 0, 0, 255)).save(png, "PNG")
        self.png = png.getvalue()
        site = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def _json(self, status, body=None):
                self.send_response(status)
                self.end_headers()
                if body is not None:
                    self.wfile.write(json.dumps(body).encode())

            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(site.png)

            def do_PUT(self):
                site.uploads.append(self.rfile.read(int(self.headers["Content-Length"])))
                self._json(200)

            def do_POST(self):
                if self.headers.get("Authorization") != f"Bearer {SECRET}":
                    return self._json(401, {"error": "unauthorized"})
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                assert body["protocol"] == 1 and body["worker"] == "pc-1"
                if self.path == "/api/worker/claim":
                    if not site.jobs:
                        return self._json(204)
                    job = site.jobs.pop(0)
                    base = f"http://127.0.0.1:{self.server.server_port}"
                    return self._json(200, {"job": {**job, "image_url": f"{base}/image.png", "upload_url": f"{base}/upload"}})
                site.events.append((self.path, body))
                self._json(200, {})

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


class FakeEngine:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def to_stl(self, image, generate, printable, on_progress=None):
        self.calls.append((image.size, generate, printable))
        on_progress(25, 50)
        if self.fail:
            raise RuntimeError("CUDA out of memory")
        return b"solid", {"faces": 1}


@pytest.fixture(autouse=True)
def fast_heartbeat(monkeypatch):
    monkeypatch.setattr(serve_module, "HEARTBEAT_SECONDS", 0.01)


def test_job_runs_uploads_and_reports_success():
    site = FakeSite({"height_mm": 150, "seed": 7})
    engine = FakeEngine()
    serve(lambda: engine, Client(site.url, SECRET, "pc-1"), once=True)
    assert site.uploads == [b"solid"]
    (size, generate, printable), = engine.calls
    assert size == (8, 8) and generate.seed == 7 and printable.height_mm == 150
    assert site.events[-1] == ("/api/worker/jobs/gen-1", {"protocol": 1, "worker": "pc-1", "event": "succeeded", "stats": {"faces": 1}})


def test_engine_error_is_reported_as_failed():
    site = FakeSite({"height_mm": 150})
    serve(lambda: FakeEngine(fail=True), Client(site.url, SECRET, "pc-1"), once=True)
    assert site.uploads == []
    path, body = site.events[-1]
    assert body["event"] == "failed" and "CUDA out of memory" in body["error"]


def test_missing_height_fails_the_job_not_the_worker():
    site = FakeSite({})
    serve(lambda: FakeEngine(), Client(site.url, SECRET, "pc-1"), once=True)
    assert site.events[-1][1]["event"] == "failed"


def test_no_job_returns_none():
    site = FakeSite({"height_mm": 1})
    site.jobs.clear()
    assert Client(site.url, SECRET, "pc-1").claim() is None


def test_wrong_secret_is_a_protocol_error():
    site = FakeSite({"height_mm": 1})
    with pytest.raises(ProtocolError, match="401"):
        Client(site.url, "wrong", "pc-1").claim()


def test_client_requires_configuration():
    with pytest.raises(ValueError):
        Client("", SECRET, "pc-1")


def test_options_defaults():
    from img2mesh.protocol import Job

    generate, printable = options_from(Job("1", "", "", {"height_mm": 120}))
    assert generate.remove_background == "auto" and generate.steps == 50
    assert printable.flat_back == 0.04 and printable.front == "+z"
