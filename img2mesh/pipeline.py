"""What every engine shares: options, background removal, and the printable STL step.

An engine turns a prepared RGBA image into a raw mesh (see `img2mesh/engines/`).
Heavy imports happen inside the engines, so the CLI, protocol and geometry code
load (and test) on machines without torch or a GPU.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field, replace

import trimesh
from PIL import Image

from .printable import PrintOptions, make_printable

log = logging.getLogger("img2mesh")

REMBG_MODEL = os.environ.get("REMBG_MODEL", "u2net")
ENGINES = ("hunyuan", "trellis2")


@dataclass(frozen=True)
class GenerateOptions:
    # "auto": remove the background only if the image has no transparency.
    remove_background: str = "auto"  # auto | always | never
    seed: int = 1234
    # None = the engine's own default (Hunyuan 50 steps / guidance 5; TRELLIS.2 12 / 7.5).
    steps: int | None = None
    guidance_scale: float | None = None
    # Detail. Hunyuan: octree resolution 256/384/512. TRELLIS.2: 512, 1024 or 1536 (cascade).
    detail: int | None = None
    max_faces: int = 300_000

    def validate(self) -> None:
        if self.remove_background not in ("auto", "always", "never"):
            raise ValueError("remove_background must be auto, always or never")
        if self.steps is not None and not 1 <= self.steps <= 200:
            raise ValueError("steps must be between 1 and 200")
        if self.guidance_scale is not None and not 0 < self.guidance_scale <= 20:
            raise ValueError("guidance_scale must be between 0 and 20")
        if self.detail is not None and self.detail not in (256, 384, 512, 1024, 1536):
            raise ValueError("detail must be 256, 384, 512, 1024 or 1536")
        if not 10_000 <= self.max_faces <= 5_000_000:
            raise ValueError("max_faces must be between 10,000 and 5,000,000")


@dataclass
class Result:
    mesh: trimesh.Trimesh
    prepared_image: Image.Image
    seconds: dict = field(default_factory=dict)


# After errors like these the process's CUDA context is broken: every later job
# fails too ("device not ready", then allocator asserts). Only a new process helps.
GPU_FATAL_MARKERS = ("CUDA driver error", "CUDA error", "device not ready", "CUDACachingAllocator", "cudaError", "illegal memory access")


def is_gpu_fatal(error: BaseException) -> bool:
    text = f"{type(error).__name__}: {error}"
    return any(marker in text for marker in GPU_FATAL_MARKERS)


def has_transparency(image: Image.Image) -> bool:
    if image.mode != "RGBA":
        return False
    return image.getchannel("A").getextrema()[0] < 255


class Engine:
    """Base class. Subclasses set `name`, `raw_up`, `raw_front` and implement `make_mesh`."""

    name = "engine"
    raw_up = "+y"
    raw_front = "+z"
    detail_choices: tuple[int, ...] = ()

    def __init__(self) -> None:
        from rembg import new_session

        self.rembg_session = new_session(REMBG_MODEL)

    def prepare(self, image: Image.Image, mode: str) -> Image.Image:
        from rembg import remove

        image = image.convert("RGBA")
        if mode == "always" or (mode == "auto" and not has_transparency(image)):
            image = remove(image.convert("RGB"), session=self.rembg_session, bgcolor=[255, 255, 255, 0])
        return image

    def make_mesh(self, image: Image.Image, options: GenerateOptions, on_progress=None) -> tuple[trimesh.Trimesh, dict]:
        raise NotImplementedError

    def generate(self, image: Image.Image, options: GenerateOptions, on_progress=None) -> Result:
        options.validate()
        if options.detail is not None and options.detail not in self.detail_choices:
            raise ValueError(f"{self.name} detail must be one of {', '.join(map(str, self.detail_choices))}")
        start = time.monotonic()
        prepared = self.prepare(image, options.remove_background)
        seconds = {"background": time.monotonic() - start}
        mesh, more = self.make_mesh(prepared, options, on_progress)
        seconds.update(more)
        return Result(mesh=mesh, prepared_image=prepared, seconds=seconds)

    def to_stl(self, image: Image.Image, generate: GenerateOptions, printable: PrintOptions, on_progress=None) -> tuple[bytes, dict]:
        """The whole job: image in, printable binary STL out, plus stats."""
        printable = replace(printable, up=printable.up or self.raw_up, front=printable.front or self.raw_front)
        printable.validate()
        result = self.generate(image, generate, on_progress)
        start = time.monotonic()
        mesh = make_printable(result.mesh, printable)
        result.seconds["printable"] = time.monotonic() - start
        stl = mesh.export(file_type="stl")
        stats = {
            "engine": self.name,
            "seconds": {k: round(v, 2) for k, v in result.seconds.items()},
            "faces": int(len(mesh.faces)),
            "watertight": bool(mesh.is_watertight),
            "size_mm": [round(float(v), 2) for v in mesh.extents],
        }
        return stl, stats


def engine_name() -> str:
    name = os.environ.get("ENGINE", "hunyuan").strip().lower()
    if name not in ENGINES:
        raise ValueError(f"ENGINE must be one of {', '.join(ENGINES)}, not {name!r}")
    return name


def load_engine() -> Engine:
    """The engine named by ENGINE (default hunyuan). Loads its model, which takes a while."""
    name = engine_name()
    log.info("engine: %s", name)
    if name == "trellis2":
        from .engines.trellis2 import Trellis2Engine

        return Trellis2Engine()
    from .engines.hunyuan import HunyuanEngine

    return HunyuanEngine()
