"""Image -> raw mesh with Hunyuan3D 2.1 (shape only). Needs an NVIDIA GPU.

Heavy imports happen inside `Img2Mesh` so the CLI, protocol and geometry code
load (and test) on machines without torch or a GPU.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import trimesh
from PIL import Image

from .printable import PrintOptions, make_printable

MODEL = os.environ.get("HUNYUAN_MODEL", "tencent/Hunyuan3D-2.1")
REMBG_MODEL = os.environ.get("REMBG_MODEL", "u2net")


@dataclass(frozen=True)
class GenerateOptions:
    # "auto": remove the background only if the image has no transparency.
    remove_background: str = "auto"  # auto | always | never
    seed: int = 1234
    steps: int = 50
    guidance_scale: float = 5.0
    octree_resolution: int = 384
    max_faces: int = 300_000

    def validate(self) -> None:
        if self.remove_background not in ("auto", "always", "never"):
            raise ValueError("remove_background must be auto, always or never")
        if not 1 <= self.steps <= 200:
            raise ValueError("steps must be between 1 and 200")
        if self.octree_resolution not in (256, 384, 512):
            raise ValueError("octree_resolution must be 256, 384 or 512")


@dataclass
class Result:
    mesh: trimesh.Trimesh
    prepared_image: Image.Image
    seconds: dict = field(default_factory=dict)


def has_transparency(image: Image.Image) -> bool:
    if image.mode != "RGBA":
        return False
    return image.getchannel("A").getextrema()[0] < 255


class Img2Mesh:
    def __init__(self) -> None:
        import torch  # noqa: F401 - fail early with a clear error if torch is missing
        from rembg import new_session
        from hy3dshape import DegenerateFaceRemover, FaceReducer, FloaterRemover
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

        self.shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(MODEL)
        self.rembg_session = new_session(REMBG_MODEL)
        self.remove_floaters = FloaterRemover()
        self.remove_degenerate = DegenerateFaceRemover()
        self.reduce_faces = FaceReducer()

    def prepare(self, image: Image.Image, mode: str) -> Image.Image:
        from rembg import remove

        image = image.convert("RGBA")
        if mode == "always" or (mode == "auto" and not has_transparency(image)):
            image = remove(image.convert("RGB"), session=self.rembg_session, bgcolor=[255, 255, 255, 0])
        return image

    def generate(self, image: Image.Image, options: GenerateOptions, on_progress=None) -> Result:
        import torch

        options.validate()
        seconds = {}
        start = time.monotonic()
        prepared = self.prepare(image, options.remove_background)
        seconds["background"] = time.monotonic() - start

        start = time.monotonic()
        mesh = self.shape(
            image=prepared,
            num_inference_steps=options.steps,
            guidance_scale=options.guidance_scale,
            generator=torch.Generator().manual_seed(options.seed),
            octree_resolution=options.octree_resolution,
            callback=(lambda step, *_: on_progress(step, options.steps)) if on_progress else None,
            callback_steps=1 if on_progress else None,
        )[0]
        seconds["shape"] = time.monotonic() - start

        start = time.monotonic()
        mesh = self.remove_floaters(mesh)
        mesh = self.remove_degenerate(mesh)
        mesh = self.reduce_faces(mesh, max_facenum=options.max_faces)
        seconds["cleanup"] = time.monotonic() - start
        return Result(mesh=mesh, prepared_image=prepared, seconds=seconds)

    def to_stl(self, image: Image.Image, generate: GenerateOptions, printable: PrintOptions, on_progress=None) -> tuple[bytes, dict]:
        """The whole job: image in, printable binary STL out, plus stats."""
        result = self.generate(image, generate, on_progress)
        start = time.monotonic()
        mesh = make_printable(result.mesh, printable)
        result.seconds["printable"] = time.monotonic() - start
        stl = mesh.export(file_type="stl")
        stats = {
            "seconds": {k: round(v, 2) for k, v in result.seconds.items()},
            "faces": int(len(mesh.faces)),
            "watertight": bool(mesh.is_watertight),
            "size_mm": [round(float(v), 2) for v in mesh.extents],
        }
        return stl, stats
