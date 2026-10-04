"""Image -> raw mesh with Hunyuan3D 2.1 (shape only). Needs an NVIDIA GPU.

Heavy imports happen inside `Img2Mesh` so the CLI, protocol and geometry code
load (and test) on machines without torch or a GPU.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import trimesh
from PIL import Image

from .printable import PrintOptions, make_printable

log = logging.getLogger("img2mesh")

MODEL = os.environ.get("HUNYUAN_MODEL", "tencent/Hunyuan3D-2.1")
REMBG_MODEL = os.environ.get("REMBG_MODEL", "u2net")
# Hunyuan's own config for the 2.1 shape model, for single-file checkpoints
# (such as ComfyUI's hunyuan_3d_v2.1.safetensors) that don't carry one.
SINGLE_FILE_CONFIG = Path(__file__).with_name("hunyuan3d-dit-v2-1.yaml")


def single_file_checkpoint() -> Path | None:
    """A single-file checkpoint mounted at CHECKPOINT_DIR, if there is one."""
    folder = Path(os.environ.get("CHECKPOINT_DIR", "/checkpoints"))
    path = folder / os.environ.get("HUNYUAN_CHECKPOINT", "hunyuan_3d_v2.1.safetensors")
    return path if path.is_file() else None


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

        checkpoint = single_file_checkpoint()
        if checkpoint:
            log.info("loading shape model from %s", checkpoint)
            self.shape = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
                str(checkpoint), str(SINGLE_FILE_CONFIG), use_safetensors=checkpoint.suffix == ".safetensors"
            )
        else:
            folder = Path(os.environ.get("CHECKPOINT_DIR", "/checkpoints"))
            seen = sorted(p.name for p in folder.iterdir())[:10] if folder.is_dir() else []
            log.info(
                "no single-file checkpoint %s in %s (found: %s); using the Tencent layout "
                "(HUNYUAN_DIR), which downloads if missing. To use ComfyUI's file, set "
                "COMFYUI_CHECKPOINTS in a file named exactly .env next to docker-compose.yml.",
                os.environ.get("HUNYUAN_CHECKPOINT", "hunyuan_3d_v2.1.safetensors"), folder, ", ".join(seen) or "nothing",
            )
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
