"""Hunyuan3D 2.1, shape only (image: ghcr.io/parunzz/img2mesh-worker:latest)."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from ..pipeline import Engine, GenerateOptions

log = logging.getLogger("img2mesh")

MODEL = os.environ.get("HUNYUAN_MODEL", "tencent/Hunyuan3D-2.1")
# Hunyuan's own config for the 2.1 shape model, for single-file checkpoints
# (such as ComfyUI's hunyuan_3d_v2.1.safetensors) that don't carry one.
SINGLE_FILE_CONFIG = Path(__file__).resolve().parent.parent / "hunyuan3d-dit-v2-1.yaml"


def single_file_checkpoint() -> Path | None:
    """A single-file checkpoint mounted at CHECKPOINT_DIR, if there is one."""
    folder = Path(os.environ.get("CHECKPOINT_DIR", "/checkpoints"))
    path = folder / os.environ.get("HUNYUAN_CHECKPOINT", "hunyuan_3d_v2.1.safetensors")
    return path if path.is_file() else None


class HunyuanEngine(Engine):
    name = "hunyuan"
    raw_up, raw_front = "+y", "+z"  # glTF-style output
    detail_choices = (256, 384, 512)

    def __init__(self) -> None:
        super().__init__()
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
        self.remove_floaters = FloaterRemover()
        self.remove_degenerate = DegenerateFaceRemover()
        self.reduce_faces = FaceReducer()

    def make_mesh(self, image, options: GenerateOptions, on_progress=None):
        import torch

        steps = options.steps or 50
        start = time.monotonic()
        mesh = self.shape(
            image=image,
            num_inference_steps=steps,
            guidance_scale=options.guidance_scale or 5.0,
            generator=torch.Generator().manual_seed(options.seed),
            octree_resolution=options.detail or 384,
            callback=(lambda step, *_: on_progress(step, steps)) if on_progress else None,
            callback_steps=1 if on_progress else None,
        )[0]
        seconds = {"shape": time.monotonic() - start}

        start = time.monotonic()
        mesh = self.remove_floaters(mesh)
        mesh = self.remove_degenerate(mesh)
        mesh = self.reduce_faces(mesh, max_facenum=options.max_faces)
        seconds["cleanup"] = time.monotonic() - start
        return mesh, seconds
