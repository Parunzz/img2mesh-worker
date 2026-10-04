"""Microsoft TRELLIS.2-4B, shape only (image: ghcr.io/parunzz/img2mesh-worker:trellis2).

Only the five shape models are loaded (no texture stage). Background removal is
the worker's own, so TRELLIS.2's default remover (briaai/RMBG-2.0, licensed
non-commercial) is never downloaded or used.

All weights live under TRELLIS_ROOT (mounted from TRELLIS_DIR):

    TRELLIS.2-4B/                    pipeline.json + ckpts/  (microsoft/TRELLIS.2-4B)
    TRELLIS-image-large/ckpts/       ss_dec_conv3d_16l8_fp16.*  (microsoft/TRELLIS-image-large)
    dinov3-vitl16-pretrain-lvd1689m/ (facebook/..., gated: needs HF_TOKEN to download)

Anything missing is downloaded there once.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from ..pipeline import Engine, GenerateOptions

log = logging.getLogger("img2mesh")

ROOT = Path(os.environ.get("TRELLIS_ROOT", "/models/trellis2"))
MAIN_REPO = "microsoft/TRELLIS.2-4B"
DINO_REPO = "facebook/dinov3-vitl16-pretrain-lvd1689m"
SHAPE_MODELS = (
    "sparse_structure_flow_model",
    "sparse_structure_decoder",
    "shape_slat_flow_model_512",
    "shape_slat_flow_model_1024",
    "shape_slat_decoder",
)
# detail -> TRELLIS.2 pipeline type
PIPELINES = {512: "512", 1024: "1024_cascade", 1536: "1536_cascade"}


def local_path(repo_path: str) -> Path:
    """'microsoft/TRELLIS-image-large/ckpts/x' -> ROOT/TRELLIS-image-large/ckpts/x;
    a bare 'ckpts/x' belongs to TRELLIS.2-4B."""
    parts = repo_path.split("/")
    if parts[0] == "ckpts":
        return ROOT / "TRELLIS.2-4B" / repo_path
    return ROOT.joinpath(*parts[1:])


def _download(repo: str, folder: Path, patterns: list[str]) -> None:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    log.info("downloading %s (%s) into %s", repo, ", ".join(patterns), folder)
    try:
        snapshot_download(repo, local_dir=folder, allow_patterns=patterns)
    except (GatedRepoError, RepositoryNotFoundError) as error:
        raise RuntimeError(
            f"{repo} is gated. Request access on https://huggingface.co/{repo}, then put a "
            f"Hugging Face token in .env as HF_TOKEN=..., or copy the model into {folder}."
        ) from error


def ensure_weights() -> dict:
    """Download whatever shape weights are missing; return pipeline.json args."""
    main = ROOT / "TRELLIS.2-4B"
    if not (main / "pipeline.json").is_file():
        _download(MAIN_REPO, main, ["pipeline.json"])
    args = json.loads((main / "pipeline.json").read_text())["args"]
    for name in SHAPE_MODELS:
        repo_path = args["models"][name]
        path = local_path(repo_path)
        if not (path.with_suffix(".json").is_file() and path.with_suffix(".safetensors").is_file()):
            parts = repo_path.split("/")
            repo, inner = (MAIN_REPO, repo_path) if parts[0] == "ckpts" else ("/".join(parts[:2]), "/".join(parts[2:]))
            _download(repo, path.parents[len(Path(inner).parts) - 1], [f"{inner}.json", f"{inner}.safetensors"])
    dino = ROOT / DINO_REPO.split("/")[1]
    if not (dino / "config.json").is_file():
        _download(DINO_REPO, dino, ["*.json", "*.safetensors"])
    return args


def crop_like_trellis(image: Image.Image) -> Image.Image:
    """TRELLIS.2's own preprocessing for an RGBA image: square crop around the
    subject, background multiplied to black, at most 1024 px."""
    scale = min(1, 1024 / max(image.size))
    if scale < 1:
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
    alpha = np.array(image)[:, :, 3]
    found = np.argwhere(alpha > 0.8 * 255)
    if not len(found):
        raise ValueError("the image is empty after background removal")
    top, left = found.min(axis=0)
    bottom, right = found.max(axis=0)
    cx, cy = (left + right) / 2, (top + bottom) / 2
    size = max(right - left, bottom - top)
    image = image.crop((cx - size // 2, cy - size // 2, cx + size // 2, cy + size // 2))
    rgba = np.array(image).astype(np.float32) / 255
    return Image.fromarray((rgba[:, :, :3] * rgba[:, :, 3:4] * 255).astype(np.uint8))


class Trellis2Engine(Engine):
    name = "trellis2"
    raw_up, raw_front = "+z", "-y"  # TRELLIS works Z-up; its GLB export maps -Y to glTF's front
    detail_choices = tuple(PIPELINES)

    def __init__(self) -> None:
        super().__init__()
        import torch
        from trellis2 import models
        from trellis2.modules import image_feature_extractor
        from trellis2.pipelines import Trellis2ImageTo3DPipeline, samplers

        args = ensure_weights()
        loaded = {}
        for name in SHAPE_MODELS:
            log.info("loading %s", name)
            loaded[name] = models.from_pretrained(str(local_path(args["models"][name])))
        pipeline = Trellis2ImageTo3DPipeline(loaded, low_vram=os.environ.get("TRELLIS_LOW_VRAM", "1") == "1")
        for stage in ("sparse_structure", "shape_slat"):
            spec = args[f"{stage}_sampler"]
            setattr(pipeline, f"{stage}_sampler", getattr(samplers, spec["name"])(**spec["args"]))
            setattr(pipeline, f"{stage}_sampler_params", spec["params"])
        pipeline.shape_slat_normalization = args["shape_slat_normalization"]
        pipeline.image_cond_model = image_feature_extractor.DinoV3FeatureExtractor(model_name=str(ROOT / DINO_REPO.split("/")[1]))
        pipeline.rembg_model = None  # never used: images arrive with transparency
        pipeline.to(torch.device("cuda"))
        self.pipeline = pipeline

    def make_mesh(self, image, options: GenerateOptions, on_progress=None):
        import torch

        p = self.pipeline
        kind = PIPELINES[options.detail or 1024]
        overrides = {}
        if options.steps:
            overrides["steps"] = options.steps
        if options.guidance_scale:
            overrides["guidance_strength"] = options.guidance_scale
        progress = on_progress or (lambda *_: None)
        seconds = {}

        start = time.monotonic()
        torch.manual_seed(options.seed)
        cropped = crop_like_trellis(image)
        cond_512 = p.get_cond([cropped], 512)
        cond_1024 = p.get_cond([cropped], 1024) if kind != "512" else None
        progress(1, 4)
        coords = p.sample_sparse_structure(cond_512, 64 if kind == "1024" else 32, 1, overrides)
        progress(2, 4)
        if kind == "512":
            slat = p.sample_shape_slat(cond_512, p.models["shape_slat_flow_model_512"], coords, overrides)
            resolution = 512
        else:
            slat, resolution = p.sample_shape_slat_cascade(
                cond_512, cond_1024,
                p.models["shape_slat_flow_model_512"], p.models["shape_slat_flow_model_1024"],
                512, int(kind.split("_")[0]), coords, overrides,
            )
        progress(3, 4)
        torch.cuda.empty_cache()
        meshes, _ = p.decode_shape_slat(slat, resolution)
        mesh = meshes[0]
        seconds["shape"] = time.monotonic() - start

        start = time.monotonic()
        mesh.fill_holes()
        if len(mesh.faces) > options.max_faces:
            mesh.simplify(options.max_faces)
        out = trimesh.Trimesh(mesh.vertices.cpu().numpy(), mesh.faces.cpu().numpy().astype(np.int64), process=True)
        torch.cuda.empty_cache()
        progress(4, 4)
        seconds["cleanup"] = time.monotonic() - start
        return out, seconds
