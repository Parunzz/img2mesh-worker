"""Microsoft TRELLIS.2-4B, shape only (image: ghcr.io/parunzz/img2mesh-worker:trellis2).

Only the five shape models are loaded (no texture stage). Background removal is
the worker's own, so TRELLIS.2's default remover (briaai/RMBG-2.0, licensed
non-commercial) is never downloaded or used.

All weights live under TRELLIS_ROOT (mounted from TRELLIS_DIR):

    TRELLIS.2-4B/                    pipeline.json + ckpts/  (microsoft/TRELLIS.2-4B)
    TRELLIS-image-large/ckpts/       ss_dec_conv3d_16l8_fp16.*  (microsoft/TRELLIS-image-large)
    dinov3/                          DINOv3 ViT-L image encoder (see below)

Anything missing is downloaded there once; no account or token is needed.

DINOv3 (Meta's image encoder) comes from, in order: an HF-format folder already in
TRELLIS_ROOT/dinov3 (config.json + model.safetensors); a ComfyUI clip_vision file
mounted at /clip_vision (dino_v3_vit_l.safetensors, or dino_v3_L_naf_fp32.safetensors,
which holds the same weights plus an add-on that is ignored); else a download of
Comfy-Org's ungated copy. Meta's own repo is gated, which is why it is not used.
The tensors of these files match transformers' DINOv3ViTModel name for name and
shape for shape; the config (img2mesh/dinov3-vitl16-config.json) supplies what the
single files lack.
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
DINO_REPO = "Comfy-Org/TRELLIS.2"  # ungated copy of DINOv3 ViT-L/16
DINO_FILE_IN_REPO = "clip_vision/dino_v3_vit_l.safetensors"
DINO_CONFIG = Path(__file__).resolve().parent.parent / "dinov3-vitl16-config.json"
CLIP_VISION = Path(os.environ.get("CLIP_VISION_DIR", "/clip_vision"))
DINO_FILE_NAMES = ("dino_v3_vit_l.safetensors", "dino_v3_L_naf_fp32.safetensors")
# A few tensors any DINOv3 ViT-L file must have, to refuse a wrong file early
# (a missing tensor would otherwise be silently random-initialised).
DINO_REQUIRED = {
    "embeddings.patch_embeddings.weight": [1024, 3, 16, 16],
    "layer.23.mlp.up_proj.weight": [4096, 1024],
    "norm.weight": [1024],
}
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

    log.info("downloading %s (%s) into %s", repo, ", ".join(patterns), folder)
    snapshot_download(repo, local_dir=folder, allow_patterns=patterns)


def check_dino_file(path: Path) -> None:
    """Refuse a file that is not a DINOv3 ViT-L/16 checkpoint."""
    from safetensors import safe_open

    with safe_open(str(path), framework="numpy") as f:
        keys = set(f.keys())
        for name, shape in DINO_REQUIRED.items():
            if name not in keys or list(f.get_slice(name).get_shape()) != shape:
                raise RuntimeError(f"{path} is not a DINOv3 ViT-L/16 checkpoint (no {name} {shape})")


def find_dino_file() -> Path | None:
    wanted = os.environ.get("DINOV3_FILE")
    for name in ([wanted] if wanted else DINO_FILE_NAMES):
        path = CLIP_VISION / name
        if path.is_file():
            return path
    return None


def ensure_dino() -> Path:
    """An HF-format folder (config.json + model.safetensors) for DinoV3FeatureExtractor."""
    folder = ROOT / "dinov3"
    if (folder / "config.json").is_file() and any(folder.glob("*.safetensors")):
        return folder
    source = find_dino_file()
    if source is None:
        _download(DINO_REPO, folder, [DINO_FILE_IN_REPO])
        source = folder / DINO_FILE_IN_REPO
    log.info("DINOv3 image encoder from %s", source)
    check_dino_file(source)
    # A small view folder: our config plus a link to the weights file, so nothing large is copied.
    view = Path(os.environ.get("HF_HOME", "/tmp")) / "img2mesh-dinov3"
    view.mkdir(parents=True, exist_ok=True)
    (view / "config.json").write_text(DINO_CONFIG.read_text())
    link = view / "model.safetensors"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(source)
    return view


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
        dino = ensure_dino()
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
        pipeline.image_cond_model = image_feature_extractor.DinoV3FeatureExtractor(model_name=str(dino))
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
