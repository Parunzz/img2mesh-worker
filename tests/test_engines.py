import numpy as np
import pytest
import trimesh
from PIL import Image

from img2mesh.pipeline import Engine, GenerateOptions, engine_name
from img2mesh.printable import PrintOptions


class ZUpEngine(Engine):
    """Like TRELLIS.2: raw output Z-up, facing -Y. Skips rembg (no model download)."""

    name = "fake"
    raw_up, raw_front = "+z", "-y"
    detail_choices = (512, 1024)

    def __init__(self):
        self.seen = None

    def prepare(self, image, mode):
        return image.convert("RGBA")

    def make_mesh(self, image, options, on_progress=None):
        self.seen = options
        # 1 wide (x), 0.3 deep (y), 2 tall (z), with a bump on the front (-y).
        body = trimesh.creation.box(extents=[1, 0.3, 2])
        bump = trimesh.creation.box(extents=[0.2, 0.2, 0.2]).apply_translation([0, -0.2, 0.5])
        return trimesh.util.concatenate([body, bump]), {"shape": 0.1}


def test_engine_axes_are_used_when_job_gives_none():
    stl, stats = ZUpEngine().to_stl(Image.new("RGB", (8, 8)), GenerateOptions(), PrintOptions(height_mm=100))
    mesh = trimesh.load(trimesh.util.wrap_as_stream(stl), file_type="stl")
    assert np.isclose(mesh.extents[2], 100)  # raw Z became the printed height
    assert stats["engine"] == "fake"
    # The bump (raw -y front) stays at the printed front (-y).
    assert mesh.vertices[mesh.vertices[:, 1].argmin()][2] > 50


def test_explicit_axes_override_engine():
    _, stats = ZUpEngine().to_stl(
        Image.new("RGB", (8, 8)), GenerateOptions(), PrintOptions(height_mm=100, flat_back=0, flat_bottom=0, up="+x", front="-y")
    )
    assert np.isclose(stats["size_mm"][2], 100)  # height now measured along raw X


def test_detail_must_suit_the_engine():
    with pytest.raises(ValueError, match="detail must be one of 512, 1024"):
        ZUpEngine().to_stl(Image.new("RGB", (8, 8)), GenerateOptions(detail=384), PrintOptions(height_mm=10))


@pytest.mark.parametrize("kwargs", [{"steps": 0}, {"guidance_scale": 30}, {"detail": 333}, {"max_faces": 10}, {"remove_background": "x"}])
def test_generate_options_reject_bad_values(kwargs):
    with pytest.raises(ValueError):
        GenerateOptions(**kwargs).validate()


def test_engine_name(monkeypatch):
    monkeypatch.delenv("ENGINE", raising=False)
    assert engine_name() == "hunyuan"
    monkeypatch.setenv("ENGINE", " TRELLIS2 ")
    assert engine_name() == "trellis2"
    monkeypatch.setenv("ENGINE", "comfy")
    with pytest.raises(ValueError):
        engine_name()


def test_trellis_crop_is_square_around_subject_and_black_outside():
    from img2mesh.engines.trellis2 import crop_like_trellis

    image = Image.new("RGBA", (400, 300), (0, 0, 0, 0))
    image.paste((200, 100, 50, 255), (100, 50, 200, 250))  # subject 100 wide, 200 tall
    out = crop_like_trellis(image)
    # Same arithmetic as TRELLIS.2 itself: last-minus-first pixel, halved to whole pixels.
    assert out.mode == "RGB" and out.width == out.height and abs(out.width - 200) <= 2
    assert out.getpixel((100, 100)) == (200, 100, 50)  # subject kept
    assert out.getpixel((5, 100)) == (0, 0, 0)  # outside the subject goes black


def test_trellis_crop_rejects_empty_image():
    from img2mesh.engines.trellis2 import crop_like_trellis

    with pytest.raises(ValueError):
        crop_like_trellis(Image.new("RGBA", (50, 50), (0, 0, 0, 0)))


def test_trellis_weight_paths(monkeypatch, tmp_path):
    import importlib

    monkeypatch.setenv("TRELLIS_ROOT", str(tmp_path))
    import img2mesh.engines.trellis2 as t

    t = importlib.reload(t)
    assert t.local_path("ckpts/shape_dec_next_dc_f16c32_fp16") == tmp_path / "TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16"
    assert t.local_path("microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16") == tmp_path / "TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16"


def _fake_dino(path, extra=None, patch_shape=(1024, 3, 16, 16)):
    from safetensors.numpy import save_file

    tensors = {
        "embeddings.patch_embeddings.weight": np.zeros(patch_shape, dtype=np.float16),
        "layer.23.mlp.up_proj.weight": np.zeros((4096, 1024), dtype=np.float16),
        "norm.weight": np.zeros(1024, dtype=np.float16),
        **(extra or {}),
    }
    save_file(tensors, str(path))


def test_dino_check_accepts_vit_l_with_extra_tensors(tmp_path):
    from img2mesh.engines.trellis2 import check_dino_file

    path = tmp_path / "dino_v3_L_naf_fp32.safetensors"
    _fake_dino(path, extra={"naf.image_encoder.encoder.0.bias": np.zeros(8, dtype=np.float16)})
    check_dino_file(path)  # the naf add-on is ignored


def test_dino_check_refuses_other_models(tmp_path):
    from img2mesh.engines.trellis2 import check_dino_file

    path = tmp_path / "dino_small.safetensors"
    _fake_dino(path, patch_shape=(384, 3, 16, 16))  # ViT-S, not ViT-L
    with pytest.raises(RuntimeError, match="not a DINOv3 ViT-L/16"):
        check_dino_file(path)


def test_dino_file_lookup_prefers_plain_then_naf(tmp_path, monkeypatch):
    import importlib

    monkeypatch.setenv("CLIP_VISION_DIR", str(tmp_path))
    monkeypatch.delenv("DINOV3_FILE", raising=False)
    import img2mesh.engines.trellis2 as t

    t = importlib.reload(t)
    assert t.find_dino_file() is None
    (tmp_path / "dino_v3_L_naf_fp32.safetensors").write_bytes(b"x")
    assert t.find_dino_file().name == "dino_v3_L_naf_fp32.safetensors"
    (tmp_path / "dino_v3_vit_l.safetensors").write_bytes(b"x")
    assert t.find_dino_file().name == "dino_v3_vit_l.safetensors"
    monkeypatch.setenv("DINOV3_FILE", "mine.safetensors")
    assert t.find_dino_file() is None


def test_bundled_dino_config_is_vit_l16():
    import json

    from img2mesh.engines.trellis2 import DINO_CONFIG

    config = json.loads(DINO_CONFIG.read_text())
    assert config["model_type"] == "dinov3_vit"
    assert (config["hidden_size"], config["num_hidden_layers"], config["patch_size"], config["num_register_tokens"]) == (1024, 24, 16, 4)
