from img2mesh.pipeline import SINGLE_FILE_CONFIG, single_file_checkpoint


def test_uses_comfyui_file_when_present(tmp_path, monkeypatch):
    monkeypatch.setenv("CHECKPOINT_DIR", str(tmp_path))
    monkeypatch.delenv("HUNYUAN_CHECKPOINT", raising=False)
    assert single_file_checkpoint() is None
    (tmp_path / "hunyuan_3d_v2.1.safetensors").write_bytes(b"x")
    assert single_file_checkpoint() == tmp_path / "hunyuan_3d_v2.1.safetensors"


def test_custom_file_name(tmp_path, monkeypatch):
    monkeypatch.setenv("CHECKPOINT_DIR", str(tmp_path))
    monkeypatch.setenv("HUNYUAN_CHECKPOINT", "mine.safetensors")
    (tmp_path / "hunyuan_3d_v2.1.safetensors").write_bytes(b"x")
    assert single_file_checkpoint() is None
    (tmp_path / "mine.safetensors").write_bytes(b"x")
    assert single_file_checkpoint().name == "mine.safetensors"


def test_bundled_config_is_the_2_1_shape_model():
    text = SINGLE_FILE_CONFIG.read_text()
    assert "HunYuanDiTPlain" in text and "conditioner:" in text and "vae:" in text
