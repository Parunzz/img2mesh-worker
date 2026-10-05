"""Test page: the same pipeline as `serve`, driven by hand at http://localhost:7860."""

from __future__ import annotations

import io
import logging
import os
import tempfile
import threading
from pathlib import Path

import trimesh

from .pipeline import GenerateOptions, is_gpu_fatal, load_engine
from .printable import PrintOptions, to_gltf_axes

AXES = ["engine default", "+x", "-x", "+y", "-y", "+z", "-z"]
ENGINE_INFO = {
    "hunyuan": ("Tencent Hunyuan3D 2.1", "Powered by Tencent Hunyuan", 384, 50, 5.0),
    "trellis2": ("Microsoft TRELLIS.2-4B", "Microsoft TRELLIS.2 (MIT licence)", 1024, 12, 7.5),
}


def launch(host: str = "0.0.0.0", port: int = 7860) -> None:
    import gradio as gr

    engine = load_engine()
    title, credit, default_detail, default_steps, default_guidance = ENGINE_INFO[engine.name]
    out_dir = Path(tempfile.mkdtemp(prefix="img2mesh-"))

    def run(image, height_mm, flat_back, flat_bottom, remove_background, seed, steps, guidance, detail, max_faces, up, front, progress=gr.Progress()):
        if image is None:
            raise gr.Error("Upload an image first")
        generate = GenerateOptions(
            remove_background=remove_background, seed=int(seed), steps=int(steps),
            guidance_scale=float(guidance), detail=int(detail), max_faces=int(max_faces),
        )
        printable = PrintOptions(
            height_mm=float(height_mm), flat_back=float(flat_back), flat_bottom=float(flat_bottom),
            up=None if up == AXES[0] else up, front=None if front == AXES[0] else front,
        )
        try:
            generate.validate()
            printable.validate()
            stl, stats = engine.to_stl(image, generate, printable, on_progress=lambda s, t: progress(s / t, desc="Generating shape"))
        except ValueError as error:
            raise gr.Error(str(error)) from error
        except RuntimeError as error:
            if not is_gpu_fatal(error):
                raise
            # The GPU context is broken for this process; restart it (Docker brings it back).
            logging.getLogger("img2mesh").error("GPU error, restarting the test page: %s", error)
            threading.Timer(2, os._exit, [3]).start()
            raise gr.Error(
                "GPU error (often: not enough GPU memory). The test page is restarting; reload it in about 3 minutes. "
                "Try a lower Detail next time."
            ) from error
        name = f"img2mesh-{engine.name}-seed{int(seed)}"
        path = out_dir / f"{name}.stl"
        path.write_bytes(stl)
        # The viewer gets a GLB: it shows STL files mirrored. The download stays STL.
        preview = out_dir / f"{name}-preview.glb"
        mesh = trimesh.load(io.BytesIO(stl), file_type="stl")
        preview.write_bytes(to_gltf_axes(mesh).export(file_type="glb"))
        return str(preview), str(path), stats

    with gr.Blocks(title="img2mesh") as page:
        gr.Markdown(f"## img2mesh — photo to printable STL\nEngine: **{title}** (set ENGINE in .env and restart to switch) · {credit}")
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", image_mode="RGBA", label="Image (PNG with transparency, or a photo)")
                height_mm = gr.Number(value=150, label="Height (mm)")
                with gr.Accordion("Options", open=False):
                    remove_background = gr.Radio(["auto", "always", "never"], value="auto", label="Remove background (auto = only if the image has no transparency)")
                    detail = gr.Radio(list(engine.detail_choices), value=default_detail, label="Detail (higher = finer, slower, more VRAM)")
                    seed = gr.Number(value=1234, precision=0, label="Seed (change it for a different result)")
                    steps = gr.Slider(4, 100, value=default_steps, step=1, label="Steps")
                    guidance = gr.Slider(1, 15, value=default_guidance, step=0.5, label="Guidance (how closely to follow the image)")
                    max_faces = gr.Slider(100_000, 2_000_000, value=300_000, step=50_000, label="Max faces (more keeps finer detail, bigger STL)")
                    flat_back = gr.Slider(0, 0.4, value=0.04, step=0.01, label="Flat back: share of depth cut off")
                    flat_bottom = gr.Slider(0, 0.4, value=0.02, step=0.01, label="Flat bottom: share of height cut off")
                    up = gr.Dropdown(AXES, value=AXES[0], label=f"Raw up axis ({engine.name} default: {engine.raw_up})")
                    front = gr.Dropdown(AXES, value=AXES[0], label=f"Raw front axis ({engine.name} default: {engine.raw_front})")
                button = gr.Button("Generate", variant="primary")
            with gr.Column():
                model = gr.Model3D(label="Preview (the STL download is in mm, Z up, front facing -Y)")
                file = gr.File(label="Download STL")
                stats = gr.JSON(label="Stats")
        inputs = [image, height_mm, flat_back, flat_bottom, remove_background, seed, steps, guidance, detail, max_faces, up, front]
        # Lock the button while a job runs so a second click can't queue another
        # few-minute GPU job; `.then` re-enables it even when the job fails.
        button.click(
            lambda: gr.update(value="Generating… (a few minutes)", interactive=False), None, button, queue=False, trigger_mode="once"
        ).then(
            run, inputs, [model, file, stats], concurrency_limit=1
        ).then(
            lambda: gr.update(value="Generate", interactive=True), None, button, queue=False
        )

    page.queue(max_size=4).launch(server_name=host, server_port=port)
