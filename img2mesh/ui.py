"""Test page: the same pipeline as `serve`, driven by hand at http://localhost:7860."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import trimesh

from .pipeline import GenerateOptions, Img2Mesh
from .printable import PrintOptions, to_gltf_axes

AXES = ["+x", "-x", "+y", "-y", "+z", "-z"]


def launch(host: str = "0.0.0.0", port: int = 7860) -> None:
    import gradio as gr

    engine = Img2Mesh()
    out_dir = Path(tempfile.mkdtemp(prefix="img2mesh-"))

    def run(image, height_mm, flat_back, flat_bottom, remove_background, seed, steps, octree, up, front, progress=gr.Progress()):
        if image is None:
            raise gr.Error("Upload an image first")
        generate = GenerateOptions(remove_background=remove_background, seed=int(seed), steps=int(steps), octree_resolution=int(octree))
        printable = PrintOptions(height_mm=float(height_mm), flat_back=float(flat_back), flat_bottom=float(flat_bottom), up=up, front=front)
        try:
            generate.validate()
            printable.validate()
        except ValueError as error:
            raise gr.Error(str(error)) from error
        stl, stats = engine.to_stl(image, generate, printable, on_progress=lambda s, t: progress(s / t, desc="Generating shape"))
        path = out_dir / f"img2mesh-seed{int(seed)}.stl"
        path.write_bytes(stl)
        # The viewer gets a GLB: it shows STL files mirrored. The download stays STL.
        preview = out_dir / f"img2mesh-seed{int(seed)}-preview.glb"
        mesh = trimesh.load(io.BytesIO(stl), file_type="stl")
        preview.write_bytes(to_gltf_axes(mesh).export(file_type="glb"))
        return str(preview), str(path), stats

    with gr.Blocks(title="img2mesh") as page:
        gr.Markdown("## img2mesh — photo to printable STL\nPowered by Tencent Hunyuan")
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", image_mode="RGBA", label="Image (PNG with transparency, or a photo)")
                height_mm = gr.Number(value=150, label="Height (mm)")
                with gr.Accordion("Options", open=False):
                    remove_background = gr.Radio(["auto", "always", "never"], value="auto", label="Remove background (auto = only if the image has no transparency)")
                    flat_back = gr.Slider(0, 0.4, value=0.04, step=0.01, label="Flat back: share of depth cut off")
                    flat_bottom = gr.Slider(0, 0.4, value=0.02, step=0.01, label="Flat bottom: share of height cut off")
                    seed = gr.Number(value=1234, precision=0, label="Seed (change it for a different result)")
                    steps = gr.Slider(10, 100, value=50, step=1, label="Steps")
                    octree = gr.Radio([256, 384, 512], value=384, label="Detail (octree resolution)")
                    up = gr.Dropdown(AXES, value="+y", label="Raw up axis")
                    front = gr.Dropdown(AXES, value="+z", label="Raw front axis")
                button = gr.Button("Generate", variant="primary")
            with gr.Column():
                model = gr.Model3D(label="Preview (the STL download is in mm, Z up, front facing -Y)")
                file = gr.File(label="Download STL")
                stats = gr.JSON(label="Stats")
        button.click(run, [image, height_mm, flat_back, flat_bottom, remove_background, seed, steps, octree, up, front], [model, file, stats])

    page.queue(max_size=4).launch(server_name=host, server_port=port)
