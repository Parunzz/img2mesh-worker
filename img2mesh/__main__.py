"""img2mesh: generate | serve | ui"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path


def _print_options(args):
    from .printable import PrintOptions

    return PrintOptions(height_mm=args.height_mm, flat_back=args.flat_back, flat_bottom=args.flat_bottom, up=args.up, front=args.front)


def _generate_options(args):
    from .pipeline import GenerateOptions

    return GenerateOptions(
        remove_background=args.remove_background, seed=args.seed, steps=args.steps,
        guidance_scale=args.guidance, detail=args.detail, max_faces=args.max_faces,
    )


def cmd_generate(args) -> int:
    from PIL import Image

    from .pipeline import load_engine

    printable, generate = _print_options(args), _generate_options(args)
    printable.validate()
    generate.validate()
    engine = load_engine()
    stl, stats = engine.to_stl(Image.open(args.image), generate, printable)
    output = Path(args.output or Path(args.image).with_suffix(".stl"))
    output.write_bytes(stl)
    print(json.dumps({"output": str(output), **stats}, indent=2))
    return 0


def cmd_serve(args) -> int:
    from .pipeline import load_engine
    from .protocol import Client
    from .serve import serve

    client = Client(os.environ.get("SITE_URL", ""), os.environ.get("WORKER_SECRET", ""), os.environ.get("WORKER_NAME", ""))
    serve(load_engine, client, once=args.once)
    return 0


def cmd_ui(args) -> int:
    from .ui import launch

    launch(host=args.host, port=args.port)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="img2mesh", description="Photo to printable STL (engine: ENGINE=hunyuan or trellis2)")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="one image -> one STL")
    gen.add_argument("image")
    gen.add_argument("-o", "--output", help="default: next to the image, .stl")
    gen.add_argument("--height-mm", type=float, required=True)
    gen.add_argument("--flat-back", type=float, default=0.04, help="share of depth cut off the back (0-0.5)")
    gen.add_argument("--flat-bottom", type=float, default=0.02, help="share of height cut off the bottom (0-0.5)")
    gen.add_argument("--up", help="raw up axis; default: the engine's")
    gen.add_argument("--front", help="raw front axis; default: the engine's")
    gen.add_argument("--remove-background", choices=["auto", "always", "never"], default="auto")
    gen.add_argument("--seed", type=int, default=1234)
    gen.add_argument("--steps", type=int, help="default: the engine's (Hunyuan 50, TRELLIS.2 12)")
    gen.add_argument("--guidance", type=float, help="how closely to follow the image; default: the engine's")
    gen.add_argument("--detail", type=int, choices=[256, 384, 512, 1024, 1536],
                     help="Hunyuan: 256/384/512 (default 384); TRELLIS.2: 512/1024/1536 (default 1024)")
    gen.add_argument("--max-faces", type=int, default=300_000)
    gen.set_defaults(func=cmd_generate)

    srv = sub.add_parser("serve", help="pull jobs from SITE_URL (protocol v1)")
    srv.add_argument("--once", action="store_true", help="handle at most one job, then exit")
    srv.set_defaults(func=cmd_serve)

    ui = sub.add_parser("ui", help="test page in the browser")
    ui.add_argument("--host", default="0.0.0.0")
    ui.add_argument("--port", type=int, default=7860)
    ui.set_defaults(func=cmd_ui)

    args = parser.parse_args(argv)
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
