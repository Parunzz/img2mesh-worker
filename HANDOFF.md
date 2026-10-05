# Handoff: img2mesh-worker on the GPU PC

For a coding agent continuing this work **on the owner's Windows PC with the GPU**. The previous agent worked on a Linux server without a usable GPU, so everything below was built and unit-tested there but the GPU path was only ever exercised by the owner pasting logs. You can run it yourself: do that first.

Reply to the owner in English, short; they may write in Thai.

## What this repo is

Photo → printable STL on a local NVIDIA GPU, in Docker. Two engines, chosen with `ENGINE` in `.env` (each has its own image, built by `.github/workflows/image.yml` and published to `ghcr.io/parunzz/img2mesh-worker:{hunyuan,trellis2}`):

- `hunyuan`: Tencent Hunyuan3D 2.1, shape only. Works end to end on the PC. Can load ComfyUI's `hunyuan_3d_v2.1.safetensors` (`COMFYUI_CHECKPOINTS`).
- `trellis2`: Microsoft TRELLIS.2-4B, shape only. **Being brought up now** (see Status).

Three commands in one image (`img2mesh/__main__.py`): `generate` (CLI), `ui` (Gradio test page, port 7860), `serve` (pulls jobs from the shop website, see `PROTOCOL.md`).

Read `README.md` (setup), `PROTOCOL.md` (the website contract), `NOTICE` (licences) before changing anything.

### Code map

| Path | Role |
|---|---|
| `img2mesh/pipeline.py` | `GenerateOptions`, `Engine` base (background removal with rembg u2net, `to_stl`), `load_engine()`, `is_gpu_fatal()` |
| `img2mesh/engines/hunyuan.py` | Hunyuan engine |
| `img2mesh/engines/trellis2.py` | TRELLIS.2 engine: weight download/lookup, DINOv3 lookup, FA3 guard, shape-only sampling |
| `img2mesh/printable.py` | Pure geometry: `repair()` (close holes), orient, flat back/bottom cut, scale to `height_mm`; `to_gltf_axes()` for the preview |
| `img2mesh/serve.py`, `protocol.py` | Pull-protocol client + worker loop |
| `img2mesh/ui.py` | Test page |
| `Dockerfile.hunyuan`, `Dockerfile.trellis2` | Images. TRELLIS.2 compiles CuMesh, FlexGEMM, o-voxel in a build stage |
| `docker/stubs/nvdiffrast` | Stub so `o_voxel` imports without NVIDIA's non-commercial nvdiffrast |
| `tests/` | CPU-only tests (geometry, protocol, serve loop, engine helpers) |

## The owner's machine

- Windows 11, Docker Desktop (WSL 2 backend), RTX 5060 Ti **16 GB** (Blackwell, compute capability **12.0**), 32 GB RAM.
- WSL memory raised to **25 GB + 16 GB swap** via WSL Settings (default was 16 GB, which OOM-killed TRELLIS.2).
- Repo at `D:\Works\ZeNos\img2mesh-worker`. ComfyUI at `C:\AI\ComfyUI` (has `models/checkpoints/hunyuan_3d_v2.1.safetensors`).
- `.env` currently: `ENGINE=trellis2`, `COMFYUI_CLIP_VISION=C:/AI/ComfyUI/models/clip_vision`, placeholders for `SITE_URL` / `WORKER_SECRET` / `WORKER_NAME`.
- TRELLIS.2 weights already downloaded to `models\trellis2\` (≈9 GB; the owner's connection is ~3 MB/s, so never make them re-download).

## Status of the TRELLIS.2 bring-up (as of 2026-10-05)

Fixed and published, in order of discovery:

1. `Failed to find C compiler` → `gcc libc6-dev` in the runtime image (Triton compiles at runtime).
2. `flash_fwd_launch_template.h: invalid argument` at the first sampling step → xformers 0.0.31 enables FlashAttention-3 for every GPU ≥ 9.0 but its FA3 kernels are Hopper-only. `keep_flash_attention_3_on_hopper_only()` turns it off; log shows `xformers FlashAttention-3: off (compute capability 12.x)`.
3. Container `EOF` / killed at 50 % → WSL VM had 16 GB. Fixed by the owner's WSL memory change.
4. Detail **1024**: 1024 sampling ran at ~13 s/step (VRAM full, Windows sysmem fallback), then decode died with `CUDA driver error: device not ready`; afterwards every run failed with `CUDACachingAllocator ... INTERNAL ASSERT` (dead CUDA context). Mitigation shipped: `is_gpu_fatal()` → `serve` reports the job failed and exits, `ui` shows a message and exits, `restart: unless-stopped` brings a fresh process. **1024 itself is not solved.**
5. Detail **512** works: ~59 s shape, depth 50 mm on a 150 mm model, but `"watertight": false` (holes). Fix shipped: `printable.repair()` closes holes with pymeshlab before cutting. **Not yet confirmed on the PC.**
6. Hole filling made it worse. `meshing_close_holes` up to 5000 edges (plus a `selfintersection=False` retry) capped TRELLIS.2's big gaps with flat fans cutting through the model (faces 15,000x the median area) and still was not watertight; it also crashed once on non-manifold edges. Now `repair()` drops loose pieces < 5 % of the diagonal and fills only holes <= 100 edges, once. **TRELLIS.2 output is still not watertight**; TRELLIS.2's own `cumesh.remeshing.remesh_narrow_band_dc` did not make it watertight either (tested 256/512, band 1/2).
7. Shape quality on a straight-on product photo (cartoon chick, 640x480 JPG) at Detail 512 is seed-dependent: seed 1234 gave a flat relief lying on its back, seed 7 a standing chick with an invented long back. TRELLIS.2's own example (crown) comes out correct, so the pipeline and axes (`+z` up, `-y` front) are right; this is the model guessing depth. Also seen: a second `generate()` in the same process died with `CUDA driver error: device not ready` at 512.

Known open issue: `COMFYUI_CLIP_VISION` did not take effect: the log says `downloading Comfy-Org/TRELLIS.2 (clip_vision/dino_v3_vit_l.safetensors)` instead of using `/clip_vision/...`. Harmless (same weights, cached now) but unexplained.

## Your tasks, in order

1. **Update and verify the hole fix.** `git pull`, `docker compose pull`, `docker compose --profile ui up ui`, generate at Detail 512 with the owner's test image. Expect no flat sheets in the preview; `watertight` will still be false for TRELLIS.2 (see Status 6). Making it watertight needs a volume remesh (voxelize + fill + marching cubes, or a winding-number based remesher), not hole filling.
2. **Find why `COMFYUI_CLIP_VISION` is ignored.** Check `docker compose config | findstr clip_vision`, `dir C:\AI\ComfyUI\models\clip_vision`, and inside the container `ls -la /clip_vision`. Note `ensure_dino()` prefers an existing `models/trellis2/dinov3/` folder only if it has `config.json`; otherwise it looks in `/clip_vision` for `dino_v3_vit_l.safetensors` or `dino_v3_L_naf_fp32.safetensors` (or `DINOV3_FILE`).
3. **Make Detail 1024 fit 16 GB** (the owner wants better faces than 512). Ideas, cheapest first; measure VRAM with `nvidia-smi -l 1` while it runs:
   - Pass a lower `max_num_tokens` to `sample_shape_slat_cascade` (TRELLIS.2 default 49152); expose it in `GenerateOptions`/UI/protocol.
   - Decode in chunks or free everything else first (`torch.cuda.empty_cache()`, move DINOv3 and flow models to CPU before `decode_shape_slat`; low-VRAM mode should already do this, so verify what is actually resident).
   - NVIDIA Control Panel → *CUDA – Sysmem Fallback Policy* = *Prefer No Sysmem Fallback* turns the silent 13 s/step slowdown into a clear OOM, useful while measuring.
   - Windows TDR (GPU watchdog) may be what kills the long decode; check Event Viewer for `nvlddmkm` / display-driver reset events before touching registry `TdrDelay` (ask the owner first).
4. **Cut RAM use** (≈12–14 GB resident now): build each TRELLIS.2 model on the `meta` device and `load_state_dict(..., assign=True)` straight in bf16 instead of fp32-then-convert (also speeds the ~55 s per-model load); skip `shape_slat_flow_model_1024` when Detail is 512 (load lazily on first 1024 job).
5. **Progress bar**: the UI sits at 50 % through the whole 1024 cascade + decode. Report finer progress from `make_mesh` (e.g. per sampler step via a wrapper around the sampler, or at least before/after upsample and decode).
6. **Face quality comparison** for the owner: same sculpture-style input, Hunyuan (Detail 512) vs TRELLIS.2 (512 and, once it fits, 1024). Report `size_mm`, timings and screenshots. The owner prints at ~150 mm tall.

## How to work here

- Tests (CPU, no GPU needed): `docker run --rm -v "${PWD}:/app" -w /app python:3.10-slim sh -c "pip install -r requirements-dev.txt && pytest -q"`. Keep them green; add a test for any pure logic you change.
- Fast iteration on engine code without rebuilding the image: mount the package over the image's copy, e.g. add `- ./img2mesh:/app/img2mesh` under `volumes` in a local `docker-compose.override.yml` (do not commit it).
- Image changes (Dockerfile, requirements) need a rebuild: locally `docker compose build ui` (TRELLIS.2 compiles CUDA for 15–30 min), or push to `main` and let CI publish (~1 h cold, faster with the registry cache).
- Logs are UTC inside the container; the owner is UTC+7.
- Commit to `main` in small, described commits; end messages with the co-author line the session gives you. CI builds both images on every push that changes code.

## Constraints: do not break these

- **Licences** (see `NOTICE`): never add `briaai/RMBG-2.0` (CC BY-NC) or real `nvdiffrast` (NVIDIA non-commercial). The shape-only path needs neither. Hunyuan's licence excludes the EU, UK and South Korea. DINOv3 weights stay under Meta's DINOv3 License even from Comfy-Org's ungated copy.
- **No account or token** should be required to run either engine.
- **Blackwell (sm_120)**: keep torch cu128 (≥ 2.7). Prebuilt xformers 0.0.31.post1 has FA3 only for Hopper; flash-attn is not installed (compiling it needs more RAM than the CI runner has).
- **The worker stays generic**: image in, STL out. No shop concepts (Frames, Customers, quotas) in this repo. Protocol changes are a new protocol version, documented in `PROTOCOL.md`.
- Output contract: binary STL in mm, Z up, standing on Z = 0, front facing −Y, watertight where possible.

## The other side: the shop (separate private repo)

The website is `Parunzz/zenos-atelier` (Next.js + Supabase). Its task **T45** (`docs/tasks/T45-photo-frame-relief.md`) is the site side of `PROTOCOL.md`: Frames, a Generation queue, `/api/worker/claim` and `/api/worker/jobs/[id]`, quota, ordering. Not started. It waits on three owner decisions asked on 2026-10-05:

1. Generate a dev `WORKER_SECRET` and point the worker at the dev site over Tailscale (`SITE_URL=https://parunyu-server.tail2627d5.ts.net`).
2. Split T45 into the worker connection + an admin test page first, then the customer flow as a new task T53.
3. A plain photo (not a sculpture-style render) produced a flat 0.91 mm Hunyuan mesh; only a ChatGPT-made "white relief sculpture" image gave depth. Test a plain photo + Frame in TRELLIS.2 before choosing how to restyle customer photos (paid image API vs local image-edit model).

Do not change the shop from this machine unless the owner asks; that work follows the shop repo's own `AGENTS.md`.
