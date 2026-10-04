# img2mesh-worker

Photo → printable STL, on your own NVIDIA GPU, in Docker.

`image → background removal → 3D model (shape) → clean-up → flat back and bottom, scaled to a height in mm → STL`

Two engines, picked with `ENGINE` in `.env`; each has its own Docker image:

| `ENGINE` | Model | Licence | Notes |
|---|---|---|---|
| `hunyuan` (default) | [Tencent Hunyuan3D 2.1](https://huggingface.co/tencent/Hunyuan3D-2.1) | Tencent community licence (not EU/UK/South Korea) | Reads the image at 518 px. Can reuse ComfyUI's file. |
| `trellis2` | [Microsoft TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B) | MIT (+ Meta DINOv3 licence for its image encoder) | Reads the image at 1024 px: more face detail. Needs a Hugging Face token once. |

Three ways to use the same image:

| Command | What it does |
|---|---|
| `generate` | One image in, one STL out, from the command line |
| `ui` | A test page at http://localhost:7860: upload, tweak, look, download |
| `serve` | A worker that pulls jobs from a website ([PROTOCOL.md](PROTOCOL.md)) and runs forever |

Powered by Tencent Hunyuan. See [NOTICE](NOTICE) for its licence: it does not
apply in the EU, the UK or South Korea.

## Requirements

- NVIDIA GPU with **12 GB VRAM or more** (tested target: RTX 5060 Ti 16 GB). RTX 50xx works: the image uses CUDA 12.8.
- About **30 GB free disk** (image + model weights).
- **Windows 11**: latest NVIDIA driver + [Docker Desktop](https://www.docker.com/products/docker-desktop/) (keep the default WSL 2 backend). Nothing else: Docker Desktop passes the GPU through.
- **Linux**: NVIDIA driver, Docker Engine and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

Check the GPU is visible to Docker:

```sh
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

## Install

```sh
git clone https://github.com/Parunzz/img2mesh-worker.git
cd img2mesh-worker
docker compose pull
```

`pull` downloads the ready-made image from GHCR. If it fails, `docker compose build` builds it locally instead (15–30 minutes).

The first run of any command downloads the model weights (several GB) into `models/Hunyuan3D-2.1` next to `docker-compose.yml`. Later runs start in about a minute.

### Already have Hunyuan3D-2.1 downloaded?

Point the worker at it instead of downloading again. Set **one** of these in `.env` (use forward slashes):

**ComfyUI** (the single file `hunyuan_3d_v2.1.safetensors` from [Comfy-Org](https://huggingface.co/Comfy-Org/hunyuan3D_2.1_repackaged), 7.4 GB): the folder that contains it.

```
COMFYUI_CHECKPOINTS=C:/AI/ComfyUI/models/checkpoints
```

It is mounted read-only. If your copy has another name, also set `HUNYUAN_CHECKPOINT=<file name>`. The file holds the same weights as Tencent's release, tensor for tensor; the config it lacks ships with this worker. `hunyuan3d-dit-v2-mv_fp16.safetensors` is a different (multi-view, 2.0) model and won't work.

**Tencent's layout** (Hunyuan's own app, or a clone from Hugging Face): the folder that contains `hunyuan3d-dit-v2-1/config.yaml` and `model.fp16.ckpt`.

```
HUNYUAN_DIR=C:/Users/<you>/.cache/hy3dgen/tencent/Hunyuan3D-2.1
```

A Hugging Face *cache* snapshot (`models--tencent--Hunyuan3D-2.1/snapshots/...`) won't work: it is made of links. If nothing is found, the worker downloads into `HUNYUAN_DIR` (default `models/Hunyuan3D-2.1`).

### TRELLIS.2 setup

1. In `.env`: `ENGINE=trellis2`, then `docker compose pull` (a separate image).
2. Its image encoder, Meta's DINOv3, is gated. Once: open https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m , request access (wait for approval), create a **read** token at https://huggingface.co/settings/tokens and add `HF_TOKEN=hf_...` to `.env`.
3. First start downloads about 9 GB into `models/trellis2` (or `TRELLIS_DIR`): only the shape models, not the texture ones.

Already have the weights? Arrange them in one folder and set `TRELLIS_DIR` to it:

```
TRELLIS_DIR/
  TRELLIS.2-4B/                     pipeline.json, ckpts/ (from microsoft/TRELLIS.2-4B)
  TRELLIS-image-large/ckpts/        ss_dec_conv3d_16l8_fp16.json + .safetensors
  dinov3-vitl16-pretrain-lvd1689m/  config.json, model.safetensors (Hugging Face format)
```

ComfyUI's TRELLIS.2 files (`trellis_2_int8_convrot.safetensors`, `dino_v3_L_naf_fp32.safetensors`) are a different, quantised format and can't be used here.

TRELLIS.2 is officially tested on 24 GB GPUs. The worker runs it in low-VRAM mode (models move to system RAM between steps), so 16 GB cards may work at **Detail 1024**; if you run out of memory, use **512**.

## Try it: the test page

```sh
docker compose --profile ui up ui
```

Open http://localhost:7860, upload an image, set the height, press **Generate**. The STL appears in the 3D view with a download link. Change the **seed** for a different result from the same image. `Ctrl+C` stops it.

Stop the `worker` service first if it is running (`docker compose stop worker`): two copies of the model don't fit in 16 GB. To compare engines, change `ENGINE` in `.env` and start the page again.

## Try it: command line

Put images in the `data` folder next to `docker-compose.yml`, then:

```sh
docker compose run --rm worker generate /data/photo.png --height-mm 150
```

The STL lands next to the image (`data/photo.stl`), and the command prints timings, face count and size. `--help` lists every option (seed, steps, detail, background removal, flat back/bottom).

## Run as a worker for a website

1. Copy `.env.example` to `.env` and fill in `SITE_URL`, `WORKER_SECRET` (the same secret the website has) and `WORKER_NAME` (any name, e.g. `pc-1`).
2. Start it:

   ```sh
   docker compose up -d worker
   ```

It restarts by itself after a reboot (as long as Docker Desktop starts with Windows). Logs: `docker compose logs -f worker`. Stop: `docker compose stop worker`.

The worker only makes outbound HTTPS calls: no ports to open, no database keys on this machine. The website side is described in [PROTOCOL.md](PROTOCOL.md).

## Update

```sh
git pull
docker compose pull
docker compose up -d worker
```

Model weights stay in `models/` (or your `HUNYUAN_DIR`) and are not downloaded again.

## Input tips

- A PNG with a **transparent background** skips background removal (`auto` mode) and gives the cleanest edges.
- For photos of people: face the camera, even light, upper body, no more than two people.
- The output is one solid colour: no textures.

## Output

Binary STL in millimetres, Z up, standing on Z = 0, front facing −Y (the default front view in most slicers). The back and bottom are cut flat so it stands and prints without supports under the base. Always check the model in your slicer before printing.

## Development

The geometry, protocol and worker loop run without a GPU:

```sh
docker run --rm -v "$PWD":/app -w /app python:3.10-slim sh -c "pip install -r requirements-dev.txt && pytest -q"
```

| Path | |
|---|---|
| `img2mesh/printable.py` | Orientation, flat back/bottom, scaling. Pure geometry, tested. |
| `img2mesh/pipeline.py` | Shared options, background removal, engine base and STL step |
| `img2mesh/engines/` | `hunyuan.py`, `trellis2.py`: image → raw mesh (GPU) |
| `img2mesh/protocol.py`, `serve.py` | Pull protocol v1 client and the worker loop, tested against a fake site |
| `img2mesh/ui.py` | The test page (Gradio) |

---

## ภาษาไทย: เริ่มใช้งานบน Windows

1. อัปเดต driver การ์ดจอ NVIDIA เป็นตัวล่าสุด แล้วติดตั้ง **Docker Desktop** (ใช้ค่าเริ่มต้น WSL 2)
2. เปิด PowerShell แล้วรัน:
   ```
   git clone https://github.com/Parunzz/img2mesh-worker.git
   cd img2mesh-worker
   docker compose pull
   ```
   เลือกโมเดลด้วย `ENGINE=` ใน `.env`: `hunyuan` (ค่าเริ่มต้น) หรือ `trellis2` (หน้าคนละเอียดกว่า ต้องใส่ `HF_TOKEN` ครั้งแรก ดูหัวข้อ TRELLIS.2 setup ด้านบน) เปลี่ยนแล้วรัน `docker compose pull` ใหม่
   ถ้ามี Hunyuan3D-2.1 อยู่ในเครื่องแล้ว ไม่ต้องโหลดใหม่: ก๊อป `.env.example` เป็น `.env` แล้วใส่
   - ไฟล์จาก ComfyUI (`hunyuan_3d_v2.1.safetensors`): `COMFYUI_CHECKPOINTS=C:/AI/ComfyUI/models/checkpoints`
   - หรือแบบของ Tencent: `HUNYUAN_DIR=` เป็นโฟลเดอร์ที่มี `hunyuan3d-dit-v2-1` อยู่ข้างใน
   (ใช้ `/` ไม่ใช่ `\`)
3. **ลองบนหน้าเว็บทดสอบ**: `docker compose --profile ui up ui` แล้วเปิด http://localhost:7860 อัปโหลดรูป ตั้งความสูง (mm) กด Generate ครั้งแรกจะโหลดไฟล์โมเดลหลาย GB รอสักพัก
4. **ลองแบบคำสั่ง**: วางรูปในโฟลเดอร์ `data` แล้วรัน `docker compose run --rm worker generate /data/photo.png --height-mm 150` จะได้ `data/photo.stl`
5. **ต่อกับเว็บไซต์** (หลังเว็บพร้อม): ก๊อป `.env.example` เป็น `.env` ใส่ `SITE_URL`, `WORKER_SECRET`, `WORKER_NAME` แล้วรัน `docker compose up -d worker` ปิดเครื่องเปิดใหม่ worker จะเริ่มเองถ้า Docker Desktop เปิดพร้อม Windows

ถ้าผลไม่สวย ลองเปลี่ยน seed หรือใช้รูป PNG ที่ตัดพื้นหลังแล้ว
