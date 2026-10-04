# img2mesh-worker

Photo → printable STL, on your own NVIDIA GPU, in Docker.

`image → background removal → Hunyuan3D 2.1 (shape) → clean-up → flat back and bottom, scaled to a height in mm → STL`

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

The first run of any command downloads the model weights (several GB) into the `models` Docker volume. Later runs start in about a minute.

## Try it: the test page

```sh
docker compose --profile ui up ui
```

Open http://localhost:7860, upload an image, set the height, press **Generate**. The STL appears in the 3D view with a download link. Change the **seed** for a different result from the same image. `Ctrl+C` stops it.

Stop the `worker` service first if it is running (`docker compose stop worker`): two copies of the model don't fit in 16 GB.

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

Model weights stay in the volume and are not downloaded again.

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
| `img2mesh/pipeline.py` | Background removal + Hunyuan3D 2.1 + mesh clean-up (GPU) |
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
3. **ลองบนหน้าเว็บทดสอบ**: `docker compose --profile ui up ui` แล้วเปิด http://localhost:7860 อัปโหลดรูป ตั้งความสูง (mm) กด Generate ครั้งแรกจะโหลดไฟล์โมเดลหลาย GB รอสักพัก
4. **ลองแบบคำสั่ง**: วางรูปในโฟลเดอร์ `data` แล้วรัน `docker compose run --rm worker generate /data/photo.png --height-mm 150` จะได้ `data/photo.stl`
5. **ต่อกับเว็บไซต์** (หลังเว็บพร้อม): ก๊อป `.env.example` เป็น `.env` ใส่ `SITE_URL`, `WORKER_SECRET`, `WORKER_NAME` แล้วรัน `docker compose up -d worker` ปิดเครื่องเปิดใหม่ worker จะเริ่มเองถ้า Docker Desktop เปิดพร้อม Windows

ถ้าผลไม่สวย ลองเปลี่ยน seed หรือใช้รูป PNG ที่ตัดพื้นหลังแล้ว
