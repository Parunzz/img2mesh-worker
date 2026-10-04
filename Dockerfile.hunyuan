# img2mesh worker: Hunyuan3D 2.1 shape generation -> printable STL.
# Torch cu128 wheels carry their own CUDA libraries, so a plain Python base is
# enough; the GPU driver comes from the host (Docker Desktop / NVIDIA toolkit).
FROM python:3.10-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# CUDA 12.8 build: required for RTX 50xx (Blackwell, sm_120), fine for older cards.
# A current pip is needed: older ones reject some PyTorch-index wheels over name case.
RUN pip install --upgrade pip \
 && pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Only the shape package of Hunyuan3D-2.1, at a pinned commit (no texture stage).
ARG HUNYUAN_COMMIT=82920d643c0dc2f7bfd7255f45f62d386edfe60c
RUN git init -q /tmp/hy \
 && git -C /tmp/hy fetch -q --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git "$HUNYUAN_COMMIT" \
 && git -C /tmp/hy checkout -q FETCH_HEAD \
 && cp -r /tmp/hy/hy3dshape/hy3dshape /opt/hy3dshape \
 && rm -rf /tmp/hy

# Weights and caches live on the /models volume, downloaded on first start.
ENV PYTHONPATH=/opt:/app \
    HY3DGEN_MODELS=/models/hy3dgen \
    HF_HOME=/models/huggingface \
    U2NET_HOME=/models/u2net \
    GRADIO_ANALYTICS_ENABLED=False

WORKDIR /app
COPY LICENSE LICENSE-HUNYUAN3D-2.1.txt NOTICE /app/
COPY img2mesh /app/img2mesh

ENTRYPOINT ["python", "-m", "img2mesh"]
CMD ["serve"]
