"""See nvdiffrast/__init__.py: a stub; img2mesh never renders or bakes textures."""


def __getattr__(name):
    raise RuntimeError(f"nvdiffrast.torch.{name} is not available: img2mesh generates shapes only (see docker/stubs)")
