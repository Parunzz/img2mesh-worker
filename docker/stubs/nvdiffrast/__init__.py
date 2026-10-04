"""Placeholder for NVIDIA nvdiffrast, which is NOT included.

TRELLIS.2's o_voxel imports nvdiffrast at load time for texture baking and
rendering. img2mesh only generates shapes and never calls it, and nvdiffrast's
licence (NVIDIA Source Code License) allows non-commercial use only, so this
image ships this stub instead. Any real use fails loudly.
"""
