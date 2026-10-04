"""Turn a raw generated mesh into something a 3D printer can stand up.

Pure geometry: no GPU, no model, so it is tested on any machine.

Steps: orient (raw up/front -> printer Z-up, front facing -Y), cut a flat back
and a flat bottom, then scale uniformly so the height is exactly `height_mm`
and rest it on Z=0, centred on X/Y.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

AXES = {
    "+x": (1, 0, 0), "-x": (-1, 0, 0),
    "+y": (0, 1, 0), "-y": (0, -1, 0),
    "+z": (0, 0, 1), "-z": (0, 0, -1),
}


@dataclass(frozen=True)
class PrintOptions:
    height_mm: float
    # Share of the model's depth cut off the back to leave a flat back (0 = no cut).
    flat_back: float = 0.04
    # Share of the model's height cut off the bottom to leave a flat base (0 = no cut).
    flat_bottom: float = 0.02
    # Which raw axes point up and towards the viewer. Hunyuan3D outputs Y-up.
    up: str = "+y"
    front: str = "+z"

    def validate(self) -> None:
        if not self.height_mm > 0:
            raise ValueError("height_mm must be greater than 0")
        for name in ("flat_back", "flat_bottom"):
            value = getattr(self, name)
            if not 0 <= value < 0.5:
                raise ValueError(f"{name} must be between 0 and 0.5")
        if self.up not in AXES or self.front not in AXES:
            raise ValueError(f"up and front must be one of {', '.join(AXES)}")
        if abs(np.dot(AXES[self.up], AXES[self.front])) != 0:
            raise ValueError("up and front must be different, perpendicular axes")


def orientation(up: str, front: str) -> np.ndarray:
    """4x4 rotation taking raw `up` to +Z and raw `front` to -Y (slicer front)."""
    u = np.array(AXES[up], dtype=float)
    f = np.array(AXES[front], dtype=float)
    raw = np.column_stack([np.cross(u, f), f, u])  # right, front, up
    target = np.column_stack([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    matrix = np.eye(4)
    matrix[:3, :3] = target @ raw.T
    return matrix


def to_gltf_axes(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Printer axes (Z up, front -Y) -> glTF axes (Y up, front +Z), for previews.

    glTF loaders handle viewer handedness themselves; raw STL in some viewers
    (Babylon.js, used by Gradio) shows mirrored.
    """
    mesh = mesh.copy()
    mesh.apply_transform(orientation("+y", "+z").T)
    return mesh


def _keep_box(mesh: trimesh.Trimesh, lower: np.ndarray, upper: np.ndarray) -> trimesh.Trimesh:
    """The part of `mesh` inside an axis-aligned box, closed where it was cut."""
    box = trimesh.creation.box(bounds=[lower, upper])
    if mesh.is_volume:
        try:
            result = trimesh.boolean.intersection([mesh, box], engine="manifold")
            if len(result.faces):
                return result
        except Exception:  # noqa: BLE001 - fall back to plane slicing below
            pass
    # Not a clean volume (common for generated meshes): slice plane by plane and cap.
    for axis in range(3):
        for sign, bound in ((1, lower), (-1, upper)):
            normal = np.zeros(3)
            normal[axis] = sign
            mesh = mesh.slice_plane(plane_origin=bound, plane_normal=normal, cap=True)
    return mesh


def make_printable(mesh: trimesh.Trimesh, options: PrintOptions) -> trimesh.Trimesh:
    options.validate()
    mesh = mesh.copy()
    mesh.merge_vertices()
    mesh.apply_transform(orientation(options.up, options.front))

    lo, hi = mesh.bounds
    size = hi - lo
    lower = lo - 1.0  # margins so the box never coincides with an untouched face
    upper = hi + 1.0
    if options.flat_back:
        upper[1] = hi[1] - size[1] * options.flat_back  # back faces +Y
    if options.flat_bottom:
        lower[2] = lo[2] + size[2] * options.flat_bottom
    if options.flat_back or options.flat_bottom:
        mesh = _keep_box(mesh, lower, upper)

    lo, hi = mesh.bounds
    height = hi[2] - lo[2]
    if height <= 0:
        raise ValueError("the mesh has no height left after cutting")
    mesh.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    mesh.apply_scale(options.height_mm / height)
    trimesh.repair.fix_normals(mesh)
    return mesh
