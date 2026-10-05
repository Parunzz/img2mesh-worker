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
    # Which raw axes point up and towards the viewer. None = the engine's own
    # (Hunyuan: +y/+z, TRELLIS.2: +z/-y); "+y"/"+z" when used without an engine.
    up: str | None = None
    front: str | None = None

    def validate(self) -> None:
        if not self.height_mm > 0:
            raise ValueError("height_mm must be greater than 0")
        for name in ("flat_back", "flat_bottom"):
            value = getattr(self, name)
            if not 0 <= value < 0.5:
                raise ValueError(f"{name} must be between 0 and 0.5")
        up, front = self.axes()
        if up not in AXES or front not in AXES:
            raise ValueError(f"up and front must be one of {', '.join(AXES)}")
        if abs(np.dot(AXES[up], AXES[front])) != 0:
            raise ValueError("up and front must be different, perpendicular axes")

    def axes(self) -> tuple[str, str]:
        return self.up or "+y", self.front or "+z"


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


# A mesh with holes is rebuilt in a voxel grid this many cells along its longest
# side (about 0.3 mm at 150 mm, TRELLIS.2's own 512 resolution).
# ponytail: one fixed grid; raise it (memory grows with the cube) for Detail 1024.
VOXELS = 512
# Cracks up to about twice this many voxels wide are closed.
CLOSE_VOXELS = 2
# Straight tunnels into the model through a hole, up to about twice this wide, are filled.
TUNNEL_VOXELS = 4
SMOOTH = 1.0  # voxels
SURFACE_LEVEL = 0.84  # standard normal CDF at 1 / SMOOTH


def repair(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """A closed (watertight, outward-facing) version of a generated mesh.

    Watertight meshes (Hunyuan's usually are) pass through unchanged. Others
    (TRELLIS.2's have holes, cracks and inside-out patches, which show as dark
    areas and confuse slicers) are rebuilt as a solid: the surface is drawn
    into a voxel grid, small cracks are closed, the inside is filled, and the
    outside surface is traced again with marching cubes.
    """
    mesh = mesh.copy()
    mesh.merge_vertices()
    if mesh.is_watertight:
        return mesh
    return _simplify(_rebuild(mesh), len(mesh.faces))


def _rebuild(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    from scipy import ndimage
    from skimage.measure import marching_cubes

    pitch = mesh.extents.max() / VOXELS
    pad = CLOSE_VOXELS + 2
    origin = mesh.bounds[0] - pad * pitch
    shape = np.ceil(mesh.extents / pitch).astype(int) + 2 * pad + 1
    # About 8 samples per voxel face of surface, plus every vertex.
    count = int(min(mesh.area / pitch**2 * 8, 30_000_000))
    points = np.vstack([trimesh.sample.sample_surface(mesh, count, seed=0)[0], mesh.vertices])
    surface = np.zeros(shape, dtype=bool)
    surface[tuple(((points - origin) / pitch).astype(int).T)] = True
    surface = ndimage.binary_dilation(surface, iterations=CLOSE_VOXELS)
    # Outside = can see out of the grid along an axis without crossing the surface.
    # A flood fill would leak through any hole and leave a hollow shell; this only
    # lets a straight tunnel in, and tunnels narrower than the opening are dropped.
    outside = np.zeros(shape, dtype=bool)
    for axis in range(3):
        for flip in (False, True):
            run = np.flip(surface, axis) if flip else surface
            seen = ~np.logical_or.accumulate(run, axis=axis)
            outside |= np.flip(seen, axis) if flip else seen
    outside = ndimage.binary_opening(outside, iterations=TUNNEL_VOXELS, border_value=1)
    labels, _ = ndimage.label(outside)
    edge = np.unique(np.concatenate([labels[[0, -1]].ravel(), labels[:, [0, -1]].ravel(), labels[:, :, [0, -1]].ravel()]))
    solid = ~np.isin(labels, edge[edge > 0])
    # Drop loose specks (stray samples, floating bits).
    labels, count = ndimage.label(solid)
    if count > 1:
        sizes = np.bincount(labels.ravel())[1:]
        solid = np.isin(labels, 1 + np.flatnonzero(sizes >= sizes.max() * 1e-3))
    # Erode one voxel less than dilated; blur away the voxel steps, then trace the
    # surface one voxel in (a blurred flat side reads Phi(1/SMOOTH) one voxel inside).
    solid = ndimage.binary_erosion(solid, iterations=CLOSE_VOXELS - 1)
    field = ndimage.gaussian_filter(solid.astype(np.float32), SMOOTH)
    vertices, faces, _, _ = marching_cubes(field, SURFACE_LEVEL)
    out = trimesh.Trimesh(vertices * pitch + origin, faces)
    if out.volume < 0:
        out.invert()
    return out


def _simplify(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """Cut the rebuilt mesh to about `target_faces`, keeping it closed, and smooth
    away the voxel steps. Left as is without pymeshlab (tests)."""
    try:
        import pymeshlab
    except ImportError:
        return mesh
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64), face_matrix=np.asarray(mesh.faces, dtype=np.int32)))
    ms.apply_coord_taubin_smoothing(stepsmoothnum=10)
    if len(mesh.faces) > target_faces:
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces, preservetopology=True, preservenormal=True)
    out = trimesh.Trimesh(ms.current_mesh().vertex_matrix(), ms.current_mesh().face_matrix(), process=True)
    trimesh.repair.fix_normals(out)
    return out


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
    mesh = repair(mesh)
    mesh.apply_transform(orientation(*options.axes()))

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
