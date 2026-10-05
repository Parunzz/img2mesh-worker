import numpy as np
import pytest
import trimesh

from img2mesh.printable import PrintOptions, make_printable, orientation


def raw_bust():
    """A stand-in for Hunyuan output: Y-up, facing +Z, centred near the origin, ~2 units tall."""
    body = trimesh.creation.capsule(height=1.2, radius=0.4)  # along Z
    body.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))  # now along Y
    nose = trimesh.creation.icosphere(radius=0.1).apply_translation([0, 0.5, 0.45])  # sticks out to +Z (front)
    return trimesh.util.concatenate([body, nose])


def test_orientation_maps_up_to_z_and_front_to_minus_y():
    m = orientation("+y", "+z")[:3, :3]
    assert np.allclose(m @ [0, 1, 0], [0, 0, 1])
    assert np.allclose(m @ [0, 0, 1], [0, -1, 0])
    assert np.isclose(np.linalg.det(m), 1)  # a rotation, never a mirror


@pytest.mark.parametrize("up,front", [("+y", "+z"), ("+z", "-y"), ("-x", "+y")])
def test_every_valid_axis_pair_is_a_rotation(up, front):
    assert np.isclose(np.linalg.det(orientation(up, front)[:3, :3]), 1)


def test_scales_to_exact_height_and_rests_on_the_bed():
    out = make_printable(raw_bust(), PrintOptions(height_mm=150))
    lo, hi = out.bounds
    assert np.isclose(hi[2] - lo[2], 150)
    assert np.isclose(lo[2], 0)
    assert np.allclose([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2], 0, atol=1e-6)


def test_cuts_leave_flat_back_and_bottom():
    out = make_printable(raw_bust(), PrintOptions(height_mm=100, flat_back=0.1, flat_bottom=0.05))
    lo, hi = out.bounds
    # Many vertices sit exactly on the cut planes: those are the flat faces.
    on_back = np.isclose(out.vertices[:, 1], hi[1], atol=1e-6).sum()
    on_bottom = np.isclose(out.vertices[:, 2], lo[2], atol=1e-6).sum()
    assert on_back > 10 and on_bottom > 10
    assert out.is_watertight


def test_front_detail_survives_the_back_cut():
    out = make_printable(raw_bust(), PrintOptions(height_mm=100, flat_back=0.2))
    # The nose (raw +Z, 3/4 up) is the frontmost point; printed, front is -Y.
    frontmost = out.vertices[out.vertices[:, 1].argmin()]
    assert 60 < frontmost[2] < 90


def test_no_cut_keeps_shape():
    raw = raw_bust()
    out = make_printable(raw, PrintOptions(height_mm=50, flat_back=0, flat_bottom=0))
    assert len(out.faces) == len(raw.faces)


@pytest.mark.parametrize("kwargs", [
    {"height_mm": 0},
    {"height_mm": 10, "flat_back": 0.6},
    {"height_mm": 10, "up": "+y", "front": "-y"},
    {"height_mm": 10, "up": "up"},
])
def test_rejects_bad_options(kwargs):
    with pytest.raises(ValueError):
        PrintOptions(**kwargs).validate()


def test_non_watertight_mesh_is_still_cut():
    raw = raw_bust()
    raw.update_faces(np.arange(len(raw.faces)) != 0)  # punch a hole: no longer a volume
    assert not raw.is_volume
    out = make_printable(raw, PrintOptions(height_mm=80, flat_back=0.1, flat_bottom=0.05))
    assert np.isclose(out.extents[2], 80)


def test_gltf_preview_axes():
    from img2mesh.printable import to_gltf_axes

    out = make_printable(raw_bust(), PrintOptions(height_mm=100))
    preview = to_gltf_axes(out)
    # Height goes back to +Y, and the nose (frontmost) points to +Z, like the raw model.
    assert np.isclose(preview.extents[1], 100)
    nose = preview.vertices[preview.vertices[:, 2].argmax()]
    assert 60 < nose[1] < 90
    assert np.isclose(np.linalg.det(orientation("+y", "+z")[:3, :3].T), 1)  # still no mirror


def holed_bust():
    raw = raw_bust()
    raw.update_faces(np.arange(len(raw.faces)) >= 40)  # tear a 40-triangle hole
    raw.remove_unreferenced_vertices()
    assert not raw.is_watertight
    return raw


def test_repair_leaves_watertight_mesh_alone():
    from img2mesh.printable import repair

    closed = trimesh.creation.icosphere()
    assert len(repair(closed).faces) == len(closed.faces)


def test_holed_mesh_still_prints_at_height():
    out = make_printable(holed_bust(), PrintOptions(height_mm=120, flat_back=0.1, flat_bottom=0.05))
    assert np.isclose(out.extents[2], 120)


def test_repair_closes_a_small_hole_into_a_solid(monkeypatch):
    import img2mesh.printable as printable
    from img2mesh.printable import repair

    monkeypatch.setattr(printable, "VOXELS", 64)  # keep the test fast

    sphere = trimesh.creation.icosphere(subdivisions=4)
    sphere.update_faces(np.arange(len(sphere.faces)) >= 6)  # a small hole
    sphere.remove_unreferenced_vertices()
    out = repair(sphere)
    assert out.is_watertight and out.is_winding_consistent
    assert np.isclose(out.volume, 4 / 3 * np.pi, rtol=0.05)


def test_repair_closes_inside_out_patches_and_a_big_hole(monkeypatch):
    import img2mesh.printable as printable
    from img2mesh.printable import repair

    monkeypatch.setattr(printable, "VOXELS", 64)

    sphere = trimesh.creation.icosphere(subdivisions=4)
    sphere.update_faces(sphere.triangles_center[:, 2] < 0.5)  # open top
    sphere.remove_unreferenced_vertices()
    flipped = sphere.triangles_center[:, 0] > 0.3
    sphere.faces = np.where(flipped[:, None], sphere.faces[:, ::-1], sphere.faces)
    out = repair(sphere)
    assert out.is_watertight and out.volume > 0
