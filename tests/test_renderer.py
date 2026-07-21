"""Tests for StructureRenderer's cell wireframe (off-screen PyVista)."""

import numpy as np
import pytest

pytest.importorskip("pyvista")
pytest.importorskip("ase")

import pyvista as pv  # noqa: E402
from ase.build import bulk  # noqa: E402

from crystalline.core.structure import Structure  # noqa: E402
from crystalline.viz.renderer import StructureRenderer, _bonded_pairs  # noqa: E402
from crystalline.viz.render_settings import RenderSettings  # noqa: E402


def _outline_extent(renderer):
    lo = np.array(renderer._cell_actor.GetBounds()).reshape(3, 2)
    return lo[:, 1] - lo[:, 0]


def test_reference_cell_outlines_original_not_supercell():
    base = bulk("NaCl", "rocksalt", a=5.64)
    supercell = Structure.from_ase(base.repeat((2, 2, 1)))
    renderer = StructureRenderer(pv.Plotter(off_screen=True))

    # Default: the wireframe follows the (super)cell that owns the atoms.
    renderer.set_reference_cell(None)
    renderer.set_structure(supercell)
    supercell_extent = _outline_extent(renderer)

    # With a reference cell, the wireframe shrinks to the original unit cell
    # while all supercell atoms remain on screen.
    renderer.set_reference_cell(np.asarray(base.cell))
    renderer.set_structure(supercell)
    unit_extent = _outline_extent(renderer)

    assert renderer.atom_count == 4 * len(base)  # 2×2×1 supercell atoms still shown
    assert np.all(unit_extent < supercell_extent - 1e-6)  # smaller outline


def test_lattice_vectors_drawn_for_periodic_only():
    renderer = StructureRenderer(pv.Plotter(off_screen=True))

    periodic = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64))
    renderer.set_structure(periodic)
    # 3 arrow actors + 1 label actor for the a/b/c gizmo (plus atoms/bonds/cell)
    assert _count_actors(renderer) >= 3

    mol = Structure.empty()
    mol.add_atom("O", [0.0, 0.0, 0.0])
    mol.add_atom("H", [0.96, 0.0, 0.0])
    before = _count_actors(renderer)
    renderer.set_structure(mol)  # non-periodic: no lattice arrows, no crash
    assert _count_actors(renderer) < before


def _count_actors(renderer):
    return len(list(renderer.plotter.renderer.actors))


def test_settings_toggle_bonds_cell_and_polyhedra():
    nacl = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64, cubic=True))
    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    renderer.set_structure(nacl)
    assert renderer._bond_actor is not None
    assert renderer._polyhedra_actor is None  # off by default

    renderer.set_settings(RenderSettings(show_bonds=False, show_cell=False))
    assert renderer._bond_actor is None and renderer._cell_actor is None

    renderer.set_settings(RenderSettings(show_polyhedra=True))
    assert renderer._polyhedra_actor is not None  # octahedra drawn


def test_scene_settings_and_atom_labels():
    nacl = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64, cubic=True))
    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    renderer.set_structure(nacl)
    assert renderer._label_actor is None  # labels off by default

    renderer.set_settings(
        RenderSettings(
            show_atom_labels=True,
            bond_color="#ff0000",
            background_color="black",
            parallel_projection=True,
            show_orientation_axes=True,
        )
    )
    assert renderer._label_actor is not None  # element labels drawn
    assert renderer.plotter.camera.GetParallelProjection() == 1  # orthographic applied


def test_atom_labels_suppressed_for_large_cells():
    big = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64).repeat((6, 6, 6)))  # 432 atoms
    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    renderer.set_structure(big)
    renderer.set_settings(RenderSettings(show_atom_labels=True))
    assert renderer.atom_count > 400
    assert renderer._label_actor is None  # too many atoms to label legibly


def test_atom_colour_overrides_apply_per_element():
    from crystalline.viz.renderer import _hex_to_rgb

    assert list(_hex_to_rgb("#ff8800")) == [255, 136, 0]
    assert list(_hex_to_rgb("#f80")) == [255, 136, 0]  # short form

    nacl = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64, cubic=True))
    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    renderer.set_structure(nacl)
    renderer.set_settings(RenderSettings(atom_colors=((11, "#ff0000"),)))  # Na -> red
    rgb = renderer._rgb_for(np.array([11, 17, 11]))
    assert list(rgb[0]) == [255, 0, 0] and list(rgb[2]) == [255, 0, 0]  # both Na red
    assert list(rgb[1]) != [255, 0, 0]  # Cl keeps its Jmol colour


def test_selected_group_moves_together_on_drag():
    # A non-periodic cluster (no periodic images to confuse the group).
    s = Structure.empty()
    for k in range(6):
        s.add_atom("C", [k * 1.6, 0.0, 0.0])
    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    renderer.set_structure(s)

    # With no selection, only the grabbed atom's (empty) image set moves with it.
    assert renderer.active_move_group(2) == []

    # Selecting {2,3,4} and grabbing 3 drags the whole selection as one piece.
    renderer.highlight([2, 3, 4])
    assert renderer.active_move_group(3) == [2, 4]
    assert renderer.active_move_group(0) == []  # grabbing outside the selection

    renderer.preview_atom_position(3, np.array([3 * 1.6, 0.0, 2.0]))  # shift +2 in z
    for i in (2, 3, 4):
        assert np.isclose(renderer.rendered_atom_position(i)[2], 2.0)
    for i in (0, 1, 5):
        assert np.isclose(renderer.rendered_atom_position(i)[2], 0.0)  # untouched


def test_bond_colour_modes_all_draw():
    nacl = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64, cubic=True))
    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    renderer.set_structure(nacl)
    for mode in ("solid", "split", "gradient"):
        renderer.set_settings(RenderSettings(bond_color_mode=mode))
        assert renderer._bond_actor is not None, f"{mode} bonds should draw"


def test_split_and_gradient_use_the_two_chosen_colours():
    # A single hetero bond so the tube carries both endpoint colours.
    s = Structure.empty()
    s.add_atom("C", [-0.7, 0.0, 0.0])
    s.add_atom("O", [0.7, 0.0, 0.0])
    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    for mode in ("split", "gradient"):
        mesh = renderer._colored_bond_mesh(
            s.positions, s.numbers, 0.18, 1.15, mode
        )
        assert mesh is not None
        colours = {tuple(row) for row in mesh["rgb"]}
        # both picked colours (default #888888 and #4c72b0) are present on the tube
        assert (0x88, 0x88, 0x88) in colours
        assert (0x4c, 0x72, 0xb0) in colours


def test_view_up_vector_is_orthonormal_to_direction():
    from crystalline.ui.viewport import Viewport

    cell = np.array([[4.0, 0.0, 0.0], [1.0, 3.5, 0.0], [0.5, 0.4, 5.0]])
    for axis in range(3):
        direction = cell[axis] / np.linalg.norm(cell[axis])
        up = Viewport._up_for(direction, cell, axis)
        assert abs(np.dot(up, direction)) < 1e-9  # perpendicular to the view direction
        assert abs(np.linalg.norm(up) - 1.0) < 1e-9  # unit length
    # No cell → a world axis not parallel to the view direction.
    up = Viewport._up_for(np.array([0.0, 0.0, 1.0]), None, 2)
    assert abs(np.dot(up, [0.0, 0.0, 1.0])) < 1e-9


def test_bonds_drop_cation_cation_pairs():
    # Two Na (both cations) close together must NOT bond; Na-Cl must.
    na2 = Structure.empty()
    na2.add_atom("Na", [0, 0, 0])
    na2.add_atom("Na", [2.5, 0, 0])
    i, j = _bonded_pairs(na2.positions, na2.numbers, 1.15)
    assert len(i) == 0  # cation-cation excluded

    nacl = Structure.empty()
    nacl.add_atom("Na", [0, 0, 0])
    nacl.add_atom("Cl", [2.5, 0, 0])
    i, j = _bonded_pairs(nacl.positions, nacl.numbers, 1.15)
    assert len(i) == 1  # cation-anion kept


def test_periodic_images_move_together_on_preview():
    # A cell with an atom and its exact lattice image, a distinct same-element
    # atom, and a different element sitting on a lattice point.
    s = Structure.empty()
    s.set_cell(np.diag([5.0, 5.0, 5.0]), periodic=True)
    s.add_atom("Na", [0.0, 0.0, 0.0])  # 0
    s.add_atom("Na", [5.0, 0.0, 0.0])  # 1: periodic image of 0 (a-translation)
    s.add_atom("Na", [2.0, 2.0, 2.0])  # 2: a distinct Na, not an image
    s.add_atom("Cl", [0.0, 5.0, 0.0])  # 3: on a lattice point but different element
    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    renderer.set_structure(s)

    assert renderer.periodic_image_indices(0) == [1]
    assert renderer.periodic_image_indices(2) == []  # no image of the interior atom

    # Previewing a drag of atom 0 shifts its image 1 by the same vector; the
    # unrelated atoms stay put.
    renderer.preview_atom_position(0, np.array([0.5, 0.0, 0.0]))
    assert np.allclose(renderer.rendered_atom_position(0), [0.5, 0.0, 0.0])
    assert np.allclose(renderer.rendered_atom_position(1), [5.5, 0.0, 0.0])
    assert np.allclose(renderer.rendered_atom_position(2), [2.0, 2.0, 2.0])
    assert np.allclose(renderer.rendered_atom_position(3), [0.0, 5.0, 0.0])


def test_periodic_images_span_a_supercell_via_reference_cell():
    base = bulk("NaCl", "rocksalt", a=5.64)
    supercell = Structure.from_ase(base.repeat((2, 2, 2)))  # 16 atoms
    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    # The reference cell (original unit cell) defines the periodicity, so each
    # site has one image per supercell tile.
    renderer.set_reference_cell(np.asarray(base.cell))
    renderer.set_structure(supercell)

    images = renderer.periodic_image_indices(0)
    assert len(images) == 7  # 2×2×2 tiles → 8 copies of the site, minus itself
    assert all(supercell.numbers[i] == supercell.numbers[0] for i in images)
