"""Tests for StructureRenderer's cell wireframe (off-screen PyVista)."""

import numpy as np
import pytest

pytest.importorskip("pyvista")
pytest.importorskip("ase")

import pyvista as pv  # noqa: E402
from ase.build import bulk  # noqa: E402

from crystalline.core.structure import Structure  # noqa: E402
from crystalline.viz.renderer import StructureRenderer, _bonded_pairs, _cell_edges  # noqa: E402
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


def test_cell_edges_scale_with_periodic_dimension():
    a, b, c = np.eye(3)
    assert _cell_edges([a]).n_lines == 1  # 1D polymer: a single segment
    assert _cell_edges([a, b]).n_lines == 4  # 2D slab: a parallelogram
    assert _cell_edges([a, b, c]).n_lines == 12  # 3D crystal: the full box


def test_slab_cell_box_is_flat_not_500_angstroms():
    """A 2D slab must outline its in-plane cell, not a box stretched along the
    formal 500 Å vacuum vector CRYSTAL writes for the aperiodic direction."""
    from ase.build import fcc111

    slab = fcc111("Pt", size=(2, 2, 3), vacuum=0.0)
    cell = np.asarray(slab.get_cell(), dtype=float)
    cell[2] = [0.0, 0.0, 500.0]  # CRYSTAL's aperiodic placeholder
    slab.set_cell(cell)
    slab.set_pbc((True, True, False))

    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    renderer.set_structure(Structure.from_ase(slab))

    extent = _outline_extent(renderer)
    assert extent[2] < 1.0  # the box has no depth along the non-periodic axis
    assert extent[0] > 1.0 and extent[1] > 1.0  # but spans the periodic plane


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
    assert renderer._polyhedra_actor is not None  # polyhedra on by default

    renderer.set_settings(
        RenderSettings(show_bonds=False, show_cell=False, show_polyhedra=False)
    )
    assert renderer._bond_actor is None and renderer._cell_actor is None
    assert renderer._polyhedra_actor is None

    renderer.set_settings(RenderSettings(show_polyhedra=True))
    assert renderer._polyhedra_actor is not None  # octahedra drawn


def test_polyhedron_outline_shows_real_edges_only():
    """A cubic 8-coordination hull outlines its 12 edges — the triangulation's
    6 coplanar face diagonals must not show (they do unless the hull's simplices
    are consistently wound)."""
    from crystalline.viz.renderer import _POLYHEDRA_EDGE_ANGLE, _hull_mesh

    cube = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
    mesh = _hull_mesh([(20, cube)], {})
    edges = mesh.extract_feature_edges(
        feature_angle=_POLYHEDRA_EDGE_ANGLE,
        boundary_edges=False,
        non_manifold_edges=False,
        manifold_edges=False,
    )
    assert edges.n_cells == 12


def test_edge_sharing_polyhedra_keep_their_shared_edge():
    """Neighbouring polyhedra share ligand atoms; welding those points would fuse
    the two solids and drop the shared edge out of the outline."""
    from crystalline.viz.renderer import _POLYHEDRA_EDGE_ANGLE, _hull_mesh

    octa = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float
    )
    shifted = octa + np.array([2.0, 0.0, 0.0])  # shares the [1,0,0] vertex
    mesh = _hull_mesh([(11, octa), (11, shifted)], {})
    edges = mesh.extract_feature_edges(
        feature_angle=_POLYHEDRA_EDGE_ANGLE,
        boundary_edges=False,
        non_manifold_edges=False,
        manifold_edges=False,
    )
    assert edges.n_cells == 24  # 12 per octahedron, none lost to the shared vertex


def test_polyhedra_get_an_outline_actor_and_a_translucent_default():
    nacl = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64, cubic=True))
    assert RenderSettings().polyhedra_opacity == 0.3  # see through to the atoms inside

    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    renderer.set_structure(nacl)
    assert renderer._polyhedra_edge_actor is not None  # polyhedra (and outline) on by default
    assert renderer._polyhedra_edge_actor.GetMapper().GetInput().GetNumberOfCells() > 0

    renderer.set_settings(RenderSettings(show_polyhedra=False))
    assert renderer._polyhedra_edge_actor is None  # outline goes with the polyhedra


def test_measurement_annotations_are_drawn_and_survive_a_rebuild():
    from crystalline.core import measure as measure_mod

    water = Structure.empty()
    water.add_atom("O", [0.0, 0.0, 0.0])
    water.add_atom("H", [0.96, 0.0, 0.0])
    water.add_atom("H", [-0.24, 0.93, 0.0])

    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    renderer.set_structure(water)
    assert renderer._annotation_actors == []  # nothing measured yet

    positions, symbols = water.positions, water.symbols
    renderer.set_annotations(
        [
            measure_mod.measure(positions, symbols, [0, 1]),        # distance
            measure_mod.measure(positions, symbols, [1, 0, 2]),     # angle
            measure_mod.measure_plane(positions, symbols, [0, 1, 2]),  # plane
        ]
    )
    drawn = len(renderer._annotation_actors)
    assert drawn >= 4  # two paths, a plane patch, and the labels

    # An edit or a settings change clears the plotter — annotations must come back.
    renderer.set_settings(RenderSettings(show_bonds=False))
    assert len(renderer._annotation_actors) == drawn

    renderer.set_annotations([])
    assert renderer._annotation_actors == []


def test_plane_annotation_has_no_floating_label():
    """A plane patch carries no 'rms …' label — only the patch actor is drawn."""
    from crystalline.core import measure as measure_mod

    water = Structure.empty()
    water.add_atom("O", [0.0, 0.0, 0.0])
    water.add_atom("H", [0.96, 0.0, 0.0])
    water.add_atom("H", [-0.24, 0.93, 0.0])
    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    renderer.set_structure(water)

    renderer.set_annotations([measure_mod.measure_plane(water.positions, water.symbols, [0, 1, 2])])
    assert len(renderer._annotation_actors) == 1  # just the patch, no label actor


def test_measurement_colours_come_from_settings():
    from crystalline.core import measure as measure_mod

    water = Structure.empty()
    water.add_atom("O", [0.0, 0.0, 0.0])
    water.add_atom("H", [0.96, 0.0, 0.0])
    renderer = StructureRenderer(
        pv.Plotter(off_screen=True), RenderSettings(measure_line_color="#ff0000")
    )
    renderer.set_structure(water)
    renderer.set_annotations([measure_mod.measure(water.positions, water.symbols, [0, 1])])

    line_actor = renderer._annotation_actors[0]
    assert np.allclose(line_actor.GetProperty().GetColor(), (1.0, 0.0, 0.0), atol=1e-3)


def test_per_item_measurement_colour_overrides_the_group_default():
    import dataclasses

    from crystalline.core import measure as measure_mod

    water = Structure.empty()
    water.add_atom("O", [0.0, 0.0, 0.0])
    water.add_atom("H", [0.96, 0.0, 0.0])
    renderer = StructureRenderer(
        pv.Plotter(off_screen=True), RenderSettings(measure_line_color="#ff0000")
    )
    renderer.set_structure(water)

    dist = measure_mod.measure(water.positions, water.symbols, [0, 1])
    recoloured = dataclasses.replace(dist, color="#00ff00")  # this item is green
    renderer.set_annotations([recoloured])

    colour = renderer._annotation_actors[0].GetProperty().GetColor()
    assert np.allclose(colour, (0.0, 1.0, 0.0), atol=1e-3)  # the item's colour, not the red default


def test_hydrogen_bonds_follow_a_phonon_animation():
    """The H-bond overlay must move with the atoms during animation (via
    ``update_positions``), not stay pinned to the equilibrium geometry."""
    dimer = Structure.empty()
    for sym, pos in (
        ("O", [0, 0, 0]), ("H", [0.97, 0, 0]), ("H", [-0.3, 0.9, 0]),
        ("O", [2.8, 0, 0]), ("H", [3.77, 0.2, 0]), ("H", [2.5, -0.9, 0]),
    ):
        dimer.add_atom(sym, pos)
    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    renderer.set_structure(dimer)
    assert renderer._hbond_actor is not None  # the O–H···O contact is drawn
    before = np.array(renderer._hbond_actor.GetBounds())

    moved = dimer.positions.copy()
    moved[3:] += [0.6, 0.0, 0.0]  # push the acceptor water away
    renderer.update_positions(moved)
    after = np.array(renderer._hbond_actor.GetBounds())
    assert not np.allclose(before, after)  # the dashed bond followed the atoms

    far = dimer.positions.copy()
    far[3:] += [3.0, 0.0, 0.0]  # far enough that it's no longer a hydrogen bond
    renderer.update_positions(far)
    assert renderer._hbond_actor is None


def test_annotations_never_become_pick_targets():
    """Picking must still hit atoms, not a measurement drawn over them."""
    from crystalline.core import measure as measure_mod

    water = Structure.empty()
    water.add_atom("O", [0.0, 0.0, 0.0])
    water.add_atom("H", [0.96, 0.0, 0.0])

    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    renderer.set_structure(water)
    renderer.set_annotations([measure_mod.measure(water.positions, water.symbols, [0, 1])])
    assert renderer._annotation_actors
    for actor in renderer._annotation_actors:
        assert actor.GetPickable() == 0


def test_polyhedra_follow_the_atoms_during_animation():
    """A phonon animation moves atoms via update_positions; the coordination
    polyhedra (and their outline) have to travel with their ligands."""
    from crystalline.core.cells import CellView, as_view, complete_boundary, tile_supercell

    src = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64, cubic=True))
    analysis, _ = tile_supercell(as_view(src, CellView.CRYSTALLOGRAPHIC), (1, 1, 1), None)
    packed, _ = complete_boundary(analysis, None)

    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    renderer.set_settings(RenderSettings(show_polyhedra=True))
    renderer.set_structure(packed, bond_structure=analysis)
    assert renderer._polyhedra_follow is not None  # every vertex matched an atom
    faces_before = renderer._polyhedra_mesh_obj.points.copy()
    edges_before = renderer._polyhedra_edge_mesh.points.copy()

    # A rigid shift of every atom must shift every hull vertex by exactly that.
    shift = np.array([0.13, -0.07, 0.21])
    renderer.update_positions(packed.positions + shift)
    assert np.allclose(renderer._polyhedra_mesh_obj.points - faces_before, shift)
    assert np.allclose(renderer._polyhedra_edge_mesh.points - edges_before, shift)

    # A per-atom displacement lands each vertex on its own displaced atom.
    rng = np.random.default_rng(0)
    displaced = packed.positions + rng.normal(scale=0.05, size=packed.positions.shape)
    renderer.update_positions(displaced)
    index, offset = renderer._polyhedra_follow
    assert np.allclose(renderer._polyhedra_mesh_obj.points, displaced[index] + offset)


def test_polyhedra_are_drawn_on_boundary_completed_images():
    """The displayed cell is boundary-completed, so a corner cation appears at
    several cell positions; each of those must be coordinated, not just the one
    atom the (pre-packing) analysis cell knows about."""
    from crystalline.core.cells import CellView, as_view, complete_boundary, tile_supercell

    src = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64, cubic=True))
    analysis, _ = tile_supercell(as_view(src, CellView.CRYSTALLOGRAPHIC), (1, 1, 1), None)
    packed, _ = complete_boundary(analysis, None)
    assert len(packed) > len(analysis)  # genuinely packed with image atoms

    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    renderer.set_settings(RenderSettings(show_polyhedra=True))
    renderer.set_structure(packed, bond_structure=analysis)

    from crystalline.core.bonds import connectivity

    analysed = connectivity(analysis, RenderSettings().polyhedra_min_vertices)
    centres = {int(z) for z, _c, _l in analysed.polyhedra}
    shown_centres = sum(int((packed.numbers == z).sum()) for z in centres)
    assert shown_centres > len(analysed.polyhedra)  # the bug: images were left bare

    # every shown centre contributes a hull, so the mesh spans the packed cell
    mesh_extent = np.array(renderer._polyhedra_actor.GetBounds()).reshape(3, 2)
    atoms_extent = np.array([packed.positions.min(axis=0), packed.positions.max(axis=0)]).T
    assert np.all(mesh_extent[:, 0] <= atoms_extent[:, 0] + 1e-6)
    assert np.all(mesh_extent[:, 1] >= atoms_extent[:, 1] - 1e-6)


def test_scene_settings_and_atom_labels():
    nacl = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64, cubic=True))
    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    renderer.set_structure(nacl)
    assert renderer._label_actor is None  # labels off by default

    renderer.set_settings(
        RenderSettings(
            show_atom_labels=True,
            background_color="black",
            parallel_projection=True,
            show_orientation_axes=True,
        )
    )
    assert renderer._label_actor is not None  # element labels drawn
    assert renderer.plotter.camera.GetParallelProjection() == 1  # orthographic applied


def test_default_projection_is_parallel():
    # New structures render orthographic (parallel) by default.
    assert RenderSettings().parallel_projection is True
    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    renderer.set_structure(Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64)))
    assert renderer.plotter.camera.GetParallelProjection() == 1


def test_refresh_preserves_zoom_under_parallel_projection():
    """Editing (e.g. deleting an atom) rebuilds the scene; the user's zoom must
    survive. Under parallel projection zoom is the camera's parallel scale, which
    ``camera_position`` doesn't carry — so a naive save/restore let the auto-fit
    zoom back out on every edit."""
    renderer = StructureRenderer(pv.Plotter(off_screen=True))  # parallel by default
    structure = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64).repeat((2, 2, 2)))
    renderer.set_structure(structure)

    renderer.plotter.camera.Zoom(2.5)  # user zooms in
    zoomed = renderer.plotter.camera.GetParallelScale()

    structure.remove_atoms([0])  # an edit -> refresh() rebuilds the scene
    renderer.refresh()

    assert np.isclose(renderer.plotter.camera.GetParallelScale(), zoomed)


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


def test_bonds_draw_with_a_single_fixed_colour():
    nacl = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64, cubic=True))
    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    renderer.set_structure(nacl)
    assert renderer._bond_actor is not None  # bonds still drawn, just no colour options


def test_hydrogen_bonds_drawn_by_default_and_toggle_off():
    from ase import Atoms

    dimer = Structure.from_ase(Atoms(
        "OH2OH2",
        positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0],
                   [2.85, 0, 0], [3.81, 0.2, 0], [2.61, -0.93, 0]],
        pbc=False,
    ))
    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    assert RenderSettings().show_hydrogen_bonds is True  # on by default
    renderer.set_structure(dimer)
    assert renderer._hbond_actor is not None  # the O–H···O contact is drawn

    renderer.set_settings(RenderSettings(show_hydrogen_bonds=False))
    assert renderer._hbond_actor is None


def test_hydrogen_bonds_follow_the_atoms_during_animation():
    """Animation moves atoms via ``update_positions`` (no full rebuild); the
    hydrogen bonds must move with them and vanish when the contact breaks."""
    from ase import Atoms

    dimer = Structure.from_ase(Atoms(
        "OH2OH2",
        positions=[[0, 0, 0], [0.97, 0, 0], [-0.3, 0.9, 0],
                   [2.8, 0, 0], [3.77, 0.2, 0], [2.5, -0.9, 0]],
        pbc=False,
    ))
    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    renderer.set_structure(dimer)
    xmax_before = renderer._hbond_actor.GetBounds()[1]

    moved = dimer.positions.copy()
    moved[3:] += [0.6, 0.0, 0.0]  # push the acceptor water out along +x
    renderer.update_positions(moved)
    assert renderer._hbond_actor.GetBounds()[1] > xmax_before  # the bond stretched with it

    far = dimer.positions.copy()
    far[3:] += [3.0, 0.0, 0.0]  # break the contact entirely
    renderer.update_positions(far)
    assert renderer._hbond_actor is None


def test_dashed_lines_breaks_a_segment_into_pieces():
    from crystalline.viz.renderer import _dashed_lines

    seg = np.array([[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]])  # one 3 Å segment
    solid = _dashed_lines(seg, dash=3.0, gap=0.0)
    dashed = _dashed_lines(seg, dash=0.3, gap=0.2)
    assert dashed.n_lines > solid.n_lines  # dashing yields many short line cells
    assert _dashed_lines(np.empty((0, 2, 3)), 0.3, 0.2) is None  # nothing to draw


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
