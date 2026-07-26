"""Draw a :class:`~crystalline.core.structure.Structure` into a PyVista plotter.

Design notes
------------
* **All atoms are drawn as a single glyphed mesh** (one VTK actor), not one
  actor per atom. VTK's per-frame cost scales with the number of *actors*, so a
  cell with thousands of atoms (a ZIF-8 supercell, say) went from interactive to
  tens of seconds — and eventually a crash — under the old one-actor-per-atom
  scheme. Glyphing collapses that to a single actor whose build/redraw is ~1000×
  faster at those sizes. Per-atom **picking** maps a pick's world position back
  to the nearest atom centre; **position updates** (drag / phonon animation)
  re-glyph the point cloud, which is cheap now that it is one mesh.
* Element colours and radii come from ``ase.data`` (Jmol colours, covalent
  radii) so they match community conventions.
* No Qt here: the plotter is supplied by the caller (a ``pyvistaqt`` interactor
  in the app, or an off-screen ``pyvista.Plotter`` in tests).
"""

from __future__ import annotations

import itertools
from functools import lru_cache
from typing import Optional

import numpy as np
import pyvista as pv
from ase.data import chemical_symbols, covalent_radii
from ase.data.colors import jmol_colors
from scipy.spatial import cKDTree

from crystalline.core.structure import Structure
from crystalline.viz.render_settings import RenderSettings

# Fraction of the covalent radius used for the drawn sphere (ball-and-stick).
# Bond radius/tolerance are per-view settings — see :class:`RenderSettings`.
_ATOM_SCALE = 0.5

# Covalent bonds are a single fixed grey (no per-bond colouring). Hydrogen bonds
# are drawn as thin dashed light-blue lines, VESTA-style.
_BOND_COLOR = "#888888"
_HBOND_COLOR = "#4aa3df"
_HBOND_LINE_WIDTH = 2.0
_HBOND_DASH = 0.28   # Å: dash length
_HBOND_GAP = 0.20    # Å: gap between dashes

# Measurement annotations (Geometry panel). Colours are per-view settings now
# (measure_point/line/plane_color); this warm accent is only the fallback default.
_ANNOTATION_COLOR = "#ff7f0e"
_ANNOTATION_LINE_RADIUS = 0.035
_ANNOTATION_POINT_RADIUS = 0.18
_ANNOTATION_FONT_SIZE = 13
_ANNOTATION_PLANE_OPACITY = 0.28
_ANNOTATION_PLANE_MARGIN = 1.25   # patch overhang beyond the fitted atoms
_ANNOTATION_PLANE_MIN_SIZE = 2.0  # Angstrom, so a tight plane is still visible
# Depth bias that lifts annotations in front of the atoms they measure.
_ANNOTATION_DEPTH_OFFSET = -66000.0

# Coordination-polyhedra outline: only edges where adjacent faces bend by more
# than this are real polyhedron edges (the rest are the hull's triangulation of
# a flat face), drawn this wide in this fraction of the face colour.
_POLYHEDRA_EDGE_ANGLE = 15.0
_POLYHEDRA_EDGE_WIDTH = 1.5
_POLYHEDRA_EDGE_SHADE = 0.55

# Material for the atom spheres. The specular highlight is what makes small
# on-screen atoms read as 3D balls rather than flat disks.
_ATOM_MATERIAL = dict(
    smooth_shading=True,
    specular=0.5,
    specular_power=15,
    ambient=0.3,
    diffuse=0.6,
)

# Above these atom counts, bonds are not recomputed on every live position update
# (animation frames / drag) — they refresh on commit instead — to avoid lag.
#
# Covalent bonds are only a KD-tree pass and one tube mesh, which measures well
# under a millisecond per thousand atoms, so the limit is generous: a large cell
# like a MOF must still show its bonds moving with the atoms during a phonon
# animation. Hydrogen bonds cost far more (a per-hydrogen Python scan), so their
# own limit stays low — unless the topology is frozen, in which case the scan
# runs once per animation and each frame is only an array gather.
_LIVE_BOND_MAX_ATOMS = 5000
_LIVE_HBOND_MAX_ATOMS = 400

# Above this atom count, per-atom element labels are suppressed: they'd be an
# unreadable, slow-to-render cloud of text on a large cell.
_ATOM_LABEL_MAX_ATOMS = 400

# Fractional tolerance for deciding two atoms are periodic images (their
# separation is a whole lattice translation). Boundary/supercell images are exact
# lattice shifts, so this can be tight.
_IMAGE_FRAC_TOL = 1e-3

# Lattice-vector indicator: small a/b/c arrows near the structure. Colours follow
# the common convention a=red, b=green, c=blue (as in VESTA).
_LATTICE_COLORS = ("#d62728", "#2ca02c", "#1f77b4")
_LATTICE_LABELS = ("a", "b", "c")
# Fixed arrow length (Å): the gizmo shows only the a/b/c *directions*, so it stays
# the same size regardless of the lattice parameters (editing a/b/c must not
# grow or shrink it). Comparable to a bond length, so it reads well beside atoms.
_LATTICE_ARROW_LENGTH = 2.0


def _sphere_resolution(n_atoms: int) -> int:
    """Fewer triangles per sphere for larger systems, to keep rendering fast."""
    if n_atoms <= 100:
        return 32
    if n_atoms <= 500:
        return 16
    return 8


class StructureRenderer:
    """Renders atoms, bonds and the unit cell for a structure."""

    def __init__(
        self, plotter: pv.BasePlotter, settings: Optional[RenderSettings] = None
    ) -> None:
        self.plotter = plotter
        self._settings = settings if settings is not None else RenderSettings()
        self._structure: Optional[Structure] = None
        # Single glyphed actor for every atom (see module docstring). The live
        # positions/numbers/radii back both re-glyphing and pick→index mapping.
        self._atom_actor = None
        self._positions: np.ndarray = np.empty((0, 3), dtype=float)
        self._numbers: np.ndarray = np.empty(0, dtype=int)
        self._radii: np.ndarray = np.empty(0, dtype=float)
        # Cached periodic-image group for the atom currently being dragged.
        self._drag_primary: Optional[int] = None
        self._drag_group: list = []
        self._bond_actor = None
        self._hbond_actor = None
        self._cell_actor = None
        self._polyhedra_actor = None
        self._polyhedra_edge_actor = None
        self._polyhedra_mesh_obj = None       # kept so animation can move its points
        self._polyhedra_edge_mesh = None
        self._polyhedra_follow = None         # (atom index, offset) per hull vertex
        self._polyhedra_edge_follow = None
        self._label_actor = None
        self._annotations: list = []          # measurements drawn over the structure
        self._annotation_actors: list = []
        self._bond_structure: Optional[Structure] = None  # clean cell for coordination
        # Geometry that decides *which* atoms are bonded while something moves the
        # atoms without changing the chemistry (a phonon animation). None means
        # connectivity is re-derived from the drawn frame, which is what editing wants.
        self._bond_reference: Optional[np.ndarray] = None
        # Connectivity derived from that reference, computed once and reused for
        # every frame of an animation instead of being rebuilt per frame.
        self._frozen_bond_pairs: Optional[tuple] = None
        self._frozen_hbond_pairs: Optional[np.ndarray] = None
        # Cache the (expensive) CrystalNN coordination analysis so a rebuild that
        # only changed display settings — a slider nudge, a colour, a toggle —
        # doesn't re-run it. Keyed on the analysed geometry + min-coordination.
        self._poly_cache_key = None
        self._poly_cache_found: Optional[list] = None
        self._highlight_actors: dict[int, object] = {}  # atom index -> halo actor
        # If set, the cell wireframe outlines this cell (the original unit cell)
        # instead of the structure's own cell — used when a supercell is shown.
        self._reference_cell: Optional[np.ndarray] = None

    # ── public API ──────────────────────────────────────────────────────
    @property
    def settings(self) -> RenderSettings:
        return self._settings

    def set_settings(self, settings: RenderSettings) -> None:
        """Apply new appearance settings and redraw."""
        self._settings = settings
        self._rebuild()

    def set_bond_reference(self, positions: Optional[np.ndarray]) -> None:
        """Fix the bond network to the geometry at ``positions`` (``None`` clears).

        A phonon animation moves atoms far off their sites without changing the
        chemistry, so connectivity must come from the equilibrium geometry rather
        than from each frame — otherwise a stretch that crosses the bond-length
        criterion makes bonds flicker, and the larger the amplitude the worse it
        gets. Editing leaves this ``None`` so moving an atom really does re-bond it.
        """
        self._bond_reference = None if positions is None else np.asarray(positions, dtype=float)
        self._frozen_bond_pairs = None
        self._frozen_hbond_pairs = None

    def set_reference_cell(self, cell: Optional[np.ndarray]) -> None:
        """Choose which cell the wireframe outlines (``None`` = the structure's own).

        Set to the original unit cell to keep its outline while a supercell of
        atoms is displayed. Takes effect on the next rebuild.
        """
        self._reference_cell = None if cell is None else np.asarray(cell, dtype=float)

    def set_structure(self, structure: Structure, bond_structure: Optional[Structure] = None) -> None:
        """Draw ``structure`` from scratch, replacing anything shown.

        ``bond_structure`` (optional) is a clean periodic cell used for the
        chemically-aware coordination analysis behind polyhedra — passed
        separately because the displayed structure may be a boundary-completed
        "packed" cell (with duplicate images) that a near-neighbour algorithm
        can't analyse.
        """
        self._structure = structure
        self._bond_structure = bond_structure
        self._rebuild()

    def refresh(self) -> None:
        """Redraw after the current structure was edited (add/remove atoms)."""
        self._rebuild()

    def update_positions(self, positions: np.ndarray) -> None:
        """Move all atoms without a full rebuild (re-glyphs the atom mesh).

        Used by animation: the atom *count* must be unchanged. Bonds follow the
        atoms for small systems (recomputed here); for large systems bonds are
        left in place during animation to keep it smooth.
        """
        if len(self._positions) != len(positions):
            raise ValueError("position count changed; call refresh() instead")
        self._positions = np.asarray(positions, dtype=float)
        self._reglyph_atoms()
        if len(self._positions) <= _LIVE_BOND_MAX_ATOMS:
            self._update_bonds(self._positions)
            # The hydrogen-bond scan is the expensive one; a frozen topology makes
            # it a gather, so only the unfrozen path needs the tighter limit.
            if self._frozen_hbond_pairs is not None or self._bond_reference is not None:
                self._update_hydrogen_bonds()
            elif len(self._positions) <= _LIVE_HBOND_MAX_ATOMS:
                self._update_hydrogen_bonds()
        self._update_polyhedra(self._positions)
        self.plotter.render()

    @property
    def atom_count(self) -> int:
        """Number of atoms currently drawn (for size-sanity checks)."""
        return len(self._positions)

    def pick_atom_index(self, actor, world_pos) -> Optional[int]:
        """Map a pick (its actor + world position) back to an atom index.

        Only the single atom-glyph actor is pickable, so a hit on it is always an
        atom; the picked surface point is resolved to the nearest atom centre.
        Returns ``None`` for a miss or a hit on any non-atom prop.
        """
        if actor is None or actor is not self._atom_actor or len(self._positions) == 0:
            return None
        _dist, index = cKDTree(self._positions).query(np.asarray(world_pos, dtype=float))
        return int(index)

    def atom_position(self, index: int) -> np.ndarray:
        """Current cartesian position of atom ``index`` (from the model)."""
        return np.asarray(self._structure.positions[index], dtype=float)

    def rendered_atom_position(self, index: int) -> np.ndarray:
        """Where atom ``index`` is *currently drawn* (moves live during a drag /
        animation, before the model is updated)."""
        return np.asarray(self._positions[index], dtype=float)

    def periodic_image_indices(self, index: int) -> list:
        """Indices of atoms that are periodic images of ``index``.

        Two atoms are periodic images when they are the same element and separated
        by a whole lattice translation of the reference cell (the original unit
        cell when a supercell is shown, otherwise the structure's own cell). These
        are the atoms that must move together with ``index`` during a drag so the
        periodicity is preserved. The dragged atom itself is not included.
        """
        if self._structure is None or len(self._positions) < 2:
            return []
        lattice = self._reference_cell if self._reference_cell is not None else self._structure.cell
        lattice = np.asarray(lattice, dtype=float)
        if np.allclose(lattice, 0.0):
            return []
        base = np.asarray(self._structure.positions, dtype=float)
        diff = base - base[index]
        try:  # fractional coordinates of each atom relative to `index`
            frac = np.linalg.solve(lattice.T, diff.T).T
        except np.linalg.LinAlgError:  # singular (degenerate) cell
            return []
        integral = np.all(np.abs(frac - np.rint(frac)) < _IMAGE_FRAC_TOL, axis=1)
        same_element = self._numbers == self._numbers[index]
        not_self = np.arange(len(base)) != index
        return list(np.nonzero(integral & same_element & not_self)[0])

    def active_move_group(self, index: int) -> list:
        """Indices that should move *with* ``index`` during a drag (excluding it).

        If ``index`` is part of a multi-atom selection (has a halo, and it isn't
        the only one), the whole selection moves together — each selected atom
        also carrying its own periodic images — so an imported/selected fragment
        drags as one piece. Otherwise only ``index``'s periodic images move.
        """
        selected = set(self._highlight_actors.keys())
        if index in selected and len(selected) > 1:
            group: set = set()
            for atom in selected:
                group.add(atom)
                group.update(self.periodic_image_indices(atom))
            group.discard(index)
            return sorted(group)
        return self.periodic_image_indices(index)

    def preview_atom_position(self, index: int, world_pos: np.ndarray) -> None:
        """Move an atom's sphere live (with its move group), not the model.

        Used during an interactive drag: the dragged sphere follows the cursor and
        every atom in its move group (periodic images, plus the rest of a
        multi-atom selection) moves by the same cartesian shift. The model is
        updated once on drop; bonds redraw with the ensuing ``refresh``. Does not
        call ``render`` — the caller does.
        """
        world_pos = np.asarray(world_pos, dtype=float)
        base = np.asarray(self._structure.positions, dtype=float)
        delta = world_pos - base[index]
        # The move group is fixed for the duration of a drag; cache it per atom.
        if index != self._drag_primary:
            self._drag_primary = index
            self._drag_group = self.active_move_group(index)
        for i in (index, *self._drag_group):
            self._positions[i] = base[i] + delta
            if i in self._highlight_actors:
                self._highlight_actors[i].SetPosition(*delta)
        self._reglyph_atoms()
        # keep the bonds attached to the dragged atoms (small systems only)
        if len(self._positions) <= _LIVE_BOND_MAX_ATOMS:
            self._update_bonds(self._positions)
            self._update_hydrogen_bonds()

    def highlight(self, indices) -> None:
        """Draw translucent halos around the selected atoms.

        ``indices`` may be a single atom index, an iterable of indices, or
        ``None`` / empty to clear the selection.
        """
        for actor in self._highlight_actors.values():
            self.plotter.remove_actor(actor, render=False)
        self._highlight_actors = {}

        if indices is None:
            wanted = []
        elif isinstance(indices, (int, np.integer)):
            wanted = [int(indices)]
        else:
            wanted = [int(i) for i in indices]

        if self._structure is not None:
            for index in wanted:
                if not 0 <= index < len(self._structure):
                    continue
                pos = self._structure.positions[index]
                r = _sphere_radius(self._structure.numbers[index], self._settings.atom_scale) * 1.6
                halo = pv.Sphere(radius=r, center=pos)
                actor = self.plotter.add_mesh(
                    halo, color="yellow", opacity=0.35, name=f"__highlight_{index}__"
                )
                # Halos must never intercept picks, or they would block grabbing
                # the selected atoms (and any atom their larger radius overlaps).
                actor.SetPickable(False)
                self._highlight_actors[index] = actor
        self.plotter.render()

    # ── internals ───────────────────────────────────────────────────────
    def _ensure_lights(self) -> None:
        """Re-add a default light kit if the renderer has no lights.

        ``Plotter.clear()`` strips lights; without this, spheres render as flat
        ambient-lit disks instead of shaded 3D balls.
        """
        if not self.plotter.renderer.lights:
            self.plotter.enable_lightkit()

    def _apply_scene_settings(self) -> None:
        """Apply background, projection and the orientation-axes marker.

        These are scene/camera properties (not per-object actors), so they're set
        on every rebuild and independently of whether a structure is loaded.
        Guarded because a bare/off-screen plotter may not support every hook.
        """
        settings = self._settings
        try:
            self.plotter.set_background(settings.background_color)
        except Exception:  # noqa: BLE001 - invalid colour string, etc.
            pass
        try:
            if settings.parallel_projection:
                self.plotter.enable_parallel_projection()
            else:
                self.plotter.disable_parallel_projection()
        except Exception:  # noqa: BLE001
            pass
        try:
            if settings.show_orientation_axes:
                self.plotter.add_axes()
            else:
                self.plotter.hide_axes()
        except Exception:  # noqa: BLE001
            pass

    def _rebuild(self) -> None:
        # Preserve the current view across the rebuild. clear() drops pyvista's
        # "camera set" flag, so the next add_mesh would auto-fit the camera and
        # zoom the view out on every edit (e.g. dropping a dragged atom). Framing
        # is the viewport's job; a redraw must leave the camera exactly as it was.
        saved_camera = self._current_camera()
        # NB: Plotter.clear() also removes the renderer's lights, which would
        # leave every sphere flat-shaded (ambient only). Restore them after.
        self.plotter.clear()
        self._ensure_lights()
        self._apply_scene_settings()
        self._atom_actor = None
        self._positions = np.empty((0, 3), dtype=float)
        self._numbers = np.empty(0, dtype=int)
        self._radii = np.empty(0, dtype=float)
        self._drag_primary = None
        self._drag_group = []
        self._bond_actor = None
        self._hbond_actor = None
        self._cell_actor = None
        self._polyhedra_actor = None
        self._polyhedra_edge_actor = None
        self._polyhedra_mesh_obj = None
        self._polyhedra_edge_mesh = None
        self._polyhedra_follow = None
        self._polyhedra_edge_follow = None
        self._label_actor = None
        self._annotation_actors = []  # plotter.clear() dropped them; _draw_annotations re-adds
        self._highlight_actors = {}
        if self._structure is None or len(self._structure) == 0:
            self._restore_camera(saved_camera)
            self.plotter.render()
            return

        struct = self._structure
        settings = self._settings
        self._positions = np.asarray(struct.positions, dtype=float)
        self._numbers = np.asarray(struct.numbers, dtype=int)
        self._radii = _sphere_radius(self._numbers, settings.atom_scale)
        self._draw_atoms()

        if settings.show_bonds:
            self._draw_bonds(self._positions, self._numbers)
        if settings.show_hydrogen_bonds:
            self._draw_hydrogen_bonds()
        if settings.show_polyhedra:
            self._draw_polyhedra()
        if settings.show_cell:
            self._draw_cell()
        if settings.show_lattice_vectors:
            self._draw_lattice_vectors()
        if settings.show_atom_labels:
            self._draw_atom_labels()
        self._draw_annotations()  # measurements survive a rebuild (plotter.clear())
        self._restore_camera(saved_camera)
        self.plotter.render()

    def _current_camera(self):
        """The current view — placement *and* zoom — or ``None`` if unreadable.

        ``camera_position`` alone omits the zoom under parallel (orthographic)
        projection: there the zoom is the camera's *parallel scale*, not its
        distance. Capturing only the placement would let pyvista's post-``clear()``
        auto-fit reset the zoom on every edit (visible as a zoom-out when deleting
        an atom). We snapshot the parallel scale (and the perspective view angle)
        too, so a redraw leaves the view exactly where the user left it.
        """
        try:
            camera = self.plotter.camera
            return (self.plotter.camera_position, camera.GetParallelScale(), camera.GetViewAngle())
        except Exception:  # noqa: BLE001 - bare/uninitialised plotter
            return None

    def _restore_camera(self, camera) -> None:
        """Put a previously-captured view back (a redraw must not move or rezoom it)."""
        if camera is None:
            return
        position, parallel_scale, view_angle = camera
        try:
            self.plotter.camera_position = position
            self.plotter.camera.SetParallelScale(parallel_scale)  # zoom under parallel projection
            self.plotter.camera.SetViewAngle(view_angle)
        except Exception:  # noqa: BLE001
            pass

    # ── atoms (single glyphed mesh) ─────────────────────────────────────
    def _draw_atoms(self) -> None:
        """Draw every atom as one glyphed mesh (a single VTK actor)."""
        self._atom_actor = self._build_atom_glyph_actor()

    def _build_atom_glyph_actor(self):
        """Build the atom glyph: a unit sphere replicated at every atom, scaled
        by covalent radius and coloured per element, as one actor."""
        resolution = _sphere_resolution(len(self._numbers))
        cloud = pv.PolyData(self._positions)
        cloud["radius"] = self._radii
        cloud["rgb"] = self._rgb_for(self._numbers)
        geom = pv.Sphere(radius=1.0, theta_resolution=resolution, phi_resolution=resolution)
        # scale each sphere by its atom's covalent radius; don't orient by normals
        glyph = cloud.glyph(geom=geom, scale="radius", orient=False)
        return self.plotter.add_mesh(
            glyph,
            scalars="rgb",
            rgb=True,
            opacity=self._settings.atom_opacity,
            render=False,
            **_ATOM_MATERIAL,
        )

    def _reglyph_atoms(self) -> None:
        """Rebuild the atom glyph in place after positions changed (no render)."""
        if self._atom_actor is not None:
            self.plotter.remove_actor(self._atom_actor, render=False)
            self._atom_actor = None
        if len(self._positions):
            self._atom_actor = self._build_atom_glyph_actor()

    def _rgb_for(self, numbers: np.ndarray) -> np.ndarray:
        """Per-atom uint8 RGB rows (Jmol colours with the settings' overrides applied)."""
        numbers = np.asarray(numbers, dtype=int)
        rgb = (jmol_colors[numbers] * 255).astype(np.uint8)
        for z, color in self._settings.atom_colors:
            mask = numbers == int(z)
            if mask.any():
                rgb[mask] = _hex_to_rgb(color)
        return rgb

    def _draw_bonds(self, positions: np.ndarray, numbers: np.ndarray) -> None:
        radius, tol = self._settings.bond_radius, self._settings.bond_tolerance
        mesh = _bond_mesh(positions, numbers, radius, tol, pairs=self._bond_pairs(numbers, tol))
        if mesh is None:
            return
        self._bond_actor = self.plotter.add_mesh(
            mesh, color=_BOND_COLOR, smooth_shading=True, render=False
        )
        self._bond_actor.SetPickable(False)  # only atoms are pick targets

    def _bond_pairs(self, numbers: np.ndarray, tolerance: float) -> Optional[tuple]:
        """Bonded pairs to draw, or ``None`` to derive them from the live frame.

        While a bond reference is set the pairs are computed from it once and
        cached, so an animation pays the neighbour search a single time.
        """
        if self._bond_reference is None or len(self._bond_reference) != len(self._positions):
            return None
        if self._frozen_bond_pairs is None:
            self._frozen_bond_pairs = _bonded_pairs(self._bond_reference, numbers, tolerance)
        return self._frozen_bond_pairs

    def _draw_hydrogen_bonds(self) -> None:
        """Draw D–H···A hydrogen bonds as thin dashed light-blue lines.

        Computed on the atoms as drawn (``self._positions``, no periodicity), like
        the covalent bonds: the displayed cell is already boundary-completed, so
        searching its periodic images too would draw each contact twice (once to a
        visible atom, once to an image out in space). Using the *drawn* positions
        also means the bonds follow a phonon animation, which moves those.
        """
        from crystalline.core.bonds import (
            hydrogen_bond_pairs,
            hydrogen_bond_segments,
            hydrogen_bonds_from_positions,
        )

        if self._bond_reference is not None and len(self._bond_reference) == len(self._positions):
            if self._frozen_hbond_pairs is None:
                self._frozen_hbond_pairs = hydrogen_bond_pairs(self._bond_reference, self._numbers)
            segments = hydrogen_bond_segments(self._positions, self._frozen_hbond_pairs)
        else:
            segments = hydrogen_bonds_from_positions(self._positions, self._numbers)
        if len(segments) == 0:
            return
        mesh = _dashed_lines(segments, _HBOND_DASH, _HBOND_GAP)
        if mesh is None:
            return
        self._hbond_actor = self.plotter.add_mesh(
            mesh, color=_HBOND_COLOR, line_width=_HBOND_LINE_WIDTH, render=False
        )
        self._hbond_actor.SetPickable(False)  # not a pick target

    def _update_hydrogen_bonds(self) -> None:
        """Redraw the hydrogen bonds for the current positions (no render here)."""
        if self._hbond_actor is not None:
            self.plotter.remove_actor(self._hbond_actor, render=False)
            self._hbond_actor = None
        if self._settings.show_hydrogen_bonds:
            self._draw_hydrogen_bonds()

    def _draw_atom_labels(self) -> None:
        """Label each atom with its element symbol (suppressed for large cells)."""
        if len(self._numbers) == 0 or len(self._numbers) > _ATOM_LABEL_MAX_ATOMS:
            return
        symbols = [chemical_symbols[int(z)] for z in self._numbers]
        self._label_actor = self.plotter.add_point_labels(
            np.asarray(self._positions, dtype=float),
            symbols,
            font_size=int(self._settings.atom_label_size),
            text_color="#202020",
            show_points=False,
            shape=None,
            always_visible=True,
            pickable=False,
            render=False,
        )
        self._label_actor.SetPickable(False)  # text must never intercept picks

    def _draw_polyhedra(self) -> None:
        """Draw coordination polyhedra (convex hull of each cation's ligands).

        VESTA-style. Coordination comes from pymatgen's ``CrystalNN`` on the
        clean unit cell (``bond_structure``) so ionic crystals get the right
        cation–anion polyhedra (Si→tetrahedra, Ca→6–8) with correct geometry
        across the cell boundary. Falls back to a periodic distance search if
        that's unavailable.
        """
        mesh = self._polyhedra_mesh()
        if mesh is None:
            return
        self._polyhedra_actor = self.plotter.add_mesh(
            mesh,
            scalars="colors",
            rgb=True,
            opacity=self._settings.polyhedra_opacity,
            smooth_shading=True,
            render=False,
        )
        self._polyhedra_actor.SetPickable(False)
        # Remember which atom each hull vertex rides on, so the polyhedra can
        # follow a phonon animation without re-running the coordination analysis.
        self._polyhedra_mesh_obj = mesh
        self._polyhedra_follow = _follow_atoms(mesh.points, self._positions, self._cell_or_none())
        self._draw_polyhedra_edges(mesh)

    def _draw_polyhedra_edges(self, mesh: "pv.PolyData") -> None:
        """Outline each polyhedron's edges so its shape reads at low opacity.

        The hull arrives triangulated, so drawing every triangle edge would
        criss-cross each flat face with diagonals; ``extract_feature_edges``
        keeps only the edges where the surface actually bends — the polyhedron's
        real edges. They are drawn opaque, in a darkened shade of the face
        colour, and unlit so the outline stays an even weight from every angle.
        """
        edges = mesh.extract_feature_edges(
            feature_angle=_POLYHEDRA_EDGE_ANGLE,
            boundary_edges=False,  # closed hulls have none
            non_manifold_edges=False,
            manifold_edges=False,
        )
        if edges is None or edges.n_points == 0:
            return
        if "colors" in edges.point_data:
            shaded = edges["colors"].astype(float) * _POLYHEDRA_EDGE_SHADE
            edges["colors"] = shaded.astype(np.uint8)
        self._polyhedra_edge_actor = self.plotter.add_mesh(
            edges,
            scalars="colors" if "colors" in edges.point_data else None,
            rgb="colors" in edges.point_data,
            line_width=_POLYHEDRA_EDGE_WIDTH,
            lighting=False,
            render=False,
        )
        self._polyhedra_edge_actor.SetPickable(False)
        self._polyhedra_edge_mesh = edges
        self._polyhedra_edge_follow = _follow_atoms(
            edges.points, self._positions, self._cell_or_none()
        )

    # ── measurement annotations (points / lines / planes) ───────────────
    def set_annotations(self, annotations) -> None:
        """Draw geometry measurements over the structure, replacing any shown.

        ``annotations`` are :class:`~crystalline.core.measure.Measurement`
        objects: a point becomes a marker, a distance/angle/dihedral becomes the
        polyline through its atoms, and a plane becomes a translucent patch.
        They are kept and redrawn on every rebuild, so they survive
        an edit or a settings change rather than blinking out.
        """
        self._annotations = list(annotations)
        self._clear_annotations()
        self._draw_annotations()
        self.plotter.render()

    def _clear_annotations(self) -> None:
        for actor in self._annotation_actors:
            self.plotter.remove_actor(actor, render=False)
        self._annotation_actors = []

    def _draw_annotations(self) -> None:
        """(Re)draw the stored measurements. Never raises into a redraw."""
        from crystalline.core.measure import DIHEDRAL, PLANE, POINT

        self._annotation_actors = []
        if not self._annotations:
            return
        labels: list = []
        label_points: list = []
        for item in self._annotations:
            points = np.asarray(item.points, dtype=float)
            if len(points) == 0:
                continue
            try:
                # A per-item colour (set in the Geometry panel) overrides the
                # type's default from settings.
                if item.kind == POINT:
                    self._add_annotation_actor(
                        pv.Sphere(radius=_ANNOTATION_POINT_RADIUS, center=points[0]),
                        color=item.color or self._settings.measure_point_color,
                    )
                elif item.kind == PLANE:
                    self._draw_annotation_plane(item, points)
                else:  # distance / angle / dihedral: the path through the atoms
                    self._add_annotation_actor(
                        _polyline_tube(points),
                        color=item.color or self._settings.measure_line_color,
                    )
            except Exception:  # noqa: BLE001 - a bad measurement must not kill the redraw
                continue
            if item.kind in (POINT, PLANE):
                labels.append("")  # points and planes carry no floating label
            else:
                labels.append(f"{item.value:.3f} {item.unit}".strip())
            label_points.append(_annotation_anchor(item.kind, points, DIHEDRAL))

        texts = [t for t in labels if t]
        if texts:
            anchors = np.asarray(
                [p for t, p in zip(labels, label_points) if t], dtype=float
            )
            actor = self.plotter.add_point_labels(
                anchors, texts, font_size=_ANNOTATION_FONT_SIZE, show_points=False,
                shape_opacity=0.6, always_visible=True, pickable=False, render=False,
            )
            actor.SetPickable(False)
            self._annotation_actors.append(actor)

    def _draw_annotation_plane(self, item, points: np.ndarray) -> None:
        """A translucent patch spanning the fitted atoms."""
        origin = np.asarray(item.origin, dtype=float)
        normal = np.asarray(item.normal, dtype=float)
        # Size the patch to the atoms it was fitted through, with a little margin.
        spread = float(np.linalg.norm(points - origin, axis=1).max())
        size = max(spread * 2.0 * _ANNOTATION_PLANE_MARGIN, _ANNOTATION_PLANE_MIN_SIZE)
        patch = pv.Plane(center=origin, direction=normal, i_size=size, j_size=size)
        self._add_annotation_actor(
            patch, color=item.color or self._settings.measure_plane_color,
            opacity=_ANNOTATION_PLANE_OPACITY,
            on_top=False,  # a plane reads as a slice *through* the structure
        )

    def _add_annotation_actor(
        self, mesh, color: str = None, opacity: float = 1.0, on_top: bool = True
    ) -> None:
        actor = self.plotter.add_mesh(
            mesh,
            color=color or _ANNOTATION_COLOR,
            opacity=opacity,
            smooth_shading=True,
            render=False,
        )
        actor.SetPickable(False)  # annotations are never a pick target
        if on_top:
            _draw_over_scene(actor)
        self._annotation_actors.append(actor)

    def _cell_or_none(self) -> Optional[np.ndarray]:
        """The displayed structure's lattice, or ``None`` if it has no usable one."""
        if self._structure is None:
            return None
        cell = np.asarray(self._structure.cell, dtype=float)
        if cell.shape != (3, 3) or abs(np.linalg.det(cell)) < 1e-8:
            return None
        return cell

    def _update_polyhedra(self, positions: np.ndarray) -> None:
        """Move the polyhedra (and their outline) with the atoms they sit on.

        Coordination is fixed for the duration of an animation, so the hulls just
        follow their ligands: each vertex is re-placed on its atom (plus the
        constant lattice offset of the image it belongs to). No re-hulling, no
        near-neighbour analysis — a per-frame array gather.
        """
        for mesh, follow in (
            (self._polyhedra_mesh_obj, self._polyhedra_follow),
            (self._polyhedra_edge_mesh, self._polyhedra_edge_follow),
        ):
            if mesh is None or follow is None:
                continue
            index, offset = follow
            mesh.points = positions[index] + offset

    def _coordination_polyhedra(self, analysis, min_vertices, connectivity) -> list:
        """The coordination polyhedra for ``analysis``, cached across rebuilds.

        CrystalNN is the interactive-redraw bottleneck, and it depends only on the
        analysed geometry and ``min_vertices`` — not on colours, opacity, camera or
        any other display setting. So we key a one-entry cache on the geometry and
        reuse the result whenever only presentation changed.
        """
        positions = np.asarray(analysis.positions, dtype=float)
        numbers = np.asarray(analysis.numbers, dtype=int)
        # bond_tolerance only matters to the distance fallback, but it's cheap to
        # include and keeps that path correct when the tolerance slider moves.
        key = (
            numbers.tobytes(),
            np.round(positions, 4).tobytes(),
            int(min_vertices),
            round(float(self._settings.bond_tolerance), 4),
        )
        if key == self._poly_cache_key:
            return self._poly_cache_found

        conn = connectivity(analysis, min_vertices)
        found = list(conn.polyhedra) if conn is not None else []
        if not found:  # periodic distance search on the clean cell (CrystalNN missing/empty)
            found = _fallback_polyhedra(
                analysis.to_ase(), self._settings.bond_tolerance, min_vertices
            )
        self._poly_cache_key = key
        self._poly_cache_found = found
        return found

    def _polyhedra_mesh(self) -> Optional["pv.PolyData"]:
        """Build the polyhedra mesh: CrystalNN on the clean cell, else a fallback.

        Both paths analyse the clean ``bond_structure`` (the periodic cell before
        boundary-completion), which a near-neighbour algorithm can handle — the
        packed cell's duplicate images would confuse it. The distance fallback
        runs whenever CrystalNN is unavailable, too slow, *or* found no polyhedra
        — previously an empty CrystalNN result drew nothing (the coordination
        showed up only on big supercells, where the atom count tipped it into the
        fallback).

        The analysed polyhedra are then replicated onto every *shown* centre, so
        the image atoms that boundary-completion adds are coordinated too rather
        than sitting bare next to their neighbours.
        """
        min_vertices = self._settings.polyhedra_min_vertices
        analysis = self._bond_structure if self._bond_structure is not None else self._structure

        from crystalline.core.bonds import connectivity, replicate_polyhedra

        found = self._coordination_polyhedra(analysis, min_vertices, connectivity)
        if not found:
            return None

        shown = replicate_polyhedra(found, analysis.cell, self._positions, self._numbers)
        overrides = dict(self._settings.atom_colors)  # tint polyhedra by the centre's colour
        return _hull_mesh(shown, overrides)

    def _update_bonds(self, positions: np.ndarray) -> None:
        """Replace the bond mesh for a new set of positions (no render here)."""
        if self._bond_actor is not None:
            self.plotter.remove_actor(self._bond_actor, render=False)
            self._bond_actor = None
        if self._settings.show_bonds:
            self._draw_bonds(positions, self._structure.numbers)

    def _draw_cell(self) -> None:
        if self._structure is None or not self._structure.is_periodic:
            return
        # Outline the reference (original) cell when one is set — e.g. a supercell
        # of atoms is shown but the lines should mark the original unit cell.
        cell = self._reference_cell if self._reference_cell is not None else self._structure.cell
        cell = np.asarray(cell, dtype=float)
        if np.allclose(cell, 0.0):
            return
        # Only the periodic vectors span the drawn cell — a slab/polymer's formal
        # 500 Å vacuum edge is not a real cell edge and must not be outlined.
        vectors = [cell[i] for i, p in enumerate(self._structure.pbc) if p]
        if not vectors:
            return
        edges = _cell_edges(vectors)
        self._cell_actor = self.plotter.add_mesh(edges, color="#3355aa", line_width=2)
        self._cell_actor.SetPickable(False)

    def _draw_lattice_vectors(self) -> None:
        """Draw small a/b/c arrows (red/green/blue) as a gizmo below the structure.

        Placed just outside the structure's lower corner — not at the cell origin
        — so the arrows don't lie on top of the cell edges.
        """
        if self._structure is None or not self._structure.is_periodic:
            return
        cell = self._reference_cell if self._reference_cell is not None else self._structure.cell
        cell = np.asarray(cell, dtype=float)
        if np.allclose(cell, 0.0):
            return
        # All three arrows share one fixed length, so the gizmo indicates only
        # orientation and never rescales when the lattice parameters change.
        arrow_length = _LATTICE_ARROW_LENGTH
        # Anchor the gizmo below the lowest atom, offset far enough that the
        # arrows clear the cell rather than overlapping its edges.
        pad = 1.3 * arrow_length
        base = self._structure.positions.min(axis=0) - pad
        tips = []
        labels = []
        # One arrow per periodic direction only: a slab shows a/b, a polymer just a
        # — never a spurious c arrow along CRYSTAL's formal 500 Å vacuum vector.
        for axis, p in enumerate(self._structure.pbc):
            if not p:
                continue
            vec = cell[axis]
            length = float(np.linalg.norm(vec))
            if length < 1e-6:
                continue
            arrow = pv.Arrow(start=base, direction=vec, scale=arrow_length)
            actor = self.plotter.add_mesh(arrow, color=_LATTICE_COLORS[axis], render=False)
            actor.SetPickable(False)  # gizmo, never a pick target
            tips.append(base + (vec / length) * arrow_length)
            labels.append(_LATTICE_LABELS[axis])
        if tips:
            labels = self.plotter.add_point_labels(
                np.asarray(tips, dtype=float),
                labels,
                font_size=14,
                show_points=False,
                shape=None,
                always_visible=True,
                pickable=False,
                render=False,
            )
            labels.SetPickable(False)


# ── free helpers ─────────────────────────────────────────────────────────
def _sphere_radius(z, scale: float = _ATOM_SCALE):
    """Drawn sphere radius for atomic number(s) ``z`` (scalar or array in → same out)."""
    return covalent_radii[z] * scale


def _element_color(z: int) -> list:
    return list(jmol_colors[z])


def _hex_to_rgb(color: str) -> np.ndarray:
    """Parse ``"#rrggbb"`` (or ``"#rgb"``) into a uint8 RGB triple."""
    text = color.lstrip("#")
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    return np.array([int(text[k : k + 2], 16) for k in (0, 2, 4)], dtype=np.uint8)


# Two atoms both below this Pauling electronegativity are treated as cations and
# not bonded to each other — stops covalent radii from spuriously bonding big
# cations (Ca–Ca, Ca–Si) in ionic crystals, while leaving C–C, C–O, Si–O, etc.
_CATION_ELECTRONEGATIVITY = 2.0


@lru_cache(maxsize=None)
def _electronegativity(z: int) -> float:
    try:
        from pymatgen.core import Element

        value = Element.from_Z(int(z)).X
        return float(value) if value and not np.isnan(value) else 0.0
    except Exception:  # noqa: BLE001 - pymatgen missing / element without EN
        return 0.0


def _bonded_pairs(positions: np.ndarray, numbers: np.ndarray, tolerance: float):
    """Return the (i, j) index arrays of bonded atom pairs (KD-tree neighbour search).

    Two atoms bond if their distance < ``tolerance * (r_i + r_j)`` AND they are
    not both cations. The KD-tree finds only pairs within the maximum possible
    bond length, avoiding the O(N^2) all-pairs loop that dominated load/redraw
    time for large structures.
    """
    positions = np.asarray(positions, dtype=float)
    if len(positions) < 2:
        return np.empty(0, int), np.empty(0, int)
    radii = covalent_radii[numbers]
    max_bond = tolerance * 2.0 * float(radii.max())
    candidate_pairs = cKDTree(positions).query_pairs(r=max_bond, output_type="ndarray")
    if len(candidate_pairs) == 0:
        return np.empty(0, int), np.empty(0, int)
    i, j = candidate_pairs[:, 0], candidate_pairs[:, 1]
    dists = np.linalg.norm(positions[i] - positions[j], axis=1)
    keep = dists < tolerance * (radii[i] + radii[j])
    i, j = i[keep], j[keep]
    if len(i) == 0:
        return i, j
    en = np.array([_electronegativity(int(z)) for z in numbers])
    both_cations = (en[i] < _CATION_ELECTRONEGATIVITY) & (en[j] < _CATION_ELECTRONEGATIVITY)
    return i[~both_cations], j[~both_cations]


def _bond_mesh(
    positions: np.ndarray,
    numbers: np.ndarray,
    radius: float,
    tolerance: float,
    pairs: Optional[tuple] = None,
) -> Optional[pv.PolyData]:
    """Bond tube mesh for the bonded pairs (radius/tolerance from settings).

    ``pairs`` supplies a ready-made connectivity — from the equilibrium geometry
    during an animation — while the tubes are always drawn between the live
    ``positions``. That keeps the bond network fixed through a vibration instead
    of letting bonds wink out whenever a stretch crosses the distance criterion.
    """
    positions = np.asarray(positions, dtype=float)
    i, j = _bonded_pairs(positions, numbers, tolerance) if pairs is None else pairs
    if len(i) == 0:
        return None
    pts = np.empty((2 * len(i), 3), dtype=float)
    pts[0::2] = positions[i]
    pts[1::2] = positions[j]
    lines = np.empty((len(i), 3), dtype=np.int64)
    lines[:, 0] = 2
    lines[:, 1] = np.arange(0, 2 * len(i), 2)
    lines[:, 2] = np.arange(1, 2 * len(i), 2)
    poly = pv.PolyData(pts)
    poly.lines = lines.ravel()
    return poly.tube(radius=radius)


def _dashed_lines(segments: np.ndarray, dash: float, gap: float) -> Optional[pv.PolyData]:
    """A line mesh that renders ``segments`` (``(M, 2, 3)``) as dashes.

    VTK's line stipple is unreliable across its OpenGL backends, so the dashes are
    real geometry: each segment is chopped into ``dash``-long pieces separated by
    ``gap`` (Å). Cheap here because hydrogen bonds are few.
    """
    pts: list = []
    lines: list = []
    idx = 0
    for start, end in np.asarray(segments, dtype=float):
        vec = end - start
        length = float(np.linalg.norm(vec))
        if length < 1e-6:
            continue
        direction = vec / length
        offset = 0.0
        while offset < length:
            a = start + direction * offset
            b = start + direction * min(offset + dash, length)
            pts.append(a)
            pts.append(b)
            lines.append([2, idx, idx + 1])
            idx += 2
            offset += dash + gap
    if not pts:
        return None
    poly = pv.PolyData(np.asarray(pts, dtype=float))
    poly.lines = np.hstack(lines)
    return poly


def _draw_over_scene(actor) -> None:
    """Make ``actor`` render in front of the structure instead of inside it.

    A measurement between two bonded atoms runs almost entirely *inside* their
    spheres, so an occluded line is invisible exactly when it matters. VTK's
    coincident-topology offset pulls the actor towards the camera in depth only
    — geometry and picking are untouched. Best-effort: if the VTK build doesn't
    expose the knobs, the annotation simply draws normally.
    """
    try:
        mapper = actor.GetMapper()
        mapper.SetResolveCoincidentTopologyToPolygonOffset()
        for setter in (
            "SetRelativeCoincidentTopologyLineOffsetParameters",
            "SetRelativeCoincidentTopologyPolygonOffsetParameters",
            "SetRelativeCoincidentTopologyPointOffsetParameter",
        ):
            method = getattr(mapper, setter, None)
            if method is None:
                continue
            try:
                method(_ANNOTATION_DEPTH_OFFSET, _ANNOTATION_DEPTH_OFFSET)
            except TypeError:  # the point variant takes a single value
                method(_ANNOTATION_DEPTH_OFFSET)
    except Exception:  # noqa: BLE001 - purely cosmetic; never break a redraw
        pass


def _polyline_tube(points: np.ndarray) -> pv.PolyData:
    """A thin tube along the path through ``points`` (2+ vertices)."""
    poly = pv.PolyData()
    poly.points = np.asarray(points, dtype=float)
    segments = len(points) - 1
    poly.lines = np.hstack([[2, k, k + 1] for k in range(segments)]).astype(np.int64)
    return poly.tube(radius=_ANNOTATION_LINE_RADIUS)


def _annotation_anchor(kind: str, points: np.ndarray, dihedral_kind: str) -> np.ndarray:
    """Where a measurement's value is written: mid-path, or at the angle's vertex."""
    if len(points) == 2:
        return points.mean(axis=0)
    if len(points) == 3:
        return points[1]  # the vertex of the angle
    if kind == dihedral_kind and len(points) == 4:
        return points[1:3].mean(axis=0)  # midpoint of the central bond
    return points.mean(axis=0)


def _follow_atoms(points, positions, cell: Optional[np.ndarray]):
    """Pin each mesh vertex to the atom it sits on: ``points == positions[i] + offset``.

    A polyhedron vertex *is* a ligand atom, but often a periodic image of one —
    coordination is analysed on the unit cell, and hulls are then replicated onto
    the shown images. Matching is therefore done modulo the lattice; the leftover
    offset is that image's lattice translation, which stays constant while the
    atoms vibrate. Positions are keyed by rounded (fractional) coordinates, so
    the whole map is one pass rather than a neighbour search.

    Returns ``(index, offset)`` arrays, or ``None`` if any vertex fails to match
    (the caller then simply leaves the polyhedra where they are).
    """
    points = np.asarray(points, dtype=float)
    positions = np.asarray(positions, dtype=float)
    if len(points) == 0 or len(positions) == 0:
        return None

    def key_of(rows: np.ndarray) -> np.ndarray:
        if cell is None:  # no lattice (a molecule): plain cartesian identity
            return np.round(rows, 4)
        fractional = rows @ np.linalg.inv(cell)
        return np.round(fractional % 1.0, 4) % 1.0  # 0.99996 -> 1.0 -> 0.0

    lookup = {tuple(k): i for i, k in enumerate(key_of(positions))}
    index = np.empty(len(points), dtype=int)
    for n, k in enumerate(key_of(points)):
        found = lookup.get(tuple(k))
        if found is None:
            return None  # an unexpected vertex: don't animate rather than distort
        index[n] = found
    return index, points - positions[index]


def _center_rgb(center_z: int, overrides: Optional[dict]) -> np.ndarray:
    """RGB (0–255 floats) for a polyhedron centred on ``center_z`` — override or Jmol."""
    if overrides and int(center_z) in overrides:
        return _hex_to_rgb(overrides[int(center_z)]).astype(float)
    return np.asarray(_element_color(int(center_z))) * 255


def _hull_mesh(polyhedra, overrides: Optional[dict] = None) -> Optional[pv.PolyData]:
    """Merge ``[(atomic_number, ligand_positions), …]`` into one coloured mesh.

    Each entry's ligands become a convex-hull polyhedron coloured by the central
    atom (honouring any ``overrides`` colour map); coplanar/degenerate sets are
    skipped. Returns ``None`` if nothing drawable.
    """
    from scipy.spatial import ConvexHull

    try:  # QhullError moved across scipy versions
        from scipy.spatial import QhullError
    except ImportError:  # pragma: no cover - older scipy
        from scipy.spatial.qhull import QhullError

    meshes = []
    for center_z, ligands in polyhedra:
        pts = np.asarray(ligands, dtype=float)
        if len(pts) < 4:
            continue
        try:
            hull = ConvexHull(pts)
        except (QhullError, ValueError):
            continue  # coplanar / too few independent points
        faces = np.hstack([[3, *tri] for tri in hull.simplices]).astype(np.int64)
        poly = pv.PolyData(pts, faces)
        color = np.tile(_center_rgb(center_z, overrides), (poly.n_points, 1))
        poly["colors"] = color.astype(np.uint8)
        meshes.append(poly)

    if not meshes:
        return None
    merged = meshes[0]
    for extra in meshes[1:]:
        # merge_points=False: neighbouring polyhedra share ligand atoms, and
        # welding those vertices would fuse two independent solids — the shared
        # edge turns non-manifold and drops out of the outline, and shading
        # bleeds from one polyhedron into the next.
        merged = merged.merge(extra, merge_points=False)
    # Qhull doesn't wind its simplices consistently, so adjacent triangles can
    # come out with opposing normals — which shades the facets unevenly and, in
    # ``extract_feature_edges``, makes every edge look like a fold (a cube's 6
    # coplanar face diagonals then get outlined alongside its 12 real edges).
    return merged.compute_normals(consistent_normals=True, auto_orient_normals=True, inplace=False)


def _fallback_polyhedra(atoms, tolerance: float, min_vertices: int) -> list:
    """Distance-based coordination polyhedra (used when CrystalNN isn't available).

    Uses ``ase.neighbor_list`` (periodic, so ligands cross the cell boundary
    correctly) for each atom's neighbours, then keeps only the chemically
    sensible ones: a polyhedron is drawn around a **cation** centre using its
    **anion** ligands — neighbours more electronegative than the centre. Without
    that filter, covalent radii spuriously bond big cations to each other (a Ca
    "coordinating" nearby Ca/Si), so the hull swallows other cations; the filter
    reproduces the cation–anion polyhedra CrystalNN would give.

    Returns ``[(atomic_number, centre, ligand_positions), …]`` — the same shape
    :class:`~crystalline.core.bonds.Connectivity` uses, so both paths can be
    replicated onto the shown centres and hulled by the same code.
    """
    from collections import defaultdict

    if len(atoms) == 0:
        return []
    try:
        from ase.neighborlist import natural_cutoffs, neighbor_list

        i, j, offset = neighbor_list("ijD", atoms, natural_cutoffs(atoms, mult=tolerance))
    except Exception:  # noqa: BLE001 - bonding hiccup must not break rendering
        return []
    if len(i) == 0:
        return []

    positions = atoms.get_positions()
    numbers = atoms.get_atomic_numbers()
    en = {int(z): _electronegativity(int(z)) for z in set(int(z) for z in numbers)}

    # centre index -> list of anion ligand world positions (more electronegative)
    ligands: dict = defaultdict(list)
    for a, b, d in zip(i, j, offset):
        if en[int(numbers[b])] > en[int(numbers[a])]:  # ligand is the anion of the pair
            ligands[int(a)].append(positions[a] + d)  # neighbour image = positions[a] + d

    polyhedra = []
    for center, pts in ligands.items():
        if len(pts) < min_vertices:
            continue
        polyhedra.append(
            (int(numbers[center]), np.asarray(positions[center], dtype=float),
             np.asarray(pts, dtype=float))
        )
    return polyhedra


def _cell_edges(vectors) -> pv.PolyData:
    """Edges of the cell spanned by ``vectors`` (1–3 lattice vectors).

    Only the *periodic* lattice vectors are passed, so a bulk crystal (3 vectors)
    draws the full 12-edge box, a slab (2 vectors) a 4-edge parallelogram and a
    polymer (1 vector) a single segment — never the parallelepiped stretched along
    CRYSTAL's formal 500 Å vacuum direction.
    """
    vectors = [np.asarray(v, dtype=float) for v in vectors]
    n = len(vectors)
    # Every corner is a 0/1 combination of the spanning vectors; two corners share
    # an edge when their combinations differ in exactly one vector.
    combos = list(itertools.product((0, 1), repeat=n))
    corners = [sum((coeff[k] * vectors[k] for k in range(n)), np.zeros(3)) for coeff in combos]
    pts: list = []
    lines: list = []
    edge = 0
    for i, ci in enumerate(combos):
        for j in range(i + 1, len(combos)):
            if sum(a != b for a, b in zip(ci, combos[j])) == 1:
                pts.append(corners[i])
                pts.append(corners[j])
                lines.append([2, 2 * edge, 2 * edge + 1])
                edge += 1
    poly = pv.PolyData(np.asarray(pts, dtype=float))
    poly.lines = np.hstack(lines)
    return poly


__all__ = ["StructureRenderer"]
