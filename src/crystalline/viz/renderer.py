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
_ATOM_SCALE = 0.5
_BOND_RADIUS = 0.12
# Two atoms are bonded if their distance < scale * (r_i + r_j).
_BOND_TOLERANCE = 1.15

# Material for the atom spheres. The specular highlight is what makes small
# on-screen atoms read as 3D balls rather than flat disks.
_ATOM_MATERIAL = dict(
    smooth_shading=True,
    specular=0.5,
    specular_power=15,
    ambient=0.3,
    diffuse=0.6,
)

# Above this atom count, bonds are not recomputed on every live position update
# (animation frames / drag) — they refresh on commit instead — to avoid lag.
_LIVE_BOND_MAX_ATOMS = 300

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
        self._cell_actor = None
        self._polyhedra_actor = None
        self._label_actor = None
        self._bond_structure: Optional[Structure] = None  # clean cell for coordination
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
        self._cell_actor = None
        self._polyhedra_actor = None
        self._label_actor = None
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
        if settings.show_polyhedra:
            self._draw_polyhedra()
        if settings.show_cell:
            self._draw_cell()
        if settings.show_lattice_vectors:
            self._draw_lattice_vectors()
        if settings.show_atom_labels:
            self._draw_atom_labels()
        self._restore_camera(saved_camera)
        self.plotter.render()

    def _current_camera(self):
        """The current camera placement, or ``None`` if it can't be read."""
        try:
            return self.plotter.camera_position
        except Exception:  # noqa: BLE001 - bare/uninitialised plotter
            return None

    def _restore_camera(self, camera) -> None:
        """Put a previously-captured camera placement back (a redraw must not move it)."""
        if camera is None:
            return
        try:
            self.plotter.camera_position = camera
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
        mode = self._settings.bond_color_mode
        if mode in ("split", "gradient"):
            mesh = self._colored_bond_mesh(positions, numbers, radius, tol, mode)
            if mesh is None:
                return
            self._bond_actor = self.plotter.add_mesh(
                mesh, scalars="rgb", rgb=True, smooth_shading=True, render=False
            )
        else:  # solid
            mesh = _bond_mesh(positions, numbers, radius, tol)
            if mesh is None:
                return
            self._bond_actor = self.plotter.add_mesh(
                mesh, color=self._settings.bond_color, smooth_shading=True, render=False
            )
        self._bond_actor.SetPickable(False)  # only atoms are pick targets

    def _colored_bond_mesh(self, positions, numbers, radius, tol, mode):
        """Tube mesh coloured with the two chosen bond colours (split or gradient).

        ``gradient`` blends colour 1 into colour 2 along each bond (the tube
        interpolates the per-endpoint RGB). ``split`` colours one half colour 1
        and the other colour 2 by inserting a duplicated midpoint so the two
        halves are separate cells with a hard colour break.
        """
        positions = np.asarray(positions, dtype=float)
        i, j = _bonded_pairs(positions, numbers, tol)
        if len(i) == 0:
            return None
        c1 = _hex_to_rgb(self._settings.bond_color)
        c2 = _hex_to_rgb(self._settings.bond_color2)
        pi, pj = positions[i], positions[j]
        ci = np.tile(c1, (len(i), 1))
        cj = np.tile(c2, (len(i), 1))

        if mode == "gradient":
            pts = np.empty((2 * len(i), 3), dtype=float)
            pts[0::2], pts[1::2] = pi, pj
            rgb = np.empty((2 * len(i), 3), dtype=np.uint8)
            rgb[0::2], rgb[1::2] = ci, cj
            starts = np.arange(0, 2 * len(i), 2)
            lines = np.column_stack([np.full(len(i), 2), starts, starts + 1])
        else:  # split: 4 points (i, mid, mid, j), 2 cells per bond
            mid = (pi + pj) / 2.0
            pts = np.empty((4 * len(i), 3), dtype=float)
            pts[0::4], pts[1::4], pts[2::4], pts[3::4] = pi, mid, mid, pj
            rgb = np.empty((4 * len(i), 3), dtype=np.uint8)
            rgb[0::4], rgb[1::4], rgb[2::4], rgb[3::4] = ci, ci, cj, cj
            base = np.arange(0, 4 * len(i), 4)
            first = np.column_stack([np.full(len(i), 2), base, base + 1])
            second = np.column_stack([np.full(len(i), 2), base + 2, base + 3])
            lines = np.empty((2 * len(i), 3), dtype=np.int64)
            lines[0::2], lines[1::2] = first, second

        poly = pv.PolyData(pts)
        poly.lines = lines.ravel()
        poly["rgb"] = rgb
        return poly.tube(radius=radius)

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

    def _polyhedra_mesh(self) -> Optional["pv.PolyData"]:
        """Build the polyhedra mesh: CrystalNN on the clean cell, else a fallback.

        Both paths analyse the clean ``bond_structure`` (the periodic cell before
        boundary-completion), so the result is in the same frame as the shown
        atoms and doesn't double-count the packed cell's duplicate images. The
        distance fallback runs whenever CrystalNN is unavailable, too slow, *or*
        found no polyhedra — previously an empty CrystalNN result drew nothing
        (the coordination showed up only on big supercells, where the atom count
        tipped it into the fallback).
        """
        min_vertices = self._settings.polyhedra_min_vertices
        analysis = self._bond_structure if self._bond_structure is not None else self._structure

        from crystalline.core.bonds import connectivity

        overrides = dict(self._settings.atom_colors)  # tint polyhedra by the centre's colour
        conn = connectivity(analysis, min_vertices)
        if conn is not None and conn.polyhedra:
            return _hull_mesh([(z, pts) for z, pts in conn.polyhedra], overrides)
        # periodic distance search on the clean cell (CrystalNN missing/empty)
        return _polyhedra_mesh_fallback(
            analysis.to_ase(), self._settings.bond_tolerance, min_vertices, overrides
        )

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
        edges = _cell_edges(cell)
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
        for vec, color in zip(cell, _LATTICE_COLORS):
            length = float(np.linalg.norm(vec))
            if length < 1e-6:
                continue
            arrow = pv.Arrow(start=base, direction=vec, scale=arrow_length)
            actor = self.plotter.add_mesh(arrow, color=color, render=False)
            actor.SetPickable(False)  # gizmo, never a pick target
            tips.append(base + (vec / length) * arrow_length)
        if tips:
            labels = self.plotter.add_point_labels(
                np.asarray(tips, dtype=float),
                list(_LATTICE_LABELS[: len(tips)]),
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


def _element_colors(numbers: np.ndarray) -> np.ndarray:
    """Per-atom uint8 RGB rows (Jmol colours) for a glyph's ``rgb`` scalars."""
    return (jmol_colors[np.asarray(numbers, dtype=int)] * 255).astype(np.uint8)


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
    positions: np.ndarray, numbers: np.ndarray, radius: float, tolerance: float
) -> Optional[pv.PolyData]:
    """Bond tube mesh for the bonded pairs (radius/tolerance from settings)."""
    positions = np.asarray(positions, dtype=float)
    i, j = _bonded_pairs(positions, numbers, tolerance)
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
        merged = merged.merge(extra)
    return merged


def _polyhedra_mesh_fallback(
    atoms, tolerance: float, min_vertices: int, overrides: Optional[dict] = None
) -> Optional[pv.PolyData]:
    """Distance-based coordination polyhedra (used when CrystalNN isn't available).

    Uses ``ase.neighbor_list`` (periodic, so ligands cross the cell boundary
    correctly) for each atom's neighbours, then keeps only the chemically
    sensible ones: a polyhedron is drawn around a **cation** centre using its
    **anion** ligands — neighbours more electronegative than the centre. Without
    that filter, covalent radii spuriously bond big cations to each other (a Ca
    "coordinating" nearby Ca/Si), so the hull swallows other cations; the filter
    reproduces the cation–anion polyhedra CrystalNN would give. Returns one
    merged mesh with per-vertex RGB, or ``None`` if nothing is drawable.
    """
    from collections import defaultdict

    if len(atoms) == 0:
        return None
    try:
        from ase.neighborlist import natural_cutoffs, neighbor_list

        i, j, offset = neighbor_list("ijD", atoms, natural_cutoffs(atoms, mult=tolerance))
    except Exception:  # noqa: BLE001 - bonding hiccup must not break rendering
        return None
    if len(i) == 0:
        return None

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
        polyhedra.append((int(numbers[center]), np.asarray(pts, dtype=float)))
    return _hull_mesh(polyhedra, overrides)


def _cell_edges(cell: np.ndarray) -> pv.PolyData:
    """The 12 edges of the parallelepiped defined by lattice vectors (rows)."""
    a, b, c = cell[0], cell[1], cell[2]
    corners = np.array(
        [
            [0, 0, 0], a, b, c,
            a + b, a + c, b + c, a + b + c,
        ],
        dtype=float,
    )
    # index pairs into `corners` for the 12 edges
    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 5), (2, 4), (2, 6),
        (3, 5), (3, 6), (4, 7), (5, 7), (6, 7),
    ]
    pts = []
    lines = []
    for i, (u, v) in enumerate(edges):
        pts.append(corners[u])
        pts.append(corners[v])
        lines.append([2, 2 * i, 2 * i + 1])
    poly = pv.PolyData(np.asarray(pts, dtype=float))
    poly.lines = np.hstack(lines)
    return poly


__all__ = ["StructureRenderer"]
