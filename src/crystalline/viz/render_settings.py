"""Tunable appearance settings for :class:`~crystalline.viz.renderer.StructureRenderer`.

A plain, Qt-free dataclass so the renderer stays UI-agnostic and the values are
easy to unit-test and to drive from a settings dialog.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class RenderSettings:
    """How the structure is drawn. All fields have sensible ball-and-stick defaults."""

    # atoms
    atom_scale: float = 0.5      # sphere radius as a fraction of the covalent radius
    atom_opacity: float = 1.0    # 1 = opaque, <1 = translucent
    # Per-element colour overrides as an immutable tuple of ``(atomic_number, "#rrggbb")``
    # pairs (kept hashable so the frozen dataclass stays well-behaved); elements
    # absent here keep their default Jmol colour.
    atom_colors: Tuple[Tuple[int, str], ...] = field(default_factory=tuple)

    # bonds — a single fixed colour (see renderer._BOND_COLOR); no per-bond colouring
    show_bonds: bool = True
    bond_radius: float = 0.06    # Angstrom
    bond_tolerance: float = 1.15  # bonded if dist < tolerance * (r_i + r_j)

    # hydrogen bonds — dashed D–H···A contacts, drawn by default
    show_hydrogen_bonds: bool = True

    # cell / gizmo
    show_cell: bool = True
    show_lattice_vectors: bool = True

    # phonon displacement arrows: the selected mode's eigenvector drawn on the
    # atoms, so a mode reads in a still image instead of only while animating.
    # Off by default — the animation is the primary view, and arrows on top of
    # it are clutter until asked for. Every arrow is drawn the same length: they
    # mark the *direction* each atom moves, and letting them shrink with each
    # atom's share leaves the interesting small contributions invisible.
    show_mode_arrows: bool = False
    mode_arrow_scale: float = 1.2   # Angstrom, the length every arrow is drawn at
    mode_arrow_color: str = "#d62728"

    # thermal ellipsoids (ADP): the surface enclosing `adp_probability` of each
    # atom's displacement distribution, drawn at one of the temperatures the
    # output reported. Off unless the loaded file actually carries ADPs.
    # Off here because most files carry no ADPs at all; the Display panel
    # switches it on when a file that has them is opened.
    show_adp_ellipsoids: bool = False
    # ORTEP and structure reports draw 50%; 99% is the default here because
    # computed ADPs are small enough that the conventional surface is hard to
    # see on screen. Same tensor either way — only how far out it is drawn.
    adp_probability: float = 0.99
    adp_opacity: float = 0.85
    adp_temperature_index: int = 0   # which reported temperature to draw

    # atom labels (element symbols drawn at each atom, capped for large cells)
    show_atom_labels: bool = False
    atom_label_size: int = 16  # point size of the element-symbol labels

    # coordination polyhedra (VESTA-style), off until asked for: they hide the
    # atoms they enclose, and a first look at a structure is usually the atoms
    show_polyhedra: bool = False
    polyhedra_opacity: float = 0.3  # translucent enough to see the atoms inside
    polyhedra_min_vertices: int = 4  # only draw around atoms with >= this many bonds

    # geometry-measurement overlays (Geometry panel): dot markers, distance/angle
    # paths, and least-squares plane patches — each independently coloured.
    measure_point_color: str = "#ff7f0e"  # dot markers
    measure_line_color: str = "#ff7f0e"   # distance / angle / dihedral paths
    measure_plane_color: str = "#1f77b4"  # least-squares plane patches

    # scene / camera
    background_color: str = "white"
    parallel_projection: bool = True    # orthographic by default; perspective when False
    show_orientation_axes: bool = False  # the little xyz marker in the corner


__all__ = ["RenderSettings"]
