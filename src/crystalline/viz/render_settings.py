"""Tunable appearance settings for :class:`~crystalline.viz.renderer.StructureRenderer`.

A plain, Qt-free dataclass so the renderer stays UI-agnostic and the values are
easy to unit-test and to drive from a settings dialog.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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

    # bonds
    show_bonds: bool = True
    bond_radius: float = 0.06    # Angstrom
    bond_tolerance: float = 1.15  # bonded if dist < tolerance * (r_i + r_j)

    # bonds (continued)
    bond_color: str = "#888888"   # "colour 1": solid, and one half of split/gradient
    bond_color2: str = "#4c72b0"  # "colour 2": the other half of split / end of gradient
    # "solid" (colour 1), "split" (half colour 1 / half colour 2), "gradient" (1 → 2).
    bond_color_mode: str = "solid"

    # cell / gizmo
    show_cell: bool = True
    show_lattice_vectors: bool = True

    # atom labels (element symbols drawn at each atom, capped for large cells)
    show_atom_labels: bool = False
    atom_label_size: int = 16  # point size of the element-symbol labels

    # coordination polyhedra (VESTA-style)
    show_polyhedra: bool = False
    polyhedra_opacity: float = 0.75
    polyhedra_min_vertices: int = 4  # only draw around atoms with >= this many bonds

    # scene / camera
    background_color: str = "white"
    parallel_projection: bool = False   # orthographic when True, perspective otherwise
    show_orientation_axes: bool = False  # the little xyz marker in the corner

    def evolve(self, **changes) -> "RenderSettings":
        """Return a copy with some fields overridden (settings are immutable)."""
        return replace(self, **changes)


__all__ = ["RenderSettings"]
