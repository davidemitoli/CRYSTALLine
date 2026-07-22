"""Crystallographic analysis of a :class:`Structure` — space group, lattice,
density, etc. — derived with pymatgen (the same engine CRYSTALClear uses).

Kept Qt-free and with a lazy pymatgen import so it can be unit-tested without a
display, matching the rest of ``core``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from crystalline.core.structure import Structure


@dataclass(frozen=True)
class CrystalInfo:
    """Crystallographic summary of a structure (fields are ``None`` when N/A)."""

    formula: str
    n_atoms: int
    periodic: bool
    dimensionality: str
    space_group_symbol: Optional[str] = None
    space_group_number: Optional[int] = None
    crystal_system: Optional[str] = None
    point_group: Optional[str] = None
    z: Optional[int] = None
    a: Optional[float] = None
    b: Optional[float] = None
    c: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    gamma: Optional[float] = None
    volume: Optional[float] = None
    density: Optional[float] = None

    def rows(self) -> List[Tuple[str, str]]:
        """(label, value) pairs for display, skipping fields that don't apply."""
        rows: List[Tuple[str, str]] = [
            ("Formula", self.formula),
            ("Atoms", str(self.n_atoms)),
            ("Dimensionality", self.dimensionality),
        ]
        if self.space_group_symbol is not None:
            rows.append(("Space group", f"{self.space_group_symbol} (No. {self.space_group_number})"))
        if self.crystal_system is not None:
            rows.append(("Crystal system", self.crystal_system.capitalize()))
        if self.point_group is not None:
            rows.append(("Point group", self.point_group))
        if self.z is not None:
            rows.append(("Formula units Z", str(self.z)))
        if self.a is not None:
            rows.append(("a, b, c (Å)", f"{self.a:.4f}, {self.b:.4f}, {self.c:.4f}"))
            rows.append(("α, β, γ (°)", f"{self.alpha:.3f}, {self.beta:.3f}, {self.gamma:.3f}"))
        if self.volume is not None:
            rows.append(("Cell volume (Å³)", f"{self.volume:.3f}"))
        if self.density is not None:
            rows.append(("Density (g/cm³)", f"{self.density:.3f}"))
        return rows


_DIMENSIONALITY = {0: "Molecule (0D)", 1: "Polymer (1D)", 2: "Slab (2D)", 3: "Bulk crystal (3D)"}


def analyze(structure: Structure, symprec: float = 1e-2) -> CrystalInfo:
    """Return a :class:`CrystalInfo` for ``structure``.

    For a periodic system the reported lattice/volume/density/Z are those of the
    conventional (crystallographic) cell — what a crystallographer expects in a
    summary table — regardless of which cell was loaded. Non-periodic systems get
    just the composition.
    """
    ase_atoms = structure.to_ase()
    n_atoms = len(ase_atoms)
    ndim = int(sum(bool(p) for p in ase_atoms.get_pbc()))

    if n_atoms == 0:
        return CrystalInfo(formula="—", n_atoms=0, periodic=False, dimensionality="—")

    from pymatgen.core import Composition

    composition = Composition(ase_atoms.get_chemical_formula())
    formula = composition.reduced_formula

    if not structure.is_periodic:
        return CrystalInfo(
            formula=formula, n_atoms=n_atoms, periodic=False,
            dimensionality=_DIMENSIONALITY.get(ndim, "Molecule (0D)"),
        )

    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    pmg = AseAtomsAdaptor().get_structure(ase_atoms)
    dimensionality = _DIMENSIONALITY.get(ndim, "Bulk crystal (3D)")

    try:
        sga = SpacegroupAnalyzer(pmg, symprec=symprec)
        conventional = sga.get_conventional_standard_structure()
        lattice = conventional.lattice
        reduced, factor = conventional.composition.get_reduced_composition_and_factor()
        return CrystalInfo(
            formula=formula,
            n_atoms=n_atoms,
            periodic=True,
            dimensionality=dimensionality,
            space_group_symbol=sga.get_space_group_symbol(),
            space_group_number=sga.get_space_group_number(),
            crystal_system=sga.get_crystal_system(),
            point_group=sga.get_point_group_symbol(),
            z=int(round(factor)),
            a=lattice.a, b=lattice.b, c=lattice.c,
            alpha=lattice.alpha, beta=lattice.beta, gamma=lattice.gamma,
            volume=conventional.volume,
            density=float(conventional.density),
        )
    except Exception:  # noqa: BLE001 - symmetry analysis can fail; still show the lattice
        lattice = pmg.lattice
        return CrystalInfo(
            formula=formula, n_atoms=n_atoms, periodic=True, dimensionality=dimensionality,
            a=lattice.a, b=lattice.b, c=lattice.c,
            alpha=lattice.alpha, beta=lattice.beta, gamma=lattice.gamma,
            volume=pmg.volume, density=float(pmg.density),
        )


__all__ = ["CrystalInfo", "analyze"]
