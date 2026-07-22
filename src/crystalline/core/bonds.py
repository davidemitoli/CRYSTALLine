"""Chemically-aware, periodic connectivity: bonds and coordination polyhedra.

Ball-and-stick tools usually bond atoms whose distance is within a covalent-radius
tolerance. That over-bonds ionic crystals — a big cation like Ca (covalent radius
1.76 Å) ends up "bonded" to nearby Ca and Si, so its coordination polyhedron
swallows other cations. Instead we use pymatgen's ``CrystalNN`` near-neighbour
algorithm, which is periodic-aware and finds real cation–anion bonds (Si→4 O,
Ca→6–8 O for alite), and place bonds/polyhedra using the *image* coordinates of
each neighbour so they correctly cross the cell boundary.

pymatgen is imported lazily; the caller falls back to distance-based bonding when
this returns ``None`` (pymatgen missing, non-periodic, or too large to analyse).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from crystalline.core.structure import Structure

# CrystalNN is O(N · neighbours); above this it's too slow for an interactive
# rebuild, so the renderer falls back to the fast distance method.
_MAX_ANALYSIS_ATOMS = 400


@dataclass
class Connectivity:
    """Bonds and coordination polyhedra for one structure.

    ``bonds`` is an ``(M, 2, 3)`` array of cartesian segment endpoints (an
    endpoint may lie outside the cell — a neighbour's periodic image).
    ``polyhedra`` is a list of ``(atomic_number, ligand_positions)`` for each
    cation-centred coordination polyhedron.
    """

    bonds: np.ndarray
    polyhedra: List[Tuple[int, np.ndarray]] = field(default_factory=list)


def connectivity(structure: Structure, min_vertices: int = 4) -> Optional[Connectivity]:
    """Return :class:`Connectivity` via CrystalNN, or ``None`` if not applicable.

    ``None`` means "fall back to distance-based bonding": non-periodic system,
    too many atoms, or pymatgen/CrystalNN unavailable or failing.
    """
    if not structure.is_periodic or len(structure) == 0:
        return None
    if len(structure) > _MAX_ANALYSIS_ATOMS:
        return None

    try:
        from pymatgen.analysis.local_env import CrystalNN
        from pymatgen.io.ase import AseAtomsAdaptor
    except Exception:  # noqa: BLE001 - pymatgen missing/broken
        return None

    try:
        pmg = AseAtomsAdaptor().get_structure(structure.to_ase())
        near = CrystalNN()
        electroneg = [site.specie.X for site in pmg]

        segments: List[np.ndarray] = []
        seen: set = set()
        polyhedra: List[Tuple[int, np.ndarray]] = []
        for i, site in enumerate(pmg):
            infos = near.get_nn_info(pmg, i)
            ligands = np.array([info["site"].coords for info in infos], dtype=float)
            for coord in ligands:
                # de-duplicate the symmetric i–j bond by its rounded endpoints
                key = tuple(sorted((_round(site.coords), _round(coord))))
                if key in seen:
                    continue
                seen.add(key)
                segments.append(np.array([site.coords, coord], dtype=float))
            # a coordination polyhedron only for cation centres (the centre is
            # less electronegative than its ligands) with enough vertices
            if len(ligands) >= min_vertices:
                neighbour_x = [electroneg[info["site_index"]] for info in infos]
                if electroneg[i] is not None and all(x is not None for x in neighbour_x):
                    if electroneg[i] < np.mean(neighbour_x):
                        polyhedra.append((int(site.specie.Z), ligands))
    except Exception:  # noqa: BLE001 - CrystalNN can choke on odd cells
        return None

    bonds = np.asarray(segments, dtype=float) if segments else np.empty((0, 2, 3))
    return Connectivity(bonds=bonds, polyhedra=polyhedra)


def _round(xyz: np.ndarray) -> tuple:
    return tuple(np.round(xyz, 3))


__all__ = ["Connectivity", "connectivity"]
