"""Primitive ↔ crystallographic (conventional) cell views of a structure.

Building the crystallographic cell the way a crystallographer expects to see it
is two steps: get the right *lattice* (with the right number of atoms), then
place those atoms into the drawn cell box sensibly.

**Lattice — the centring expansion.** For a centred lattice (F, I, C, R…) the
primitive cell CRYSTAL works in holds fewer atoms than the conventional cell
(an FCC crystal's 2-atom primitive cell vs. its 8-atom cubic cell). We expand
with the integer matrix ``M`` where ``conventional_vectors = M · primitive_vectors``
(``det M`` = the centring multiplicity: 1 for P, 2 for I/C, 3 for R, 4 for F),
applied via ``ase.build.make_supercell``. A P lattice (dry ice's Pa-3, say) has
``M = I`` and is left untouched. Not pymatgen's
``get_conventional_standard_structure``: that re-orders atoms (destroying the
ordering phonon eigenvectors rely on) and re-wraps them. The tiling instead
keeps every conventional atom mapped to the primitive atom it came from, which
is what lets phonon modes animate on the conventional cell too (see
:func:`expand_modes_to_conventional`).

**Placement — molecule-aware wrapping.** CRYSTAL centres its coordinates on the
origin (fractional roughly ``[-0.5, 0.5]``), but the viewport draws the cell box
as ``[0, 1)``, so half the atoms fall outside it and any molecule straddling a
face renders as broken fragments (dry ice's four CO₂ show up as one molecule
plus three bare carbons). :func:`_wrap_molecules` reconstructs whole molecules
across the periodic boundary and translates each into the box. Wrapping is only
ever integer lattice translations, so atom order — and thus the phonon
correspondence — is preserved.

pymatgen/ase are imported lazily so ``core`` stays importable (and unit-testable)
without them, matching ``crystalio.loader``. No function mutates its argument.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Tuple

import numpy as np

from crystalline.core.phonons import PhononMode, PhononModes
from crystalline.core.structure import Structure

# ase array key used to carry each atom's source index through the supercell
# expansion, so we can map conventional atoms back to their primitive parent.
_SOURCE_INDEX_KEY = "__crystalline_source_index__"


class CellView(str, Enum):
    """Which cell the viewport draws.

    ``CRYSTALLOGRAPHIC`` is the conventional cell (the default); ``PRIMITIVE``
    is the cell as loaded from the CRYSTAL file.
    """

    CRYSTALLOGRAPHIC = "crystallographic"
    PRIMITIVE = "primitive"


def _conventional_expansion(structure: Structure, symprec: float) -> Tuple["object", np.ndarray]:
    """Build the conventional cell of ``structure`` by centring expansion.

    Returns ``(ase.Atoms, mapping)`` where ``mapping[i]`` is the index of the
    primitive-cell atom that conventional atom ``i`` is an image of. A
    non-periodic system, an already-conventional cell, or a P lattice all yield
    the loaded atoms unchanged with an identity mapping.
    """
    source = structure.to_ase()
    identity = (source, np.arange(len(source)))

    if not structure.is_periodic or np.allclose(structure.cell, 0.0):
        return identity

    # Lazy imports: keep `core` importable without pymatgen/ase build tools.
    from ase.build import make_supercell
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    pmg = AseAtomsAdaptor().get_structure(source)
    sga = SpacegroupAnalyzer(pmg, symprec=symprec)

    # Only expand when the loaded cell really is the primitive one; if it is
    # already the conventional cell (or otherwise not primitive), it is its own
    # conventional cell — expanding again would wrongly multiply the atoms.
    primitive = sga.find_primitive()
    if primitive is None or len(source) != len(primitive):
        return identity

    # M maps primitive → conventional lattice vectors, in the lattice basis
    # (orientation-independent, so it applies to the loaded cell as-is).
    conv_to_prim = np.asarray(sga.get_conventional_to_primitive_transformation_matrix(), dtype=float)
    matrix = np.rint(np.linalg.inv(conv_to_prim)).astype(int)
    if abs(int(round(np.linalg.det(matrix)))) <= 1:
        return identity  # P lattice: conventional == primitive

    tagged = source.copy()
    tagged.set_array(_SOURCE_INDEX_KEY, np.arange(len(source)))
    conventional = make_supercell(tagged, matrix)

    if _SOURCE_INDEX_KEY not in conventional.arrays:
        # make_supercell didn't carry the tag through: fall back rather than
        # return a mapping we can't trust.
        return identity
    mapping = np.asarray(conventional.get_array(_SOURCE_INDEX_KEY), dtype=int)
    del conventional.arrays[_SOURCE_INDEX_KEY]
    return conventional, mapping


def _wrap_molecules(atoms, mult: float = 1.15):
    """Return ``atoms`` with whole molecules translated into the cell box.

    Molecules are found by covalent-radius bonding under the minimum-image
    convention, each connected component is reassembled so it is contiguous
    (never split across a cell face), then shifted so its anchor atom sits in
    ``[0, 1)`` fractional coordinates. A component that percolates the lattice
    (an extended framework, not a discrete molecule) can't be made whole, so its
    atoms are simply wrapped individually — the standard depiction for those.

    Only integer lattice translations are applied, so atom order is preserved
    (and with it the phonon-eigenvector correspondence). Non-periodic systems
    are returned unchanged.
    """
    import numpy as np

    if not atoms.pbc.any() or np.allclose(np.asarray(atoms.cell), 0.0):
        return atoms

    try:
        from ase.neighborlist import natural_cutoffs, neighbor_list

        i, j, offset = neighbor_list("ijD", atoms, natural_cutoffs(atoms, mult=mult))
    except Exception:  # noqa: BLE001 - never let a bonding hiccup break rendering
        return atoms

    n = len(atoms)
    cell = np.asarray(atoms.cell, dtype=float)
    neighbours: list[list] = [[] for _ in range(n)]
    for a, b, disp in zip(i, j, offset):
        neighbours[a].append((b, disp))

    positions = atoms.get_positions()
    unwrapped = positions.copy()
    visited = [False] * n
    for start in range(n):
        if visited[start]:
            continue
        visited[start] = True
        component = [start]
        stack = [start]
        while stack:  # depth-first reassembly via minimum-image displacements
            u = stack.pop()
            for v, disp in neighbours[u]:
                if not visited[v]:
                    visited[v] = True
                    unwrapped[v] = unwrapped[u] + disp
                    component.append(v)
                    stack.append(v)
        comp = np.asarray(component)
        frac = np.linalg.solve(cell.T, unwrapped[comp].T).T
        if np.any(frac.max(0) - frac.min(0) >= 1.0):
            # Extended framework: reassembly is meaningless — wrap each atom.
            unwrapped[comp] = (frac % 1.0) @ cell
        else:
            unwrapped[comp] -= np.floor(frac[0]) @ cell  # anchor into [0, 1)

    out = atoms.copy()
    out.set_positions(unwrapped)
    return out


def to_conventional(structure: Structure, symprec: float = 1e-2) -> Structure:
    """Return the crystallographic (conventional) cell of ``structure``.

    The lattice is expanded for centred cells and the atoms are wrapped
    molecule-wise into the cell box. Non-periodic systems are returned unchanged
    as a fresh copy.

    Parameters
    ----------
    structure:
        The structure to expand (typically the primitive cell as loaded).
    symprec:
        Symmetry tolerance in Angstrom handed to the space-group analyser.
    """
    atoms, _ = _conventional_expansion(structure, symprec)
    return Structure.from_ase(_wrap_molecules(atoms))


def expand_modes_to_conventional(
    structure: Structure,
    modes: PhononModes,
    symprec: float = 1e-2,
) -> Tuple[Structure, PhononModes]:
    """Return the conventional cell together with phonon modes defined on it.

    Each conventional atom is an image of a primitive atom; at the Gamma point
    (the modes we load) every image of a primitive cell moves in phase, so a
    conventional-cell eigenvector is just the primitive eigenvector replicated
    onto each atom's parent. When no expansion happens (P lattice, molecule),
    the modes come back unchanged. The returned structure is molecule-wrapped to
    match :func:`to_conventional`; wrapping is pure lattice translation, so the
    per-atom eigenvectors still line up.
    """
    atoms, mapping = _conventional_expansion(structure, symprec)
    expanded = PhononModes(
        [PhononMode(frequency=m.frequency, eigenvector=m.eigenvector[mapping]) for m in modes]
    )
    return Structure.from_ase(_wrap_molecules(atoms)), expanded


def as_view(structure: Structure, view: CellView) -> Structure:
    """Return ``structure`` rendered in the requested :class:`CellView`.

    Always a fresh copy, so editing the returned structure never mutates the
    source the view was derived from.
    """
    if view is CellView.PRIMITIVE:
        return Structure.from_ase(structure.to_ase())
    return to_conventional(structure)


def tile_supercell(
    structure: Structure,
    reps: Tuple[int, int, int],
    modes: Optional[PhononModes] = None,
) -> Tuple[Structure, Optional[PhononModes]]:
    """Tile ``structure`` ``reps`` = ``(na, nb, nc)`` times along its lattice vectors.

    Applied on top of whichever cell view is active, so the supercell inherits
    its molecule-wrapped atoms (each image cell stays whole). Non-periodic axes
    are never tiled. If phonon ``modes`` are given they are replicated onto the
    image atoms — correct at the Gamma point, where every image cell moves in
    phase — and returned alongside the supercell; the atom order is preserved so
    each image atom's eigenvector matches its parent.
    """
    atoms = structure.to_ase()
    pbc = atoms.get_pbc()
    reps = tuple(int(r) if pbc[k] else 1 for k, r in enumerate(reps))
    if reps == (1, 1, 1) or not pbc.any():
        return Structure.from_ase(atoms), modes

    tagged = atoms.copy()
    tagged.set_array(_SOURCE_INDEX_KEY, np.arange(len(atoms)))
    supercell = tagged.repeat(reps)

    mapping = None
    if _SOURCE_INDEX_KEY in supercell.arrays:
        mapping = np.asarray(supercell.get_array(_SOURCE_INDEX_KEY), dtype=int)
        del supercell.arrays[_SOURCE_INDEX_KEY]

    tiled_modes = modes
    if modes is not None and mapping is not None:
        tiled_modes = PhononModes(
            [PhononMode(frequency=m.frequency, eigenvector=m.eigenvector[mapping]) for m in modes]
        )
    elif modes is not None:
        # Couldn't recover the mapping: drop the modes rather than hand back
        # eigenvectors whose atom count no longer matches the supercell.
        tiled_modes = None
    return Structure.from_ase(supercell), tiled_modes


def _boundary_completion(structure: Structure, tol: float, mult: float = 1.15):
    """Add periodic images of every molecule/atom that pokes into the cell.

    Returns ``(ase.Atoms, mapping)`` where the atoms are the cell's own atoms
    PLUS the atoms of any periodic image whose molecule has at least one atom
    within ``[-tol, 1+tol]`` fractional (i.e. inside or just touching the box).
    Whole molecules are kept together, so a corner CO2 shows up complete at every
    corner it belongs to. ``mapping[i]`` is the index of the original atom that
    image atom ``i`` is a copy of (for replicating phonon eigenvectors).
    """
    import itertools

    atoms = structure.to_ase()
    n = len(atoms)
    identity = (atoms, np.arange(n))
    if not structure.is_periodic or np.allclose(structure.cell, 0.0) or n == 0:
        return identity

    try:
        from ase import Atoms
        from ase.neighborlist import natural_cutoffs, neighbor_list

        i, j, disp = neighbor_list("ijD", atoms, natural_cutoffs(atoms, mult=mult))
    except Exception:  # noqa: BLE001 - never let a bonding hiccup break rendering
        return identity

    cell = np.asarray(atoms.cell, dtype=float)
    positions = atoms.get_positions()
    numbers = atoms.get_atomic_numbers()
    neighbours: list[list] = [[] for _ in range(n)]
    for a, b, d in zip(i, j, disp):
        neighbours[a].append((b, d))

    # Reassemble contiguous molecules (whole across the periodic boundary).
    unwrapped = positions.copy()
    visited = [False] * n
    components: list = []
    for start in range(n):
        if visited[start]:
            continue
        visited[start] = True
        component = [start]
        stack = [start]
        while stack:
            u = stack.pop()
            for v, d in neighbours[u]:
                if not visited[v]:
                    visited[v] = True
                    unwrapped[v] = unwrapped[u] + d
                    component.append(v)
                    stack.append(v)
        components.append(np.asarray(component))

    # A component that percolates the lattice (extended framework) is imaged atom
    # by atom instead of as one blob. Crucially, such atoms are imaged from their
    # ORIGINAL (in-cell) positions, not the DFS-``unwrapped`` ones: unwrapping an
    # extended network drifts by lattice vectors around its loops, which would
    # place the drawn atoms off their true sites — visibly detaching the bonds
    # and coordination polyhedra (analysed on the clean cell) from the atoms.
    base_pos = unwrapped.copy()
    clusters: list = []
    for comp in components:
        frac = np.linalg.solve(cell.T, unwrapped[comp].T).T
        if np.any(frac.max(0) - frac.min(0) >= 1.0):
            base_pos[comp] = positions[comp]  # canonical sites, no unwrap drift
            clusters.extend(np.array([idx]) for idx in comp)
        else:
            clusters.append(comp)

    out_pos: list = []
    out_num: list = []
    out_parent: list = []
    seen: set = set()
    shifts = list(itertools.product((-1, 0, 1), repeat=3))
    for comp in clusters:
        cluster_pos = base_pos[comp]
        for shift in shifts:
            image = cluster_pos + np.asarray(shift, dtype=float) @ cell
            frac = np.linalg.solve(cell.T, image.T).T
            touches = np.any(np.all((frac >= -tol) & (frac <= 1.0 + tol), axis=1))
            if not touches:
                continue
            for k, idx in enumerate(comp):
                key = tuple(np.round(image[k], 3))
                if key in seen:
                    continue
                seen.add(key)
                out_pos.append(image[k])
                out_num.append(int(numbers[idx]))
                out_parent.append(int(idx))

    if not out_pos:
        return identity
    result = Atoms(
        numbers=out_num, positions=np.asarray(out_pos), cell=cell, pbc=atoms.get_pbc()
    )
    return result, np.asarray(out_parent, dtype=int)


def complete_boundary(
    structure: Structure,
    modes: Optional[PhononModes] = None,
    tol: float = 2e-2,
) -> Tuple[Structure, Optional[PhononModes]]:
    """Return ``structure`` with boundary molecules/atoms completed by images.

    Every molecule that partially belongs to the unit cell is drawn whole, at
    each cell position it touches (all corners/edges/faces). Phonon ``modes`` are
    replicated onto the image atoms (in phase, correct at Gamma).
    """
    atoms, mapping = _boundary_completion(structure, tol)
    new_modes = modes
    if modes is not None:
        new_modes = PhononModes(
            [PhononMode(frequency=m.frequency, eigenvector=m.eigenvector[mapping]) for m in modes]
        )
    return Structure.from_ase(atoms), new_modes


__all__ = [
    "CellView",
    "to_conventional",
    "expand_modes_to_conventional",
    "as_view",
    "tile_supercell",
    "complete_boundary",
]
