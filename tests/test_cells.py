"""Tests for the primitive ↔ crystallographic cell views (no display needed).

These exercise the real symmetry analysis, so they need pymatgen; they skip if
it (or ase's structure builders) is unavailable.
"""

import numpy as np
import pytest

from crystalline.core.cells import (
    CellView,
    as_view,
    complete_boundary,
    expand_modes_to_conventional,
    tile_supercell,
    to_conventional,
)
from crystalline.core.phonons import PhononMode, PhononModes
from crystalline.core.structure import Structure

pytest.importorskip("pymatgen")
from ase import Atoms  # noqa: E402  (after importorskip)
from ase.build import bulk  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402


def _dry_ice() -> Structure:
    """Solid CO2 (Pa-3, space group 205): 4 CO2 molecules in a cubic P cell.

    Coordinates as CRYSTAL prints them — centred on the origin (fractional
    roughly [-0.4, 0.5]), so three of the four CO2 straddle the [0, 1) cell box
    and render as broken fragments until molecule-aware wrapping reassembles
    them. This is the real regression case.
    """
    a = 5.67747679
    d = 0.6752559463681  # C–O projected onto each cubic axis
    p = 2.838738395  # a / 2
    positions = np.array(
        [
            [0, 0, 0], [p, 0, p], [p, p, 0], [0, p, p],           # 4 carbons
            [d, d, d], [p - d, -d, -(p - d)], [-(p - d), p - d, -d], [-d, -(p - d), p - d],
            [-d, -d, -d], [-(p - d), d, p - d], [p - d, -(p - d), d], [d, p - d, -(p - d)],
        ]
    )
    atoms = Atoms(["C"] * 4 + ["O"] * 8, positions=positions, cell=np.diag([a, a, a]), pbc=True)
    return Structure.from_ase(atoms)


def _o_neighbours_per_c(structure: Structure) -> list:
    """How many O atoms sit within a C–O bond length of each C (no PBC).

    2 for every carbon means the four CO2 molecules render whole, as the drawn
    cell sees them.
    """
    pos = structure.positions
    syms = structure.symbols
    carbons = [i for i, s in enumerate(syms) if s == "C"]
    oxygens = [i for i, s in enumerate(syms) if s == "O"]
    tree = cKDTree(pos[oxygens])
    return [len(tree.query_ball_point(pos[c], 1.4)) for c in carbons]


def test_conventional_expands_centred_lattice():
    # FCC rock-salt NaCl: 2-atom primitive cell -> 8-atom cubic conventional cell.
    primitive = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64))
    conventional = to_conventional(primitive)

    assert len(primitive) == 2
    assert len(conventional) == 8
    assert np.allclose(conventional.to_ase().cell.angles(), 90.0)


def test_p_lattice_keeps_all_atoms_and_order():
    # Dry ice is a P lattice: no centring expansion, so every atom and the atom
    # order survive (only lattice-translation wrapping may move atoms).
    dry = _dry_ice()
    conventional = to_conventional(dry)
    assert len(conventional) == len(dry) == 12
    assert conventional.symbols == dry.symbols


def test_conventional_reassembles_broken_molecules():
    # The regression: as loaded, only the origin CO2 is whole; the other three
    # carbons have lost their oxygens across the cell boundary. The
    # crystallographic cell must show all four CO2 intact.
    dry = _dry_ice()
    assert _o_neighbours_per_c(dry) != [2, 2, 2, 2]  # broken as loaded

    conventional = to_conventional(dry)
    assert _o_neighbours_per_c(conventional) == [2, 2, 2, 2]  # four whole CO2


def test_conventional_leaves_source_untouched():
    primitive = Structure.from_ase(bulk("Fe", "bcc", a=2.87))
    _ = to_conventional(primitive)
    assert len(primitive) == 1  # transforming must not mutate the input


def test_non_periodic_structure_is_returned_unchanged():
    mol = Structure.empty()
    mol.add_atom("O", [0.0, 0.0, 0.0])
    mol.add_atom("H", [0.76, 0.59, 0.0])

    result = to_conventional(mol)
    assert len(result) == 2
    assert not result.is_periodic
    assert result is not mol


def test_as_view_dispatches_and_copies():
    primitive = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64))

    prim_view = as_view(primitive, CellView.PRIMITIVE)
    assert len(prim_view) == 2
    assert prim_view is not primitive  # editable copy, never the source

    conv_view = as_view(primitive, CellView.CRYSTALLOGRAPHIC)
    assert len(conv_view) == 8


def test_modes_replicate_onto_conventional_cell():
    # A 2-atom primitive mode (Na +x, Cl −x) must map onto all 8 conventional
    # atoms following each atom's primitive parent, so phonons animate there too.
    primitive = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64))
    modes = PhononModes(
        [PhononMode(frequency=100.0, eigenvector=np.array([[1, 0, 0], [-1, 0, 0]], float))]
    )

    conv_struct, conv_modes = expand_modes_to_conventional(primitive, modes)

    assert len(conv_struct) == 8
    evec = conv_modes[0].eigenvector
    assert evec.shape == (8, 3)
    for sym, disp in zip(conv_struct.symbols, evec):
        expected = [1, 0, 0] if sym == "Na" else [-1, 0, 0]
        assert np.allclose(disp, expected)


def test_modes_unchanged_for_p_lattice():
    # No expansion for a P lattice -> the modes come back one-for-one.
    dry = _dry_ice()
    evec = np.random.default_rng(0).standard_normal((len(dry), 3))
    modes = PhononModes([PhononMode(frequency=50.0, eigenvector=evec)])

    conv_struct, conv_modes = expand_modes_to_conventional(dry, modes)
    assert len(conv_struct) == len(dry)
    assert np.allclose(conv_modes[0].eigenvector, evec)


def test_supercell_tiles_atoms_and_replicates_modes():
    nacl = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64))
    modes = PhononModes(
        [PhononMode(frequency=100.0, eigenvector=np.array([[1, 0, 0], [-1, 0, 0]], float))]
    )

    sup, sup_modes = tile_supercell(nacl, (2, 2, 2), modes)

    assert len(sup) == 2 * 8  # 2-atom cell, 8 image cells
    assert sup_modes[0].eigenvector.shape == (16, 3)
    for sym, disp in zip(sup.symbols, sup_modes[0].eigenvector):
        expected = [1, 0, 0] if sym == "Na" else [-1, 0, 0]
        assert np.allclose(disp, expected)


def test_boundary_completion_adds_partial_molecules():
    # Dry ice: 4 whole CO2 in the cell -> completing the boundary adds the
    # corner/edge/face molecules that partially belong, all kept whole.
    dry = to_conventional(_dry_ice())  # molecule-wrapped, 12 atoms
    assert _o_neighbours_per_c(dry) == [2, 2, 2, 2]

    packed, _ = complete_boundary(dry)
    assert len(packed) > len(dry)  # image atoms added
    carbons = [s for s in packed.symbols if s == "C"]
    assert len(carbons) > 4  # more molecules than the 4 in the cell
    assert _o_neighbours_per_c(packed) == [2] * len(carbons)  # every CO2 whole

    frac = packed.to_ase().get_scaled_positions(wrap=False)
    assert frac.min() < 0.0 and frac.max() > 1.0  # atoms poke outside the box


def test_boundary_completion_replicates_modes():
    dry = to_conventional(_dry_ice())
    evec = np.arange(len(dry) * 3, dtype=float).reshape(len(dry), 3)
    modes = PhononModes([PhononMode(60.0, evec)])

    packed, packed_modes = complete_boundary(dry, modes)
    assert packed_modes[0].eigenvector.shape == (len(packed), 3)
    # every image atom carries its parent's eigenvector (rows drawn from evec)
    for row in packed_modes[0].eigenvector:
        assert any(np.allclose(row, evec[k]) for k in range(len(dry)))


def test_supercell_identity_and_molecule_are_noops():
    nacl = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64))
    same, _ = tile_supercell(nacl, (1, 1, 1))
    assert len(same) == len(nacl)  # 1×1×1 changes nothing

    mol = Structure.empty()
    mol.add_atom("O", [0.0, 0.0, 0.0])
    mol.add_atom("H", [0.96, 0.0, 0.0])
    tiled, _ = tile_supercell(mol, (3, 3, 3))
    assert len(tiled) == 2  # a non-periodic system is never tiled
