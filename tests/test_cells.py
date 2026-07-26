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
    to_analysis_cell,
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


def test_conventional_keeps_a_centred_slab_in_one_piece():
    """Regression: a slab centred on z = 0 must not be split into two sheets
    ~500 Å apart. Wrapping into the cell box may only touch the periodic axes —
    wrapping the formal 500 Å aperiodic axis flung negative-z atoms to z ≈ 500."""
    from ase.build import fcc111

    at = fcc111("Pt", size=(1, 1, 4), vacuum=0.0)
    at.translate([0, 0, -at.get_positions()[:, 2].mean()])  # centre at z = 0, as CRYSTAL does
    cell = np.asarray(at.get_cell(), dtype=float)
    cell[2] = [0.0, 0.0, 500.0]
    at.set_cell(cell)
    at.set_pbc((True, True, False))

    conv = to_conventional(Structure.from_ase(at))
    thickness = np.ptp(conv.positions[:, 2])
    assert thickness < 20.0  # the whole slab stays together (real thickness ~7 Å)


def test_boundary_completion_never_images_along_aperiodic_axis():
    """A slab must not spawn phantom copies 500 Å away in the vacuum: boundary
    completion may only image along the periodic (in-plane) axes."""
    from ase.build import fcc111

    at = fcc111("Pt", size=(1, 1, 3), vacuum=0.0)
    cell = np.asarray(at.get_cell(), dtype=float)
    cell[2] = [0.0, 0.0, 500.0]  # CRYSTAL's aperiodic placeholder
    at.set_cell(cell)
    at.set_pbc((True, True, False))
    slab = Structure.from_ase(at)

    packed, _ = complete_boundary(slab)
    # every completed atom stays with the slab, never at z ≈ 500
    assert packed.positions[:, 2].max() < 10.0
    original_zmax = slab.positions[:, 2].max()
    assert packed.positions[:, 2].max() <= original_zmax + 1e-6


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


def test_to_analysis_cell_folds_supercell_and_boundary_back():
    """A displayed (tiled + boundary-completed) cell folds back to one clean cell.

    The shown structure is unusable for symmetry analysis directly — a supercell
    box has the wrong shape and boundary images overlap their originals — so the
    Info panel folds it back into the original unit cell before analysing.
    """
    from crystalline.core.crystallography import analyze

    base = as_view(Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64)), CellView.CRYSTALLOGRAPHIC)
    unit_cell = base.to_ase().get_cell()

    analysis, _ = tile_supercell(base, (2, 2, 2))
    shown, _ = complete_boundary(analysis)
    assert len(shown) > len(base)  # tiling + images inflate the shown atom count

    folded = to_analysis_cell(shown, unit_cell)
    assert len(folded) == len(base)  # collapsed back to a single unit cell
    # and the fold recovers the crystal's true point group (m-3m), which neither
    # the supercell (wrong box) nor the boundary-completed cell (overlaps) yields.
    assert analyze(folded).point_group == analyze(base).point_group == "m-3m"


def test_to_analysis_cell_carries_edits_and_lowers_symmetry():
    """An edit made on the shown structure survives the fold and changes symmetry."""
    from crystalline.core.crystallography import analyze

    base = as_view(Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64)), CellView.CRYSTALLOGRAPHIC)
    unit_cell = base.to_ase().get_cell()

    shown, _ = complete_boundary(*tile_supercell(base, (1, 1, 1)))
    edited = Structure.from_ase(shown.to_ase())
    edited.translate_atoms([0], [0.4, 0.0, 0.0])  # nudge one atom off its site

    folded = to_analysis_cell(edited, unit_cell)
    assert analyze(folded).point_group != "m-3m"  # symmetry dropped from cubic


def test_to_analysis_cell_passes_molecules_through():
    mol = Structure.empty()
    mol.add_atom("O", [0.0, 0.0, 0.0])
    mol.add_atom("H", [0.96, 0.0, 0.0])
    out = to_analysis_cell(mol, np.zeros((3, 3)))  # degenerate cell, non-periodic
    assert len(out) == 2
