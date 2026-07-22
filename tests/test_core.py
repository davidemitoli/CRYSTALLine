"""Unit tests for the Qt-free domain core (no display needed)."""

import numpy as np
import pytest

from crystalline.core.structure import Structure
from crystalline.core.phonons import PhononMode, PhononModes, displaced_positions


def test_add_atoms_batch_appends_and_fires_once():
    s = Structure.empty()
    s.add_atom("C", [0.0, 0.0, 0.0])
    events = []
    s.add_listener(lambda st: events.append(len(st)))

    new = s.add_atoms(["O", "H", "H"], [[1, 0, 0], [2, 0, 0], [0, 1, 0]])
    assert new == [1, 2, 3]
    assert len(s) == 4
    assert s.symbols == ["C", "O", "H", "H"]
    assert events == [4]  # a single notification for the whole batch

    assert s.add_atoms([], []) == []  # empty is a no-op
    with pytest.raises(ValueError):
        s.add_atoms(["C"], [[0, 0, 0], [1, 1, 1]])  # length mismatch


def test_add_move_remove_atom_notifies():
    s = Structure.empty()
    events = []
    s.add_listener(lambda st: events.append(len(st)))

    i = s.add_atom("C", [0.0, 0.0, 0.0])
    j = s.add_atom("O", [1.2, 0.0, 0.0])
    assert (i, j) == (0, 1)
    assert len(s) == 2
    assert s.symbols == ["C", "O"]

    s.move_atom(1, [1.5, 0.0, 0.0])
    assert np.allclose(s.positions[1], [1.5, 0.0, 0.0])

    s.set_symbol(0, "N")
    assert s.symbols[0] == "N"

    s.remove_atom(0)
    assert len(s) == 1 and s.symbols == ["O"]

    # each mutating call fired exactly one notification
    assert events == [1, 2, 2, 2, 1]


def test_invalid_symbol_and_index():
    s = Structure.empty()
    with pytest.raises(ValueError):
        s.add_atom("Xx", [0, 0, 0])
    s.add_atom("H", [0, 0, 0])
    with pytest.raises(IndexError):
        s.move_atom(5, [0, 0, 0])


def test_batch_edits_translate_duplicate_set_remove():
    s = Structure.empty()
    for i, el in enumerate(["C", "O", "N", "H"]):
        s.add_atom(el, [i, 0, 0])
    events = []
    s.add_listener(lambda st: events.append(len(st)))

    s.translate_atoms([0, 2], [0.0, 1.0, 0.0])
    assert np.allclose(s.positions[0], [0, 1, 0])
    assert np.allclose(s.positions[2], [2, 1, 0])
    assert np.allclose(s.positions[1], [1, 0, 0])  # untouched

    new = s.duplicate_atoms([1, 3], offset=[0.0, 0.0, 5.0])
    assert new == [4, 5]
    assert s.symbols[4] == "O" and s.symbols[5] == "H"
    assert np.allclose(s.positions[4], [1, 0, 5])

    s.set_symbols([0, 4], "S")
    assert s.symbols[0] == "S" and s.symbols[4] == "S"

    s.remove_atoms([5, 0])  # order-independent, high-to-low internally
    assert len(s) == 4
    assert s.symbols == ["O", "N", "H", "S"]

    # one notification per batch action (4 actions)
    assert events == [4, 6, 6, 4]


def test_batch_edits_validate_and_ignore_empty():
    s = Structure.empty()
    s.add_atom("C", [0, 0, 0])
    calls = []
    s.add_listener(lambda st: calls.append(1))

    with pytest.raises(IndexError):
        s.remove_atoms([0, 9])  # 9 out of range -> nothing removed
    assert len(s) == 1
    with pytest.raises(ValueError):
        s.set_symbols([0], "Zz")

    # empty selection is a no-op that does not notify
    s.translate_atoms([], [1, 2, 3])
    assert s.duplicate_atoms([]) == []
    assert calls == []


def test_phonon_mode_shape_validation():
    with pytest.raises(ValueError):
        PhononMode(frequency=100.0, eigenvector=np.zeros((4,)))  # not (N, 3)
    m = PhononMode(frequency=-2.0, eigenvector=np.zeros((3, 3)))
    assert m.is_imaginary and m.n_atoms == 3


def test_set_lattice_parameters_scales_atoms_and_notifies():
    s = Structure.empty()
    s.set_cell(np.diag([4.0, 4.0, 4.0]), periodic=True)
    s.add_atom("Na", [0.0, 0.0, 0.0])
    s.add_atom("Cl", [2.0, 2.0, 2.0])  # fractional (0.5, 0.5, 0.5)
    frac_before = np.linalg.solve(s.cell.T, s.positions.T).T

    events = []
    s.add_listener(lambda st: events.append(1))
    s.set_lattice_parameters(6.0, 6.0, 6.0, 90.0, 90.0, 90.0)

    assert np.allclose(s.cellpar, [6.0, 6.0, 6.0, 90.0, 90.0, 90.0])
    # atoms moved with the cell: fractional coordinates preserved
    frac_after = np.linalg.solve(s.cell.T, s.positions.T).T
    assert np.allclose(frac_before, frac_after)
    assert np.allclose(s.positions[1], [3.0, 3.0, 3.0])  # 0.5 * 6.0
    assert events == [1]  # exactly one notification

    # a non-orthogonal angle is applied faithfully
    s.set_lattice_parameters(6.0, 6.0, 6.0, 90.0, 90.0, 120.0)
    assert np.isclose(s.cellpar[5], 120.0)


def test_displaced_positions_at_key_phases():
    eq = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    evec = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
    mode = PhononMode(frequency=50.0, eigenvector=evec)

    # phase 0 -> sin(0)=0 -> equilibrium
    assert np.allclose(displaced_positions(eq, mode, amplitude=1.0, phase=0.0), eq)
    # phase pi/2 -> sin=1 -> full displacement
    peak = displaced_positions(eq, mode, amplitude=0.5, phase=np.pi / 2)
    assert np.allclose(peak, eq + 0.5 * evec)


def test_phonon_modes_collection():
    modes = PhononModes(
        [PhononMode(10.0, np.zeros((2, 3))), PhononMode(-5.0, np.zeros((2, 3)))]
    )
    assert len(modes) == 2
    assert np.allclose(modes.frequencies, [10.0, -5.0])
    assert modes[1].is_imaginary
