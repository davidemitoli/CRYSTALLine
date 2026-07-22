"""Tests for crystallographic analysis (needs pymatgen)."""

import pytest

from crystalline.core.crystallography import analyze
from crystalline.core.structure import Structure

pytest.importorskip("pymatgen")
from ase.build import bulk  # noqa: E402


def test_analyze_fcc_nacl():
    info = analyze(Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64)))
    assert info.periodic
    assert info.formula == "NaCl"
    assert info.space_group_symbol == "Fm-3m"
    assert info.space_group_number == 225
    assert info.crystal_system == "cubic"
    # conventional cell params (not the 2-atom primitive) are reported
    assert abs(info.a - 5.64) < 1e-3 and abs(info.alpha - 90.0) < 1e-3
    assert info.z == 4
    assert info.density and info.density > 0
    # every reported row is a (label, value) string pair
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in info.rows())


def test_analyze_molecule_is_non_periodic():
    mol = Structure.empty()
    mol.add_atom("O", [0.0, 0.0, 0.0])
    mol.add_atom("H", [0.96, 0.0, 0.0])
    mol.add_atom("H", [-0.24, 0.93, 0.0])

    info = analyze(mol)
    assert not info.periodic
    assert info.space_group_symbol is None
    assert "0D" in info.dimensionality or "Molecule" in info.dimensionality
    assert info.n_atoms == 3


def test_analyze_empty_structure():
    info = analyze(Structure.empty())
    assert info.n_atoms == 0
    assert info.rows()  # still yields at least the placeholder rows
