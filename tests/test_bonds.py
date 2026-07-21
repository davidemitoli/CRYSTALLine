"""Tests for chemically-aware connectivity (CrystalNN); needs pymatgen."""

import pytest

from crystalline.core.bonds import connectivity
from crystalline.core.structure import Structure

pytest.importorskip("pymatgen")
from ase.build import bulk  # noqa: E402
from ase.data import chemical_symbols  # noqa: E402


def test_connectivity_cation_polyhedra_only():
    # NaCl: Na is the cation centre (Na-Cl octahedra); Cl (anion) gets none.
    nacl = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64, cubic=True))
    conn = connectivity(nacl, min_vertices=4)
    assert conn is not None
    assert len(conn.bonds) > 0
    centers = {chemical_symbols[z] for z, _ in conn.polyhedra}
    assert centers == {"Na"}  # only the cation is a polyhedron centre
    # each polyhedron is an octahedron (6 ligands)
    assert all(len(pts) == 6 for _, pts in conn.polyhedra)


def test_connectivity_none_for_non_periodic():
    mol = Structure.empty()
    mol.add_atom("C", [0, 0, 0])
    mol.add_atom("O", [1.16, 0, 0])
    assert connectivity(mol) is None
