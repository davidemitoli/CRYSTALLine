"""Loading structures from the file formats the GUI accepts."""

import numpy as np
import pytest

pytest.importorskip("pymatgen")  # the CIF reader (CifParser) needs pymatgen

from crystalline.crystalio.loader import load, load_structure  # noqa: E402

# A minimal, self-contained CIF: rock-salt NaCl in its cubic conventional cell.
_NACL_CIF = """
data_NaCl
_cell_length_a     5.64
_cell_length_b     5.64
_cell_length_c     5.64
_cell_angle_alpha  90
_cell_angle_beta   90
_cell_angle_gamma  90
_symmetry_space_group_name_H-M   'F m -3 m'
_symmetry_Int_Tables_number      225
loop_
_symmetry_equiv_pos_as_xyz
  'x,y,z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Na Na 0.0 0.0 0.0
Cl Cl 0.5 0.5 0.5
"""


def _write_cif(tmp_path) -> str:
    path = tmp_path / "nacl.cif"
    path.write_text(_NACL_CIF)
    return str(path)


def test_load_structure_reads_a_cif(tmp_path):
    structure = load_structure(_write_cif(tmp_path))
    assert structure.is_periodic
    assert set(structure.symbols) == {"Na", "Cl"}
    # the CIF's cubic cell survives the round-trip through pymatgen/ase
    a, b, c, alpha, beta, gamma = structure.cellpar
    assert np.allclose([a, b, c], 5.64, atol=1e-3)
    assert np.allclose([alpha, beta, gamma], 90.0, atol=1e-3)


def test_load_of_a_cif_carries_no_phonon_modes(tmp_path):
    result = load(_write_cif(tmp_path))
    assert result.structure.is_periodic
    assert result.modes is None  # a CIF is geometry only — never scanned as a CRYSTAL out
    assert not result.has_phonons


def test_per_mode_normalises_crystalclear_activity_arrays():
    """CRYSTALClear leaves IR/Raman/intens empty when the output has no
    selection-rule analysis; the modes must then read "unknown" rather than
    "inactive", which would let the panel's filter hide every mode."""
    from crystalline.crystalio.loader import _per_mode

    assert _per_mode([True, False, True], 3) == [True, False, True]
    assert _per_mode([], 3) == [None, None, None]  # no analysis in the output
    assert _per_mode(None, 2) == [None, None]
    assert _per_mode([True, False], 3) == [None, None, None]  # misaligned -> unlabelled
    assert _per_mode([1.5, 0.0], 2, cast=float) == [1.5, 0.0]
    assert _per_mode([np.nan, 2.0], 2, cast=float) == [None, 2.0]  # NaN'd imaginary mode
