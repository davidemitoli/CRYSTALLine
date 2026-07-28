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


class _StubOutput:
    """Stands in for CRYSTALClear's Crystal_output — the row builder only ever
    calls ``get_calculation_info``."""

    def __init__(self, info):
        self._info = info

    def get_calculation_info(self):
        return self._info


_PBE_D3 = {
    "code": "CRYSTAL17",
    "run_type": ["CPHF", "FREQCALC"],
    "terminated": True,
    "method": "DFT",
    "exchange": "PERDEW-BURKE-ERNZERHOF",
    "correlation": "PERDEW-BURKE-ERNZERHOF",
    "hybrid_exchange": None,
    "dispersion": "DFT-D3(BJ)",
    "shell_type": "RESTRICTED CLOSED SHELL",
    "shrink": [8, 8, 8],
    "n_kpoints_ibz": 125,
    "n_ao": 444,
    "n_shells": 236,
    "n_electrons": 160,
    "n_core_electrons": 64,
    "n_symmops": 4,
    "toldee": 9,
    "tolinteg": [8, 8, 20, 20, 20],
    "grid_points": 399573,
}


def test_setup_rows_describe_the_calculation():
    from crystalline.crystalio.loader import _setup_rows

    rows = _setup_rows(_StubOutput(_PBE_D3))

    assert rows["Code"] == "CRYSTAL17"
    assert rows["Task"] == "CPHF + FREQCALC"
    assert rows["Method"] == "DFT"
    # CRYSTAL's full spelling is shortened to the name people use
    assert rows["Exchange"] == "PBE"
    assert rows["Correlation"] == "PBE"
    # the "DFT-" prefix is dropped: the row is already labelled Dispersion
    assert rows["Dispersion"] == "D3(BJ)"
    assert rows["Shell type"] == "Restricted closed shell"
    # mesh and count are separate rows — together they overflowed the dock
    mesh = "\u00a0\u00d7\u00a0".join("888")  # non-breaking spaces: the mesh never wraps mid-value
    assert rows["k-point mesh"] == mesh
    assert rows["k points (IBZ)"] == "125"
    assert rows["Basis functions"] == "444"
    assert rows["Electrons/cell"] == "160 (64 core)"
    assert rows["SCF tolerance"] == "10⁻⁹ Ha"
    assert rows["TOLINTEG"] == "8 8 20 20 20"
    assert rows["DFT grid"] == "399,573"
    assert "Termination" not in rows  # only flagged when the run ended badly


def test_setup_rows_report_a_hybrid_and_an_unlisted_functional():
    from crystalline.crystalio.loader import _setup_rows

    info = dict(_PBE_D3, hybrid_exchange=25.0, exchange="SOME-NEW-FUNCTIONAL")
    rows = _setup_rows(_StubOutput(info))

    assert rows["Method"] == "DFT (hybrid, 25% exact exchange)"
    # an acronym is only substituted when it's known; otherwise CRYSTAL's wording
    assert rows["Exchange"] == "SOME-NEW-FUNCTIONAL"


def test_setup_rows_omit_what_the_run_did_not_report():
    """A Hartree-Fock run has no functional, and a molecule no k points; those
    rows must be absent rather than blank or wrong."""
    from crystalline.crystalio.loader import _setup_rows

    rows = _setup_rows(
        _StubOutput(
            {
                "code": "CRYSTAL23",
                "method": "Hartree-Fock",
                "run_type": [],
                "terminated": True,
                "exchange": None,
                "correlation": None,
                "shrink": None,
                "grid_points": None,
            }
        )
    )

    assert rows["Method"] == "Hartree-Fock"
    for absent in ("Exchange", "Correlation", "k-point mesh", "k points (IBZ)",
                   "DFT grid", "Task"):
        assert absent not in rows


def test_setup_rows_flag_a_run_that_did_not_finish():
    from crystalline.crystalio.loader import _setup_rows

    rows = _setup_rows(_StubOutput(dict(_PBE_D3, terminated=False)))
    assert rows["Termination"] == "did not terminate normally"


def test_setup_rows_survive_an_older_crystalclear():
    """``get_calculation_info`` landed in CRYSTALClear 0.2.16; against an older
    one the panel must fall back to the computed-property rows, not error."""
    from crystalline.crystalio.loader import _setup_rows

    class _Old:
        pass

    assert _setup_rows(_Old()) == {}


def test_dispersion_drops_the_dft_prefix():
    from crystalline.crystalio.loader import _dispersion

    assert _dispersion("DFT-D3(BJ)") == "D3(BJ)"
    assert _dispersion("DFT-D3") == "D3"
    assert _dispersion("DFT-D2") == "D2"
    # older outputs name only the author, so there is no version to report
    assert _dispersion("GRIMME DISPERSION") == "yes"
    assert _dispersion(None) is None
