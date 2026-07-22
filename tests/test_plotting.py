"""The CRYSTALClear plotting adapter: registry shape, figure normalisation,
and error handling. The end-to-end read→plot path needs both CRYSTALClear and a
real data file, so that part is skipped when either is missing.
"""

import glob
import os

import pytest

from crystalline.crystalio import plotting


def test_registry_is_well_formed():
    kinds = plotting.available_plots()
    assert kinds, "expected at least one plot kind"
    keys = [k.key for k in kinds]
    assert len(keys) == len(set(keys)), "plot keys must be unique"
    for k in kinds:
        assert k.label and k.caption and k.file_filter
        assert k.source in ("output", "data")
        assert callable(k.build)
    # electronic/phonon dispersion come from data files; spectra/elastic/EOS
    # come straight from the CRYSTAL output.
    assert {"electron_band", "electron_dos", "phonon_band", "phonon_dos"} <= set(keys)
    assert {"ir", "raman", "ela_young", "eos"} <= set(keys)


def test_output_plots_read_the_out_file_and_data_plots_need_a_file():
    by_key = {k.key: k for k in plotting.available_plots()}
    assert by_key["ir"].source == "output"
    assert by_key["raman"].source == "output"
    assert by_key["ela_young"].source == "output"
    assert by_key["eos"].source == "output"
    assert by_key["electron_band"].source == "data"
    assert by_key["xrd"].source == "data"
    # elastic surfaces are grouped into a submenu
    assert by_key["ela_young"].group == "Elastic properties"
    assert by_key["ir"].group is None


def test_to_figure_normalises_return_shapes():
    from matplotlib.figure import Figure

    fig = Figure()
    ax = fig.add_subplot(111)
    assert plotting._to_figure(fig) is fig
    assert plotting._to_figure((fig, ax)) is fig  # (fig, ax) tuple
    assert plotting._to_figure(([fig], [ax], [None])) is fig  # list-of-lists (ela)


def test_missing_crystalclear_raises_plot_unavailable(monkeypatch):
    # Simulate CRYSTALClear being absent: _require_crystalclear must raise.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("CRYSTALClear"):
            raise ImportError("simulated missing CRYSTALClear")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(plotting.PlotUnavailable):
        plotting._require_crystalclear()
    assert plotting.crystalclear_available() is False


def test_output_availability_empty_without_a_file():
    # No output loaded -> nothing to enable (and no CRYSTALClear call needed).
    assert plotting.output_availability(None) == set()
    assert plotting.output_availability("") == set()


def test_bad_file_raises(tmp_path):
    pytest.importorskip("CRYSTALClear")
    kind = {k.key: k for k in plotting.available_plots()}["electron_dos"]
    missing = str(tmp_path / "nope.DAT")
    with pytest.raises(Exception):  # FileNotFoundError / parse error, surfaced to UI
        kind.build(missing)


# Optional end-to-end: only runs where CRYSTALClear and the sample files exist.
_F25 = glob.glob(os.path.expanduser("~/Desktop/PyCrystal/ZnO/*_bands.f25"))
_ELA = glob.glob(os.path.expanduser("~/Desktop/PyCrystal/coesite/coesite_ela.out"))
_FREQ = glob.glob(
    os.path.expanduser("~/Desktop/PyCrystal/anharmonic_freq/thiourea_*freqcalc*.out")
)


@pytest.mark.skipif(not _F25, reason="no sample .f25 file available")
def test_phonon_band_end_to_end():
    pytest.importorskip("CRYSTALClear")
    from matplotlib.figure import Figure

    kind = {k.key: k for k in plotting.available_plots()}["phonon_band"]
    fig = kind.build(_F25[0])
    assert isinstance(fig, Figure)
    assert fig.axes  # something was drawn


@pytest.mark.skipif(not _ELA, reason="no sample elastic .out available")
def test_elastic_from_output_end_to_end():
    pytest.importorskip("CRYSTALClear")
    from matplotlib.figure import Figure

    kind = {k.key: k for k in plotting.available_plots()}["ela_young"]
    fig = kind.build(_ELA[0])  # reads the .out directly — no separate data file
    assert isinstance(fig, Figure)
    assert fig.axes


@pytest.mark.skipif(not (_ELA and _FREQ), reason="need elastic + freq sample .out")
def test_output_availability_discriminates_by_file():
    pytest.importorskip("CRYSTALClear")
    # The elastic run exposes the elastic surfaces but no IR/Raman…
    ela = plotting.output_availability(_ELA[0])
    assert "ela_young" in ela
    assert "ir" not in ela and "raman" not in ela
    # …and the frequency run is the other way round.
    freq = plotting.output_availability(_FREQ[0])
    assert {"ir", "raman"} <= freq
    assert "ela_young" not in freq


@pytest.mark.skipif(not _FREQ, reason="no sample frequency .out available")
def test_ir_and_raman_from_output_end_to_end():
    pytest.importorskip("CRYSTALClear")
    from matplotlib.figure import Figure

    by_key = {k.key: k for k in plotting.available_plots()}
    for key in ("ir", "raman"):
        fig = by_key[key].build(_FREQ[0])  # straight from the CRYSTAL output
        assert isinstance(fig, Figure) and fig.axes
