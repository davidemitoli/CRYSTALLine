"""The view pipeline: cell view → supercell → boundary completion.

``MainWindow`` can't be constructed headless (VTK interactor), so ``_compose_view``
is exercised unbound against a stub window carrying only the attributes it reads.
That matters more than the usual convenience: this is the code path every file
load goes through, and it had no coverage when a change to the cell functions'
return arity broke it for every file *without* ADPs — the common case, and the
one a test on the ADP-carrying path would have missed.
"""

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pymatgen")

from ase.build import bulk  # noqa: E402

from crystalline.core.adp import ADPSet  # noqa: E402
from crystalline.core.cells import CellView  # noqa: E402
from crystalline.core.phonons import PhononMode, PhononModes  # noqa: E402
from crystalline.core.structure import Structure  # noqa: E402
from crystalline.ui.main_window import MainWindow  # noqa: E402
from crystalline.viz.render_settings import RenderSettings  # noqa: E402


class _StubWindow:
    """Only what the view/ADP methods read — no VTK, no widgets."""

    def __init__(self, source: Structure, adps=None, boundary=True) -> None:
        self._source = source
        self._adps = adps
        self._adp_index = None
        self._show_boundary = boundary
        settings = RenderSettings()
        self.viewport = type("V", (), {"renderer": type("R", (), {"settings": settings})()})()

    _displayed_adp = MainWindow._displayed_adp
    _adp_tensors_for_view = MainWindow._adp_tensors_for_view
    _refresh_adp_tensors = MainWindow._refresh_adp_tensors
    _apply_render_settings = MainWindow._apply_render_settings
    _compose_view = MainWindow._compose_view


def _nacl() -> Structure:
    return Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64))


def _modes(structure: Structure) -> PhononModes:
    eigenvector = np.zeros((len(structure), 3))
    eigenvector[0] = [1.0, 0.0, 0.0]
    return PhononModes([PhononMode(frequency=100.0, eigenvector=eigenvector)])


def _adps(structure: Structure) -> ADPSet:
    tensors = np.stack(
        [np.tile(np.eye(3) * u, (len(structure), 1, 1)) for u in (0.005, 0.010)]
    )
    return ADPSet(temperatures=[10.0, 300.0], tensors=tensors)


@pytest.mark.parametrize("view", list(CellView))
@pytest.mark.parametrize("supercell", [(1, 1, 1), (2, 2, 1)])
def test_the_view_composes_and_indexes_every_displayed_atom(view, supercell):
    """Every drawn atom must name the source atom it images — that index is what
    lays a per-atom quantity onto an expanded, tiled, boundary-completed cell.

    (This also covers the regression where the cell functions handed back two
    values for a file with nothing to replicate, where the pipeline unpacks three.)
    """
    source = _nacl()
    window = _StubWindow(source)

    shown, modes, unit_cell, analysis, source_index = window._compose_view(
        view, supercell, _modes(source)
    )

    assert len(shown) >= len(source)
    assert modes is not None and modes[0].eigenvector.shape == (len(shown), 3)
    assert unit_cell.shape == (3, 3)
    assert len(analysis) >= len(source)
    assert source_index.shape == (len(shown),)
    assert set(np.unique(source_index)) <= set(range(len(source)))
    # an index is only useful if it names the *right* atom: same element, always
    assert list(shown.numbers) == [source.numbers[i] for i in source_index]


def test_the_index_survives_with_neither_modes_nor_adps():
    """Loading a plain geometry — no FREQCALC at all — is the most common case."""
    window = _StubWindow(_nacl())

    shown, modes, _cell, _analysis, source_index = window._compose_view(
        CellView.CRYSTALLOGRAPHIC, (1, 1, 1), None
    )

    assert len(shown) > 0
    assert modes is None
    assert len(source_index) == len(shown)


@pytest.mark.parametrize("view", list(CellView))
@pytest.mark.parametrize("supercell", [(1, 1, 1), (2, 2, 1)])
def test_tensors_land_on_the_atoms_they_belong_to(view, supercell):
    source = _nacl()
    window = _StubWindow(source, adps=_adps(source))

    shown, _m, _cell, _analysis, window._adp_index = window._compose_view(
        view, supercell, _modes(source)
    )
    tensors = window._adp_tensors_for_view()

    assert tensors.shape == (len(shown), 3, 3)
    for drawn, parent in zip(tensors, window._adp_index):
        assert np.allclose(drawn, window._displayed_adp()[parent])


def test_the_temperature_setting_chooses_which_tensors_are_shown():
    source = _nacl()
    window = _StubWindow(source, adps=_adps(source))
    assert np.allclose(window._displayed_adp(), np.eye(3) * 0.005)  # index 0 by default

    renderer = window.viewport.renderer
    renderer.settings = RenderSettings(adp_temperature_index=1)
    assert np.allclose(window._displayed_adp(), np.eye(3) * 0.010)

    # an index past the end clamps rather than raising
    renderer.settings = RenderSettings(adp_temperature_index=9)
    assert np.allclose(window._displayed_adp(), np.eye(3) * 0.010)


def test_boundary_completion_can_be_switched_off():
    source = _nacl()
    with_boundary = _StubWindow(source, adps=_adps(source), boundary=True)
    without = _StubWindow(source, adps=_adps(source), boundary=False)

    packed, _m, _c, _a, packed_index = with_boundary._compose_view(
        CellView.PRIMITIVE, (1, 1, 1), None
    )
    plain, _m2, _c2, _a2, plain_index = without._compose_view(
        CellView.PRIMITIVE, (1, 1, 1), None
    )

    assert len(packed) >= len(plain)
    assert len(packed_index) == len(packed)
    assert len(plain_index) == len(plain)


def test_changing_the_temperature_changes_the_drawn_tensors():
    """The regression: the temperature is a *setting*, but the tensors it names
    live on the renderer, so moving the picker has to push new ones. Without
    this the ellipsoids keep whatever shape they had when the file loaded."""
    source = _nacl()
    window = _StubWindow(source, adps=_adps(source))
    _shown, _m, _c, _a, window._adp_index = window._compose_view(
        CellView.PRIMITIVE, (1, 1, 1), None
    )

    at_10 = window._adp_tensors_for_view(0)
    at_300 = window._adp_tensors_for_view(1)

    assert not np.allclose(at_10, at_300)
    assert np.allclose(at_10, np.eye(3) * 0.005)
    assert np.allclose(at_300, np.eye(3) * 0.010)


def test_applying_settings_pushes_the_new_temperature_to_the_renderer():
    source = _nacl()
    pushed = []

    class _Renderer:
        settings = RenderSettings()

        def set_adp_tensors(self, tensors, redraw=True):
            pushed.append((None if tensors is None else tensors.copy(), redraw))

        def set_settings(self, settings):
            type(self).settings = settings

    renderer = _Renderer()
    window = _StubWindow(source, adps=_adps(source))
    window.viewport = type("V", (), {"renderer": renderer})()
    _shown, _m, _c, _a, window._adp_index = window._compose_view(
        CellView.PRIMITIVE, (1, 1, 1), None
    )

    window._apply_render_settings(RenderSettings(adp_temperature_index=1))
    assert len(pushed) == 1, "a temperature change must re-push the tensors"
    tensors, redraw = pushed[0]
    assert np.allclose(tensors[0], np.eye(3) * 0.010)
    assert redraw is False, "staged, so applying the settings is the only rebuild"

    # a settings change that leaves the temperature alone must not re-push
    window._apply_render_settings(RenderSettings(adp_temperature_index=1, atom_scale=0.9))
    assert len(pushed) == 1
