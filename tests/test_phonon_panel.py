"""The phonon panel's mode list: IR/Raman filtering and selection bookkeeping.

Needs PySide6 (the panel) and pyvista (the renderer the animator drives);
skips otherwise.
"""

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyvista")

import pyvista as pv  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from crystalline.core.phonons import PhononMode, PhononModes  # noqa: E402
from crystalline.core.structure import Structure  # noqa: E402
from crystalline.ui.panels.phonon_panel import PhononPanel  # noqa: E402
from crystalline.viz.phonon_animator import PhononAnimator  # noqa: E402
from crystalline.viz.renderer import StructureRenderer  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _structure() -> Structure:
    s = Structure.empty()
    s.set_cell(np.eye(3) * 8, periodic=False)
    s.add_atom("C", [4, 4, 4])
    s.add_atom("O", [5, 4, 4])
    return s


def _panel(structure: Structure) -> PhononPanel:
    renderer = StructureRenderer(pv.Plotter(off_screen=True))
    renderer.set_structure(structure)
    return PhononPanel(PhononAnimator(renderer))


def _stretch(freq: float, ir=None, raman=None) -> PhononMode:
    return PhononMode(
        frequency=freq,
        eigenvector=np.array([[1.0, 0, 0], [-1.0, 0, 0]]),
        ir_active=ir,
        raman_active=raman,
    )


def _labelled_modes() -> PhononModes:
    return PhononModes(
        [
            _stretch(100.0, ir=True, raman=False),  # 0: IR only
            _stretch(200.0, ir=False, raman=True),  # 1: Raman only
            _stretch(300.0, ir=False, raman=False),  # 2: silent
            _stretch(400.0, ir=True, raman=True),  # 3: both
        ]
    )


def _rows(panel: PhononPanel) -> list[str]:
    return [panel.mode_list.item(i).text() for i in range(panel.mode_list.count())]


def test_filter_narrows_the_list_to_active_modes(qapp):
    panel = _panel(_structure())
    panel.set_modes(_structure().positions, _labelled_modes())

    assert panel.filter_box.isEnabled()
    assert len(_rows(panel)) == 4  # "All modes" by default

    panel.filter_box.setCurrentIndex(1)  # IR active
    assert [r.split(":")[0] for r in _rows(panel)] == ["0", "3"]

    panel.filter_box.setCurrentIndex(2)  # Raman active
    assert [r.split(":")[0] for r in _rows(panel)] == ["1", "3"]

    panel.filter_box.setCurrentIndex(3)  # IR or Raman
    assert [r.split(":")[0] for r in _rows(panel)] == ["0", "1", "3"]


def test_rows_are_labelled_with_their_activity(qapp):
    panel = _panel(_structure())
    panel.set_modes(_structure().positions, _labelled_modes())

    rows = _rows(panel)
    assert rows[0].endswith("[IR]")
    assert rows[1].endswith("[R]")
    assert "[" not in rows[2]  # silent mode carries no tag
    assert rows[3].endswith("[IR, R]")


def test_selection_follows_the_mode_not_the_row(qapp):
    """A filtered list renumbers the rows; the selected *mode* must survive."""
    structure = _structure()
    panel = _panel(structure)
    modes = _labelled_modes()
    panel.set_modes(structure.positions, modes)

    panel.mode_list.setCurrentRow(3)  # the IR+Raman mode, row 3 of 4
    assert panel.current_mode_index() == 3

    panel.filter_box.setCurrentIndex(1)  # IR active -> the same mode is now row 1
    assert panel.mode_list.currentRow() == 1
    assert panel.current_mode_index() == 3
    _equilibrium, mode = panel.current_selection()
    assert mode.frequency == 400.0


def test_filtering_out_the_selection_falls_back_to_the_first_match(qapp):
    structure = _structure()
    panel = _panel(structure)
    panel.set_modes(structure.positions, _labelled_modes())

    panel.mode_list.setCurrentRow(2)  # the silent mode
    panel.filter_box.setCurrentIndex(1)  # IR active: it's gone
    assert panel.current_mode_index() == 0


def test_filter_is_offered_only_when_the_output_labels_the_modes(qapp):
    structure = _structure()
    panel = _panel(structure)

    panel.set_modes(structure.positions, PhononModes([_stretch(100.0), _stretch(200.0)]))
    assert not panel.filter_box.isEnabled()  # no analysis -> would hide everything
    assert len(_rows(panel)) == 2

    panel.set_modes(structure.positions, _labelled_modes())
    assert panel.filter_box.isEnabled()


def test_switching_files_resets_the_filter_and_the_selection(qapp):
    structure = _structure()
    panel = _panel(structure)
    panel.set_modes(structure.positions, _labelled_modes())
    panel.filter_box.setCurrentIndex(2)  # Raman active
    panel.mode_list.setCurrentRow(1)

    # A file without the IR/Raman analysis must not come up empty because of a
    # filter left over from the previous one.
    panel.set_modes(structure.positions, PhononModes([_stretch(50.0), _stretch(60.0)]))
    assert panel.filter_box.currentIndex() == 0
    assert len(_rows(panel)) == 2
    assert panel.current_mode_index() == 0


def test_speed_scales_the_phase_advanced_per_frame(qapp):
    """Speed changes how far the phase moves per timer tick, not the tick rate,
    so the animation stays smooth at every setting."""
    from crystalline.ui.panels import phonon_panel as pp

    structure = _structure()
    panel = _panel(structure)
    panel.set_modes(structure.positions, _labelled_modes())

    panel._tick()
    assert np.isclose(panel._phase, pp._PHASE_STEP)  # 1.0x default

    panel._phase = 0.0
    panel.speed_box.setValue(2.5)
    panel._tick()
    assert np.isclose(panel._phase, 2.5 * pp._PHASE_STEP)

    # the timer keeps its ~30 fps interval whatever the speed
    assert panel._timer.interval() == pp._FRAME_INTERVAL_MS


def test_phase_wraps_and_stop_returns_to_rest(qapp):
    structure = _structure()
    panel = _panel(structure)
    panel.set_modes(structure.positions, _labelled_modes())

    panel.speed_box.setValue(10.0)
    for _ in range(200):
        panel._tick()
    assert 0.0 <= panel._phase < 2 * np.pi  # never runs off

    panel._stop()
    assert panel._phase == 0.0  # next Play starts from rest, not mid-cycle


def test_transport_buttons_are_icons_at_the_top(qapp):
    structure = _structure()
    panel = _panel(structure)
    panel.set_modes(structure.positions, _labelled_modes())
    panel.resize(240, 320)
    panel.show()
    qapp.processEvents()

    # labelled for screen readers / tooltips even though they carry no caption
    assert panel.play_btn.toolTip() == "Play"
    assert panel.stop_btn.toolTip() == "Stop"
    assert not panel.play_btn.icon().isNull() or panel.play_btn.text()

    # above the list, and both on the same row
    assert panel.play_btn.geometry().bottom() <= panel.mode_list.geometry().top()
    assert panel.play_btn.geometry().top() == panel.stop_btn.geometry().top()
    panel.hide()


def test_play_needs_a_selected_mode(qapp):
    structure = _structure()
    panel = _panel(structure)
    panel.set_modes(structure.positions, _labelled_modes())

    panel._play()
    assert panel._timer.isActive()
    panel._stop()
    assert not panel._timer.isActive()

    panel.clear()
    panel._play()  # nothing loaded: must not start a timer against no mode
    assert not panel._timer.isActive()


def test_mode_list_can_shrink_so_every_control_stays_visible(qapp):
    """Regression (Windows): the list's size hint pushed the transport controls
    out of the dock, so they only appeared when the window was maximised. The
    list must give way — everything else has to fit in a short dock."""
    panel = _panel(_structure())
    panel.set_modes(_structure().positions, _labelled_modes())

    assert panel.mode_list.minimumSizeHint().height() <= 100
    assert panel.minimumSizeHint().height() <= 320
    panel.resize(240, 300)
    panel.show()
    qapp.processEvents()

    for widget in (panel.play_btn, panel.stop_btn, panel.filter_box, panel.amp_box,
                   panel.speed_box):
        assert widget.isVisible()
        assert widget.geometry().bottom() <= panel.height()
    panel.hide()
