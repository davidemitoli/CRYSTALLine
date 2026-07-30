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


def _co_numbers() -> list:
    return list(_structure().numbers)


def _localised_on(atom: int) -> PhononMode:
    """A mode in which only ``atom`` of the two-atom C–O structure moves."""
    eigenvector = np.zeros((2, 3))
    eigenvector[atom] = [1.0, 0.0, 0.0]
    return PhononMode(frequency=500.0, eigenvector=eigenvector)


def test_composition_stays_out_of_the_row_and_in_the_tooltip(qapp):
    """The row is for frequency and activity; composition has no room there."""
    structure = _structure()
    panel = _panel(structure)
    modes = PhononModes([_localised_on(0), _localised_on(1)])

    panel.set_modes(structure.positions, modes, structure.numbers)

    rows = _rows(panel)
    assert rows[0] == "0:    500.00 cm⁻¹"
    assert "C" not in rows[0] and "O" not in rows[1]
    assert "O 100%" in panel.mode_list.item(1).toolTip()


def test_the_summary_line_describes_the_selected_mode(qapp):
    structure = _structure()
    panel = _panel(structure)
    panel.set_modes(structure.positions, PhononModes([_localised_on(0), _localised_on(1)]),
                    structure.numbers)

    panel.mode_list.setCurrentRow(1)
    assert "O 100%" in panel.character_label.text()
    assert "1.0 of 2 atoms move" in panel.character_label.text()

    panel.mode_list.setCurrentRow(0)
    assert "C 100%" in panel.character_label.text()


def test_modes_load_without_a_geometry_to_analyse(qapp):
    """``numbers`` is optional — the panel falls back to bare frequencies."""
    structure = _structure()
    panel = _panel(structure)

    panel.set_modes(structure.positions, _labelled_modes())

    assert len(_rows(panel)) == 4
    assert panel.character(0) is None
    assert panel.character_label.text() == ""
    assert panel.mode_list.item(0).toolTip() == ""


def test_select_mode_reveals_one_the_filter_had_hidden(qapp):
    """Clicking an IR peak must land on that mode even under a Raman filter."""
    structure = _structure()
    panel = _panel(structure)
    panel.set_modes(structure.positions, _labelled_modes(), structure.numbers)
    panel.filter_box.setCurrentIndex(2)  # Raman active: mode 0 (IR only) is hidden

    assert panel.select_mode(0) is True

    assert panel.current_mode_index() == 0
    assert panel.filter_box.currentIndex() == 0  # dropped back to "All modes"


def test_select_mode_rejects_an_index_that_is_not_a_mode(qapp):
    structure = _structure()
    panel = _panel(structure)
    panel.set_modes(structure.positions, _labelled_modes(), structure.numbers)

    assert panel.select_mode(99) is False
    assert panel.select_mode(-1) is False


def test_frequencies_are_exposed_for_matching_a_spectrum_click(qapp):
    structure = _structure()
    panel = _panel(structure)
    assert panel.frequencies() is None  # nothing loaded

    panel.set_modes(structure.positions, _labelled_modes(), structure.numbers)
    assert list(panel.frequencies()) == [100.0, 200.0, 300.0, 400.0]


def test_selecting_a_mode_puts_its_arrows_on_the_view(qapp):
    from crystalline.viz.render_settings import RenderSettings

    structure = _structure()
    # Arrows are off by default; this is the behaviour once they're turned on.
    renderer = StructureRenderer(pv.Plotter(off_screen=True),
                                 RenderSettings(show_mode_arrows=True))
    renderer.set_structure(structure)
    panel = PhononPanel(PhononAnimator(renderer))

    panel.set_modes(structure.positions, PhononModes([_localised_on(1)]), structure.numbers)
    assert renderer._arrow_actor is not None

    panel.clear()  # a file with no phonons must not leave the last mode's arrows
    assert renderer._arrow_actor is None


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


def test_a_costly_frame_yields_the_event_loop_to_the_ui(qapp):
    """A frame is a VTK rebuild plus a synchronous render; on a large cell it
    outlasts the frame interval, and the timer then fires back to back with
    nothing left for input — camera rotation stops tracking the mouse and buttons
    take a visible moment. Beats are skipped until the animation is back within
    its share of the loop."""
    import time

    from crystalline.ui.panels import phonon_panel as pp

    structure = _structure()
    panel = _panel(structure)
    panel.set_modes(structure.positions, _labelled_modes())

    frames = []
    slow_frame = 0.02  # 20 ms, comfortably over _FRAME_DUTY of the 33 ms interval
    panel._tick = lambda: (time.sleep(slow_frame), frames.append(1))

    panel._on_timer()
    assert frames == [1]  # the first beat draws

    panel._on_timer()  # immediately after: still inside the hold-off
    assert frames == [1], "a beat during the hold-off must not draw"

    # The hold-off is proportional to what the frame cost, not a fixed sleep.
    expected = slow_frame * (1.0 / pp._FRAME_DUTY - 1.0)
    assert panel._next_frame_at - time.perf_counter() <= expected + 0.01

    while time.perf_counter() < panel._next_frame_at:
        pass
    panel._on_timer()
    assert frames == [1, 1]  # ...and it resumes once the loop has had its share


def test_cheap_frames_are_not_paced_at_all(qapp):
    """Small structures cost a fraction of the interval, so they keep the full
    ~30 fps: the pacing must not slow down the common case."""
    structure = _structure()
    panel = _panel(structure)
    panel.set_modes(structure.positions, _labelled_modes())

    frames = []
    panel._tick = lambda: frames.append(1)

    for _ in range(5):
        panel._on_timer()
    assert frames == [1] * 5  # every beat drew


def test_play_draws_the_first_frame_without_waiting(qapp):
    """A hold-off left over from a previous run must not delay the next Play."""
    structure = _structure()
    panel = _panel(structure)
    panel.set_modes(structure.positions, _labelled_modes())
    panel._next_frame_at = float("inf")  # as a very slow previous frame would leave it

    panel._play()

    assert panel._next_frame_at == 0.0
    frames = []
    panel._tick = lambda: frames.append(1)
    panel._on_timer()
    assert frames == [1]
