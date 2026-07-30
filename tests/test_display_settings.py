"""The Display dock round-trips every render setting.

The panel rebuilds a whole :class:`RenderSettings` from its widgets on each
change, so a field that gets a widget but is left out of that rebuild silently
reverts to its default the moment anything else is touched. One round-trip over
a fully non-default settings object catches that for every field at once,
including ones added later.
"""

import os
from dataclasses import fields

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from crystalline.ui.panels.display_settings import DisplayPanel  # noqa: E402
from crystalline.viz.render_settings import RenderSettings  # noqa: E402

# Every field set away from its default, and within each widget's range.
_TWEAKED = RenderSettings(
    atom_scale=0.75,
    atom_opacity=0.6,
    atom_colors=((8, "#123456"),),
    show_bonds=False,
    bond_radius=0.2,
    bond_tolerance=1.4,
    show_hydrogen_bonds=False,
    show_cell=False,
    show_lattice_vectors=False,
    show_adp_ellipsoids=True,
    adp_probability=0.9,
    adp_opacity=0.4,
    adp_temperature_index=2,
    show_mode_arrows=True,
    mode_arrow_scale=2.5,
    mode_arrow_color="#00ff00",
    show_atom_labels=True,
    atom_label_size=22,
    show_polyhedra=False,
    polyhedra_opacity=0.8,
    polyhedra_min_vertices=6,
    measure_point_color="#111111",
    measure_line_color="#222222",
    measure_plane_color="#333333",
    background_color="#444444",
    parallel_projection=False,
    show_orientation_axes=True,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_every_setting_survives_a_trip_through_the_panel(qapp):
    emitted = []
    panel = DisplayPanel(_TWEAKED, emitted.append)
    # The ellipsoid controls are driven by the loaded file: with no temperatures
    # the panel forces them off, so give it some before checking the round-trip.
    panel.set_adp_temperatures(["10 K", "150 K", "300 K"])
    panel._show_adp.setChecked(_TWEAKED.show_adp_ellipsoids)
    emitted.clear()

    panel._emit()

    assert len(emitted) == 1
    for field in fields(RenderSettings):
        assert getattr(emitted[0], field.name) == getattr(_TWEAKED, field.name), field.name


def test_loading_settings_does_not_emit(qapp):
    """Building the panel must not look like a user change, or opening the dock
    would push a redraw before anything was touched."""
    emitted = []
    DisplayPanel(_TWEAKED, emitted.append)

    assert emitted == []


def test_toggling_a_control_reports_the_change(qapp):
    emitted = []
    panel = DisplayPanel(_TWEAKED, emitted.append)

    panel._show_arrows.setChecked(False)

    assert emitted[-1].show_mode_arrows is False
    assert emitted[-1].mode_arrow_scale == _TWEAKED.mode_arrow_scale  # nothing else moved


def test_ellipsoids_cannot_be_switched_on_without_adp_data(qapp):
    """Most frequency runs carry no ADPs; the controls say so rather than
    offering a toggle that would draw nothing."""
    emitted = []
    panel = DisplayPanel(_TWEAKED, emitted.append)
    panel.set_adp_temperatures(["10 K", "300 K"])
    panel._show_adp.setChecked(True)
    emitted.clear()

    panel.set_adp_temperatures([])  # a file with no ADP section

    assert not panel._show_adp.isChecked()
    assert emitted[-1].show_adp_ellipsoids is False  # stale ellipsoids cleared
    assert "no ADP data" in panel._show_adp.toolTip()
    assert not panel._adp_temp.isEnabled()


def test_the_temperature_picker_follows_the_loaded_file(qapp):
    emitted = []
    panel = DisplayPanel(_TWEAKED, emitted.append)

    panel.set_adp_temperatures(["10 K", "82.5 K", "300 K"])
    assert panel._adp_temp.isEnabled()
    assert [panel._adp_temp.itemText(i) for i in range(panel._adp_temp.count())] == [
        "10 K", "82.5 K", "300 K"
    ]
    # the index the settings asked for is honoured on the first file
    assert panel._adp_temp.currentIndex() == _TWEAKED.adp_temperature_index

    emitted.clear()
    panel._adp_temp.setCurrentIndex(1)
    assert emitted[-1].adp_temperature_index == 1

    # A file with fewer temperatures must not leave the index out of range.
    panel.set_adp_temperatures(["150 K"])
    panel._emit()
    assert emitted[-1].adp_temperature_index == 0


def test_opening_a_file_with_adps_shows_them():
    """A run that went to the trouble of computing ADPs is one whose ADPs you
    want to see, so the loader switches the ellipsoids on."""
    emitted = []
    panel = DisplayPanel(RenderSettings(), emitted.append)
    assert not panel._show_adp.isChecked()  # nothing loaded yet

    panel.set_adp_temperatures(["10 K", "300 K"], autoshow=True)

    assert panel._show_adp.isChecked()
    assert emitted[-1].show_adp_ellipsoids is True  # and they are drawn, not just ticked


def test_a_view_change_does_not_overrule_switching_them_off():
    """Cell view, supercell and boundary changes re-list the same temperatures;
    none of them is a reason to turn the ellipsoids back on."""
    emitted = []
    panel = DisplayPanel(RenderSettings(), emitted.append)
    panel.set_adp_temperatures(["10 K", "300 K"], autoshow=True)
    panel._show_adp.setChecked(False)  # the user doesn't want them

    panel.set_adp_temperatures(["10 K", "300 K"])  # a supercell change, say

    assert not panel._show_adp.isChecked()


def test_opening_a_file_without_adps_leaves_them_off():
    emitted = []
    panel = DisplayPanel(RenderSettings(), emitted.append)

    panel.set_adp_temperatures([], autoshow=True)

    assert not panel._show_adp.isChecked()
    assert not panel._adp_temp.isEnabled()


def test_the_default_probability_is_the_one_the_picker_starts_on():
    panel = DisplayPanel(RenderSettings(), lambda _s: None)

    assert RenderSettings().adp_probability == 0.99
    assert panel._adp_probability.currentText() == "99 %"


def test_a_slider_drag_is_coalesced_into_one_change(qapp):
    """A slider spans 1000 steps, and each step can cost the renderer a full scene
    rebuild — dragging one used to emit hundreds of times. The continuous controls
    wait for the drag to settle; discrete ones must stay immediate."""
    emitted = []
    panel = DisplayPanel(RenderSettings(), emitted.append)

    for value in (0.55, 0.60, 0.65, 0.70):
        panel._atom_scale.setValue(value)
    assert emitted == []  # nothing reported while the drag is still moving

    qapp.processEvents()
    _wait_for_settle(qapp, panel)

    assert len(emitted) == 1  # one change for the whole drag...
    assert emitted[-1].atom_scale == pytest.approx(0.70)  # ...carrying where it ended


def test_a_discrete_control_reports_immediately_and_flushes_a_pending_drag(qapp):
    """Waiting would add latency to a single click for nothing. And a slider tick
    still sitting on the timer must not overtake it, or the older settings would
    arrive last and undo the click."""
    emitted = []
    panel = DisplayPanel(RenderSettings(), emitted.append)

    panel._atom_scale.setValue(0.8)  # pending, not yet reported
    panel._show_bonds.setChecked(False)  # a click: immediate

    assert len(emitted) == 1
    assert emitted[-1].show_bonds is False
    assert emitted[-1].atom_scale == pytest.approx(0.8)  # the pending value came along

    _wait_for_settle(qapp, panel)
    assert len(emitted) == 1  # the flushed tick does not fire again afterwards


def _wait_for_settle(qapp, panel, timeout_ms: int = 2000) -> None:
    """Pump the event loop until the panel's debounce timer has fired."""
    from time import monotonic

    deadline = monotonic() + timeout_ms / 1000.0
    while panel._settle.isActive() and monotonic() < deadline:
        qapp.processEvents()
    qapp.processEvents()
