"""The Geometry panel: measuring the selection, and the atom-edit gate."""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from crystalline.core import measure as M  # noqa: E402
from crystalline.core.structure import Structure  # noqa: E402
from crystalline.ui.panels.geometry_panel import GeometryPanel  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _water() -> Structure:
    s = Structure.empty()
    s.add_atom("O", [0.0, 0.0, 0.0])
    s.add_atom("H", [0.9572, 0.0, 0.0])
    s.add_atom("H", [-0.2400, 0.9266, 0.0])
    return s


def test_measuring_two_atoms_lists_a_distance(qapp):
    panel = GeometryPanel(_water())
    emitted = []
    panel.annotations_changed.connect(lambda anns: emitted.append(list(anns)))

    panel.set_selection([0, 1])
    panel._measure_selection()

    assert len(panel.measurements()) == 1
    result = panel.measurements()[0]
    assert result.kind == M.DISTANCE
    assert result.value == pytest.approx(0.9572, abs=1e-4)
    # new measurements are shown in 3D straight away
    assert emitted and len(emitted[-1]) == 1


def test_measuring_three_atoms_gives_an_angle_and_a_plane_on_request(qapp):
    panel = GeometryPanel(_water())
    panel.set_selection([1, 0, 2])

    panel._measure_selection()
    assert panel.measurements()[-1].kind == M.ANGLE
    assert panel.measurements()[-1].value == pytest.approx(104.5, abs=0.05)

    panel._measure_plane()  # the same three atoms, as a plane instead
    assert panel.measurements()[-1].kind == M.PLANE


def test_unticking_a_measurement_hides_it_without_deleting_it(qapp):
    panel = GeometryPanel(_water())
    shown = []
    panel.annotations_changed.connect(lambda anns: shown.append(list(anns)))
    panel.set_selection([0, 1])
    panel._measure_selection()
    assert len(panel.shown_annotations()) == 1

    panel._items()[0].setCheckState(Qt.Unchecked)
    assert panel.shown_annotations() == []       # nothing drawn
    assert len(panel.measurements()) == 1        # but still listed
    assert shown[-1] == []                       # and the viewport was told


def test_clearing_and_removing_measurements(qapp):
    panel = GeometryPanel(_water())
    panel.set_selection([0, 1])
    panel._measure_selection()
    panel.set_selection([1, 0, 2])
    panel._measure_selection()
    assert len(panel.measurements()) == 2

    panel._list.item(0).setSelected(True)
    panel._remove_selected_measurements()
    assert len(panel.measurements()) == 1
    assert panel.measurements()[0].kind == M.ANGLE  # the right one went

    panel.clear_measurements()
    assert panel.measurements() == [] and panel.shown_annotations() == []


def test_a_new_structure_drops_stale_measurements(qapp):
    panel = GeometryPanel(_water())
    panel.set_selection([0, 1])
    panel._measure_selection()

    panel.set_structure(_water())  # e.g. a supercell or a freshly loaded file
    assert panel.measurements() == []


def test_atom_tools_are_gated_on_editing_mode(qapp):
    """The trap this panel exists to expose: selecting atoms is not enough —
    editing mode must be on before Delete does anything."""
    panel = GeometryPanel(_water())
    panel.set_selection([0, 1])
    assert not panel._delete_btn.isEnabled()  # selected, but editing is off

    panel.set_editing_enabled(True)
    assert panel._delete_btn.isEnabled()
    assert panel._duplicate_btn.isEnabled()
    assert panel._set_element_btn.isEnabled()
    assert panel._add_btn.isEnabled()  # adding needs no selection

    panel.set_selection([])
    assert not panel._delete_btn.isEnabled()  # editing on, nothing selected
    assert panel._add_btn.isEnabled()


def test_the_editing_checkbox_reflects_and_drives_the_edit_menu(qapp):
    panel = GeometryPanel(_water())
    toggles = []
    panel.editing_toggled.connect(toggles.append)

    # driven from the panel -> MainWindow is told
    panel._edit_check.setChecked(True)
    assert toggles == [True]

    # driven from the menu -> the checkbox follows, without echoing back
    panel.set_editing_enabled(False)
    assert panel._edit_check.isChecked() is False
    assert toggles == [True]


def test_edit_buttons_emit_intent_rather_than_editing_directly(qapp):
    """MainWindow runs the same slots the Edit menu uses, so undo behaves."""
    panel = GeometryPanel(_water())
    panel.set_editing_enabled(True)
    panel.set_selection([0])

    seen = []
    panel.delete_requested.connect(lambda: seen.append("delete"))
    panel.duplicate_requested.connect(lambda: seen.append("duplicate"))
    panel.translate_requested.connect(lambda: seen.append("translate"))
    panel.set_element_requested.connect(lambda s: seen.append(("element", s)))
    panel.add_atom_requested.connect(lambda s: seen.append(("add", s)))

    panel._element.setCurrentText("Fe")
    panel._delete_btn.click()
    panel._duplicate_btn.click()
    panel._translate_btn.click()
    panel._set_element_btn.click()
    panel._add_btn.click()

    assert seen == ["delete", "duplicate", "translate", ("element", "Fe"), ("add", "Fe")]
    assert len(_water()) == 3  # the panel itself changed nothing


def test_setting_a_per_item_colour(qapp, monkeypatch):
    """The 'Colour…' button recolours only the selected measurement(s); each item
    keeps its own colour, and the change reaches the emitted annotations."""
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import QColorDialog

    panel = GeometryPanel(_water())
    panel.set_selection([0, 1])
    panel._measure_selection()  # a distance
    panel.set_selection([0, 1, 2])
    panel._measure_plane()      # a plane
    assert [m.color for m in panel.measurements()] == [None, None]  # default (group) colour

    emitted = []
    panel.annotations_changed.connect(lambda anns: emitted.append(list(anns)))

    # Recolour just the distance (row 0); the colour dialog is stubbed.
    monkeypatch.setattr(QColorDialog, "getColor", staticmethod(lambda *a, **k: QColor("#123456")))
    panel._list.item(0).setSelected(True)
    panel._list.item(1).setSelected(False)
    panel._set_measurement_colour()

    assert panel.measurements()[0].color == "#123456"  # the distance is recoloured
    assert panel.measurements()[1].color is None        # the plane is untouched
    assert emitted[-1][0].color == "#123456"            # and the redraw sees it


def test_colour_button_needs_a_selected_measurement(qapp):
    panel = GeometryPanel(_water())
    panel.set_selection([0, 1])
    panel._measure_selection()
    panel._list.clearSelection()
    panel._sync_buttons()
    assert not panel._color_btn.isEnabled()  # nothing selected → nothing to recolour
    panel._list.item(0).setSelected(True)
    assert panel._color_btn.isEnabled()
