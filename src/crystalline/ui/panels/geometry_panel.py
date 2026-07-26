"""Geometry panel: measure the selection, and edit atoms with the mouse.

Docked beside Info and Display. Two halves:

* **Measure** — turn the current 3D selection into a distance (2 atoms), angle
  (3), dihedral (4) or least-squares plane (3+), keep a list of them, and draw
  the ones that are ticked over the structure as points/lines/planes.
* **Atoms** — add, delete, duplicate, re-element and translate the selection
  without going to the Edit menu. These mirror the Edit-menu actions and are
  gated the same way: they need **Editing mode**, which is exposed here as a
  checkbox so it is obvious (and one click away) why a button is greyed out.

The panel owns no editing logic — it emits intent and :class:`MainWindow` runs
the same slots the menu uses, so undo/redo and the selection model behave
identically whichever route the user takes.
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional, Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from crystalline.core import measure as measure_mod
from crystalline.core.structure import Structure

# Same starter palette the (hidden) structure panel used; free text is allowed.
_ELEMENTS = ["H", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Na", "Mg", "Al", "Ca", "Ti", "Fe"]


class GeometryPanel(QWidget):
    """Measurements and atom-editing tools for the current selection."""

    # editing intent — MainWindow runs the matching Edit-menu slot
    editing_toggled = Signal(bool)
    add_atom_requested = Signal(str)      # element symbol
    delete_requested = Signal()
    duplicate_requested = Signal()
    translate_requested = Signal()
    set_element_requested = Signal(str)   # element symbol
    # the measurements that should be drawn in 3D (possibly empty)
    annotations_changed = Signal(list)

    def __init__(self, structure: Structure, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._structure = structure
        self._selection: List[int] = []
        self._measurements: List[measure_mod.Measurement] = []
        self._editing = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.addWidget(self._build_measure_group(), 1)
        layout.addWidget(self._build_atoms_group())
        layout.addStretch(0)
        self._sync_buttons()

    # ── construction ────────────────────────────────────────────────────
    def _build_measure_group(self) -> QWidget:
        group = QGroupBox("Measure")
        box = QVBoxLayout(group)

        self._hint = QLabel(measure_mod.selection_hint(0))
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet("color: palette(mid);")
        box.addWidget(self._hint)

        row = QHBoxLayout()
        self._measure_btn = QPushButton("Measure selection")
        self._measure_btn.setToolTip(
            "2 atoms → distance, 3 → angle, 4 → dihedral, 5+ → plane fit"
        )
        self._measure_btn.clicked.connect(self._measure_selection)
        row.addWidget(self._measure_btn, 1)
        self._plane_btn = QPushButton("Fit plane")
        self._plane_btn.setToolTip("Least-squares plane through 3 or more selected atoms")
        self._plane_btn.clicked.connect(self._measure_plane)
        row.addWidget(self._plane_btn)
        box.addLayout(row)

        # Ticked measurements are drawn in the 3D view.
        self._list = QListWidget()
        self._list.setToolTip("Tick a measurement to show it in the 3D view")
        self._list.setSelectionMode(QListWidget.ExtendedSelection)
        self._list.itemChanged.connect(lambda _item: self._emit_annotations())
        box.addWidget(self._list, 1)

        row = QHBoxLayout()
        self._color_btn = QPushButton("Colour…")
        self._color_btn.setToolTip("Set the colour of the selected measurement(s)")
        self._color_btn.clicked.connect(self._set_measurement_colour)
        row.addWidget(self._color_btn)
        self._remove_btn = QPushButton("Remove")
        self._remove_btn.clicked.connect(self._remove_selected_measurements)
        row.addWidget(self._remove_btn)
        self._clear_btn = QPushButton("Clear all")
        self._clear_btn.clicked.connect(self.clear_measurements)
        row.addWidget(self._clear_btn)
        row.addStretch(1)
        box.addLayout(row)
        # The colour button follows the list selection, not just "any measurement".
        self._list.itemSelectionChanged.connect(self._sync_buttons)
        return group

    def _build_atoms_group(self) -> QWidget:
        group = QGroupBox("Atoms")
        box = QVBoxLayout(group)

        # The gate the Edit menu applies invisibly — shown here so a greyed-out
        # button explains itself.
        self._edit_check = QCheckBox("Editing mode")
        self._edit_check.setToolTip(
            "Atom tools (and dragging atoms in 3D) need editing mode — same as Edit ▸ Editing mode (Ctrl+E)"
        )
        self._edit_check.toggled.connect(self._on_editing_toggled)
        box.addWidget(self._edit_check)

        row = QHBoxLayout()
        row.addWidget(QLabel("Element"))
        self._element = QComboBox()
        self._element.addItems(_ELEMENTS)
        self._element.setCurrentText("C")
        self._element.setEditable(True)  # any symbol, e.g. "Zr"
        row.addWidget(self._element, 1)
        self._table_btn = QPushButton("⊞")
        self._table_btn.setToolTip("Pick an element from the periodic table")
        self._table_btn.setFixedWidth(32)
        self._table_btn.clicked.connect(self._pick_from_periodic_table)
        row.addWidget(self._table_btn)
        self._add_btn = QPushButton("Add")
        self._add_btn.setToolTip("Add an atom of this element at the centre of the cell")
        self._add_btn.clicked.connect(
            lambda: self.add_atom_requested.emit(self._element.currentText().strip())
        )
        row.addWidget(self._add_btn)
        self._set_element_btn = QPushButton("Set")
        self._set_element_btn.setToolTip("Change the selected atoms to this element")
        self._set_element_btn.clicked.connect(
            lambda: self.set_element_requested.emit(self._element.currentText().strip())
        )
        row.addWidget(self._set_element_btn)
        box.addLayout(row)

        row = QHBoxLayout()
        self._delete_btn = QPushButton("Delete")
        self._delete_btn.setToolTip("Delete the selected atoms (Del)")
        self._delete_btn.clicked.connect(self.delete_requested)
        row.addWidget(self._delete_btn)
        self._duplicate_btn = QPushButton("Duplicate")
        self._duplicate_btn.clicked.connect(self.duplicate_requested)
        row.addWidget(self._duplicate_btn)
        self._translate_btn = QPushButton("Translate…")
        self._translate_btn.clicked.connect(self.translate_requested)
        row.addWidget(self._translate_btn)
        box.addLayout(row)
        return group

    def _pick_from_periodic_table(self) -> None:
        """Open the visual periodic table and load the chosen element into the box."""
        from crystalline.ui.periodic_table import PeriodicTableDialog

        symbol = PeriodicTableDialog.pick(self, current=self._element.currentText().strip())
        if symbol:
            self._element.setCurrentText(symbol)

    # ── external hooks ──────────────────────────────────────────────────
    def set_structure(self, structure: Structure) -> None:
        """Rebind after a load or a view change; measurements no longer apply."""
        self._structure = structure
        self._selection = []
        self.clear_measurements()
        self._sync_buttons()

    def set_selection(self, indices: Sequence[int]) -> None:
        """Track the shared selection (driven by 3D picking)."""
        self._selection = [int(i) for i in indices]
        self._hint.setText(measure_mod.selection_hint(len(self._selection)))
        self._sync_buttons()

    def set_editing_enabled(self, enabled: bool) -> None:
        """Reflect the Edit menu's editing mode without re-emitting it."""
        self._editing = bool(enabled)
        blocked = self._edit_check.blockSignals(True)
        self._edit_check.setChecked(self._editing)
        self._edit_check.blockSignals(blocked)
        self._sync_buttons()

    def measurements(self) -> List[measure_mod.Measurement]:
        """Every measurement in the list, shown or not (handy in tests)."""
        return list(self._measurements)

    def shown_annotations(self) -> List[measure_mod.Measurement]:
        """The measurements currently ticked for display in 3D."""
        return [
            m
            for m, item in zip(self._measurements, self._items())
            if item.checkState() == Qt.Checked
        ]

    def clear_measurements(self) -> None:
        self._measurements = []
        self._list.clear()
        self._emit_annotations()

    # ── measuring ───────────────────────────────────────────────────────
    def _measure_selection(self) -> None:
        self._add_measurement(
            measure_mod.measure(
                self._structure.positions, self._structure.symbols, self._selection
            )
        )

    def _measure_plane(self) -> None:
        self._add_measurement(
            measure_mod.measure_plane(
                self._structure.positions, self._structure.symbols, self._selection
            )
        )

    def _add_measurement(self, result) -> None:
        if result is None:
            return
        self._measurements.append(result)
        item = QListWidgetItem(result.summary())
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)  # new measurements are shown straight away
        self._list.addItem(item)
        self._emit_annotations()

    def _set_measurement_colour(self) -> None:
        """Recolour the selected measurement(s) — a per-item override of the
        type's default colour. Each item can carry its own colour."""
        rows = sorted(self._list.row(i) for i in self._list.selectedItems())
        if not rows:
            return
        current = self._measurements[rows[0]].color
        seed = QColor(current) if current else QColor("#ff7f0e")
        chosen = QColorDialog.getColor(seed, self, "Measurement colour")
        if not chosen.isValid():
            return
        hex_color = chosen.name()
        for row in rows:
            self._measurements[row] = dataclasses.replace(
                self._measurements[row], color=hex_color
            )
            self._list.item(row).setIcon(_colour_swatch(hex_color))
        self._emit_annotations()

    def _remove_selected_measurements(self) -> None:
        for row in sorted((self._list.row(i) for i in self._list.selectedItems()), reverse=True):
            self._list.takeItem(row)
            del self._measurements[row]
        self._emit_annotations()

    def _items(self) -> List[QListWidgetItem]:
        return [self._list.item(row) for row in range(self._list.count())]

    def _emit_annotations(self) -> None:
        self.annotations_changed.emit(self.shown_annotations())

    # ── editing ─────────────────────────────────────────────────────────
    def _on_editing_toggled(self, enabled: bool) -> None:
        self._editing = bool(enabled)
        self._sync_buttons()
        self.editing_toggled.emit(self._editing)

    def _sync_buttons(self) -> None:
        """Measuring needs a selection; editing needs editing mode as well."""
        count = len(self._selection)
        self._measure_btn.setEnabled(count >= 1)
        self._plane_btn.setEnabled(count >= 3)
        self._remove_btn.setEnabled(bool(self._measurements))
        self._clear_btn.setEnabled(bool(self._measurements))
        self._color_btn.setEnabled(bool(self._list.selectedItems()))

        self._add_btn.setEnabled(self._editing)
        on_selection = self._editing and count >= 1
        for button in (self._delete_btn, self._duplicate_btn,
                       self._translate_btn, self._set_element_btn):
            button.setEnabled(on_selection)


def _colour_swatch(color: str, size: int = 12) -> QIcon:
    """A small solid-colour square, shown beside a recoloured measurement."""
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(color))
    return QIcon(pixmap)


__all__ = ["GeometryPanel"]
