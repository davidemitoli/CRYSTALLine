"""Structure panel: the working editing surface for the atomic structure.

Provides:
* an element picker (combo box) + **Add** (append an atom of that element),
* an atom list that supports **multi-selection** (Ctrl/Shift, or box-drag),
* a **coordinate editor** (X/Y/Z) active when a single atom is selected.

Selection is shared with the 3D view: picking atoms in the viewport updates the
selection here, and this panel is the single source of truth for *what is
selected* — it emits :attr:`selection_changed`, which the viewport turns into
highlights and the Edit menu uses to enable its tools (delete, duplicate,
translate, set element).

Mutating controls are disabled until editing is switched on (Edit ▸ Enable
editing). The panel talks only to :class:`Structure`, never to VTK directly.
"""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from crystalline.core.structure import Structure

# A small starter palette; a full periodic-table widget can replace this.
_COMMON_ELEMENTS = ["H", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Na", "Mg", "Al", "Ca", "Fe"]
_COORD_RANGE = 1000.0


class StructurePanel(QWidget):
    """Edit the atomic structure: pick elements, add atoms, multi-select, move."""

    element_selected = Signal(str)  # current palette element (drag source)
    selection_changed = Signal(list)  # the current selection (sorted atom indices)

    def __init__(self, structure: Structure, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._structure = structure
        self._structure.add_listener(self._on_structure_changed)
        self._selection: List[int] = []
        self._editing = False

        layout = QVBoxLayout(self)

        # ── element picker + add ────────────────────────────────────────
        layout.addWidget(QLabel("Element"))
        pick_row = QHBoxLayout()
        self.element_box = QComboBox()
        self.element_box.addItems(_COMMON_ELEMENTS)
        self.element_box.setCurrentText("C")
        self.element_box.setEditable(True)  # allow any symbol, e.g. "Ti"
        self.element_box.currentTextChanged.connect(self.element_selected)
        pick_row.addWidget(self.element_box, 1)
        self.add_btn = QPushButton("Add atom")
        self.add_btn.clicked.connect(self._add_atom)
        pick_row.addWidget(self.add_btn)
        layout.addLayout(pick_row)

        # ── atom list (multi-select) ────────────────────────────────────
        layout.addWidget(QLabel("Atoms  (Ctrl/Shift for multi-select)"))
        self.atom_list = QListWidget()
        self.atom_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.atom_list.itemSelectionChanged.connect(self._on_list_selection_changed)
        layout.addWidget(self.atom_list, 1)

        # ── single-atom coordinate editor (move) ────────────────────────
        self.editor = QGroupBox("Selected atom")
        form = QFormLayout(self.editor)
        self.coord_boxes = []
        for axis in ("x", "y", "z"):
            box = QDoubleSpinBox()
            box.setRange(-_COORD_RANGE, _COORD_RANGE)
            box.setDecimals(4)
            box.setSingleStep(0.1)
            box.valueChanged.connect(self._on_coord_changed)
            form.addRow(f"{axis} (Å)", box)
            self.coord_boxes.append(box)
        layout.addWidget(self.editor)

        self._refresh_list()
        self.set_editing_enabled(False)

    # ── external hooks ──────────────────────────────────────────────────
    def set_structure(self, structure: Structure) -> None:
        """Rebind the panel to a different structure (e.g. after loading a file)."""
        self._structure.remove_listener(self._on_structure_changed)
        self._structure = structure
        self._structure.add_listener(self._on_structure_changed)
        self._selection = []
        self._refresh_list()

    def set_editing_enabled(self, enabled: bool) -> None:
        """Enable/disable the mutating controls (add + coordinate editor)."""
        self._editing = bool(enabled)
        self.add_btn.setEnabled(self._editing)
        self._sync_editor()

    def current_element(self) -> str:
        return self.element_box.currentText().strip()

    def selected_indices(self) -> List[int]:
        return list(self._selection)

    def select_atom(self, index: int, additive: bool = False) -> None:
        """Select an atom from a viewport pick; ``additive`` toggles it in place."""
        if not 0 <= index < len(self._structure):
            return
        if additive:
            selection = set(self._selection)
            selection.symmetric_difference_update({index})
            self.set_selection(sorted(selection))
        else:
            self.set_selection([index])

    def set_selection(self, indices) -> None:
        """Programmatically set the selection and sync the list + listeners."""
        valid = sorted({int(i) for i in indices if 0 <= int(i) < len(self._structure)})
        self.atom_list.blockSignals(True)
        self.atom_list.clearSelection()
        for i in valid:
            self.atom_list.item(i).setSelected(True)
        self.atom_list.blockSignals(False)
        self._set_selection(valid)

    def clear_selection(self) -> None:
        """Drop the current selection (e.g. after clicking empty 3D space)."""
        self.set_selection([])

    # ── editing actions ─────────────────────────────────────────────────
    def _add_atom(self) -> None:
        symbol = self.current_element()
        try:
            index = self._structure.add_atom(symbol, self._default_position())
        except ValueError:
            return  # unknown element typed; ignore silently for now
        self.set_selection([index])

    def _on_coord_changed(self, _value: float) -> None:
        if len(self._selection) != 1:
            return
        pos = [b.value() for b in self.coord_boxes]
        self._structure.move_atom(self._selection[0], pos)

    # ── model <-> view sync ─────────────────────────────────────────────
    def _on_structure_changed(self, _structure: Structure) -> None:
        self._refresh_list()

    def _refresh_list(self) -> None:
        self.atom_list.blockSignals(True)
        self.atom_list.clear()
        positions = self._structure.positions
        for i, sym in enumerate(self._structure.symbols):
            x, y, z = positions[i]
            item = QListWidgetItem(f"{i}: {sym}  ({x:.2f}, {y:.2f}, {z:.2f})")
            item.setData(Qt.UserRole, i)
            self.atom_list.addItem(item)
        # keep whatever is still in range selected across the rebuild
        kept = [i for i in self._selection if 0 <= i < self.atom_list.count()]
        for i in kept:
            self.atom_list.item(i).setSelected(True)
        self.atom_list.blockSignals(False)
        # re-broadcast so highlights re-apply after the renderer rebuilt
        self._set_selection(kept)

    def _on_list_selection_changed(self) -> None:
        rows = sorted(i.row() for i in self.atom_list.selectedIndexes())
        self._set_selection(rows)

    def _set_selection(self, indices: List[int]) -> None:
        """Record the selection, refresh the editor, and notify listeners."""
        self._selection = list(indices)
        self._sync_editor()
        self.selection_changed.emit(list(self._selection))

    def _sync_editor(self) -> None:
        """The coordinate editor edits a single atom, and only while editing."""
        single = self._editing and len(self._selection) == 1
        self.editor.setEnabled(single)
        title = "Selected atom"
        if len(self._selection) > 1:
            title = f"{len(self._selection)} atoms selected"
        self.editor.setTitle(title)
        if not single:
            return
        pos = self._structure.positions[self._selection[0]]
        for box, value in zip(self.coord_boxes, pos):
            box.blockSignals(True)
            box.setValue(float(value))
            box.blockSignals(False)

    # ── helpers ─────────────────────────────────────────────────────────
    def _default_position(self) -> list:
        """Where a newly added atom appears: cell centre, else the origin."""
        import numpy as np

        cell = self._structure.cell
        if self._structure.is_periodic and not np.allclose(cell, 0.0):
            return list(0.5 * cell.sum(axis=0))
        return [0.0, 0.0, 0.0]


__all__ = ["StructurePanel"]
