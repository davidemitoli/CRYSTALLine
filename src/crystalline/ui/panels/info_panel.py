"""Info panel: crystallographic summary of the loaded system.

Shows the space group, lattice parameters, density, formula, etc. (derived from
the structure via :func:`crystalline.core.crystallography.analyze`), plus any
CRYSTAL-computed properties read from the ``.out`` file (energy, band gap,
Fermi energy).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from crystalline.core.crystallography import analyze
from crystalline.core.structure import Structure


class InfoPanel(QWidget):
    """Read-only crystallographic + CRYSTAL-output summary."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        self._crystal_group = QGroupBox("Crystallography")
        self._crystal_form = QFormLayout(self._crystal_group)
        layout.addWidget(self._crystal_group)

        self._output_group = QGroupBox("CRYSTAL output")
        self._output_form = QFormLayout(self._output_group)
        layout.addWidget(self._output_group)
        self._output_group.setVisible(False)

        layout.addStretch(1)
        self.clear()

    # ── public API ──────────────────────────────────────────────────────
    def show_structure(self, structure: Structure, output_props: Optional[dict] = None) -> None:
        """Analyse ``structure`` and display it, with optional CRYSTAL-output rows."""
        if structure is None or len(structure) == 0:
            self.clear()
            return
        _fill(self._crystal_form, analyze(structure).rows())
        props = output_props or {}
        _fill(self._output_form, list(props.items()))
        self._output_group.setVisible(bool(props))

    def clear(self) -> None:
        _fill(self._crystal_form, [("", "No structure loaded")])
        self._output_group.setVisible(False)


def _fill(form: QFormLayout, rows: List[Tuple[str, str]]) -> None:
    """Replace the form's contents with ``(label, value)`` rows."""
    while form.rowCount():
        form.removeRow(0)
    for label, value in rows:
        value_label = QLabel(str(value))
        value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)  # copyable
        form.addRow(f"{label}:" if label else "", value_label)


__all__ = ["InfoPanel"]
