"""Phonon panel: list vibrational modes and drive the animation.

Owns the ``QTimer`` that advances the animation phase (the animator itself is
Qt-free). Selecting a mode sets it on the :class:`PhononAnimator`; the Play/Stop
icons at the top of the panel toggle the timer; amplitude scales the
displacement and speed scales how fast the phase advances. A filter narrows the
list to the IR- and/or Raman-active modes when the output reports selection
rules.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from PySide6.QtCore import QSize, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QSizePolicy,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from crystalline.core.phonons import PhononMode, PhononModes
from crystalline.viz.phonon_animator import PhononAnimator

# Activity filters: (label, predicate over a PhononMode). ``None`` keeps every
# mode; the others are only selectable when the output carries the analysis.
_FILTERS = (
    ("All modes", None),
    ("IR active", lambda m: bool(m.ir_active)),
    ("Raman active", lambda m: bool(m.raman_active)),
    ("IR or Raman active", lambda m: bool(m.ir_active) or bool(m.raman_active)),
)

# The mode list must be able to shrink to this, or its size hint pushes the
# controls out of a short dock (reported on Windows, where the dock is laid out
# tighter). It keeps its scrollbar and stretches when there's room.
_LIST_MIN_HEIGHT = 70

# Animation timing. The timer stays at ~30 fps whatever the speed — speed
# changes how far the phase moves per tick, so the motion stays smooth instead
# of turning into a slideshow at low speed.
_FRAME_INTERVAL_MS = 33
_FRAMES_PER_CYCLE = 60  # at speed 1.0: one full vibration in ~2 s
_PHASE_STEP = 2.0 * math.pi / _FRAMES_PER_CYCLE


def _mode_label(index: int, mode: PhononMode) -> str:
    """One list row: index, frequency and the tags that apply to the mode."""
    tags = []
    if mode.is_imaginary:
        tags.append("imag")
    if mode.ir_active:
        tags.append("IR")
    if mode.raman_active:
        tags.append("R")
    suffix = f"  [{', '.join(tags)}]" if tags else ""
    return f"{index}: {mode.frequency:9.2f} cm⁻¹{suffix}"


class PhononPanel(QWidget):
    """Choose a phonon mode and animate it in the viewport."""

    mode_selected = Signal(int)

    def __init__(self, animator: PhononAnimator, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._animator = animator
        self._modes: Optional[PhononModes] = None
        self._equilibrium: Optional[np.ndarray] = None
        self._rows: list[int] = []  # list row -> index into self._modes

        self._phase = 0.0
        self._speed = 1.0
        self._timer = QTimer(self)
        self._timer.setInterval(_FRAME_INTERVAL_MS)
        self._timer.timeout.connect(self._tick)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # Transport controls sit at the top, where they're always in view even
        # if the dock is short enough to clip the bottom of the panel.
        head_row = QHBoxLayout()
        self.play_btn = self._transport_button(QStyle.SP_MediaPlay, "Play", "▶")
        self.play_btn.clicked.connect(self._play)
        self.stop_btn = self._transport_button(QStyle.SP_MediaStop, "Stop", "■")
        self.stop_btn.clicked.connect(self._stop)
        head_row.addWidget(self.play_btn)
        head_row.addWidget(self.stop_btn)
        head_row.addSpacing(6)
        head_row.addWidget(QLabel("Vibrational modes"), 1)
        layout.addLayout(head_row)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Show"))
        self.filter_box = QComboBox()
        self.filter_box.addItems([label for label, _pred in _FILTERS])
        self.filter_box.currentIndexChanged.connect(lambda _i: self._populate())
        filter_row.addWidget(self.filter_box, 1)
        layout.addLayout(filter_row)

        self.mode_list = QListWidget()
        self.mode_list.setMinimumHeight(_LIST_MIN_HEIGHT)
        self.mode_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.mode_list.currentRowChanged.connect(self._on_mode_changed)
        layout.addWidget(self.mode_list, 1)  # the only widget that takes the slack

        # Amplitude and speed share a form so their labels and fields line up.
        controls = QFormLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(4)

        self.amp_box = QDoubleSpinBox()
        self.amp_box.setRange(0.01, 5.0)
        self.amp_box.setSingleStep(0.05)
        self.amp_box.setDecimals(2)
        self.amp_box.setValue(self._animator.amplitude)
        self.amp_box.setToolTip(
            "Peak displacement of the most-displaced atom, in Angstrom.\n"
            "Independent of how many atoms the cell has."
        )
        self.amp_box.valueChanged.connect(self._on_amplitude)
        controls.addRow("Amplitude (Å)", self.amp_box)

        self.speed_box = QDoubleSpinBox()
        self.speed_box.setRange(0.1, 10.0)
        self.speed_box.setSingleStep(0.1)
        self.speed_box.setDecimals(1)
        self.speed_box.setSuffix(" ×")
        self.speed_box.setValue(self._speed)
        self.speed_box.setToolTip(
            "How fast the vibration is played back.\n"
            "1.0× is one full cycle every two seconds; it changes nothing physical."
        )
        self.speed_box.valueChanged.connect(self._on_speed)
        controls.addRow("Speed", self.speed_box)
        layout.addLayout(controls)

        self.setEnabled(False)

    def _transport_button(self, pixmap, tooltip: str, fallback: str) -> QToolButton:
        """A flat icon button for Play/Stop, captioned if the style has no icon."""
        button = QToolButton(self)
        icon = self.style().standardIcon(pixmap)
        if icon.isNull():  # some minimal styles ship no media icons
            button.setText(fallback)
        else:
            button.setIcon(icon)
            button.setIconSize(QSize(18, 18))
        button.setToolTip(tooltip)
        button.setAccessibleName(tooltip)
        button.setAutoRaise(True)
        return button

    # ── data ────────────────────────────────────────────────────────────
    def set_modes(self, equilibrium: np.ndarray, modes: PhononModes) -> None:
        self._stop()
        # Drop the old rows before repopulating: a stale row would otherwise be
        # read back as a "keep this mode" hint for a different set of modes.
        self._rows = []
        self.mode_list.blockSignals(True)
        self.mode_list.clear()
        self.mode_list.blockSignals(False)
        self._modes = modes
        self._equilibrium = np.asarray(equilibrium, dtype=float)
        self._set_filter_available(modes.has_activity)
        self._populate()
        self.setEnabled(len(modes) > 0)

    def has_mode(self) -> bool:
        """Whether a mode is currently selected (so it can be animated/exported)."""
        return self.current_mode_index() is not None

    def current_mode_index(self) -> Optional[int]:
        """Index into the loaded modes of the selected row, or ``None``.

        The list can be filtered, so the row is *not* the mode index.
        """
        if self._modes is None or self._equilibrium is None:
            return None
        row = self.mode_list.currentRow()
        if row < 0 or row >= len(self._rows):
            return None
        return self._rows[row]

    def current_selection(self):
        """``(equilibrium, PhononMode)`` for the selected mode, or ``None``."""
        index = self.current_mode_index()
        if index is None:
            return None
        return self._equilibrium, self._modes[index]

    def clear(self) -> None:
        """Reset the panel (e.g. after loading a file with no phonons).

        Only the timer is stopped — we deliberately do NOT call the animator's
        ``reset`` here: clear() runs after the geometry has already been redrawn
        (possibly with a different atom count), so pushing the old equilibrium
        back into the renderer would mismatch and is unnecessary.
        """
        self._timer.stop()
        self._modes = None
        self._equilibrium = None
        self._rows = []
        self.mode_list.clear()
        self._set_filter_available(False)
        self.setEnabled(False)

    # ── list building ───────────────────────────────────────────────────
    def _set_filter_available(self, available: bool) -> None:
        """Offer the activity filters only when the output labels the modes."""
        self.filter_box.blockSignals(True)
        if not available:
            self.filter_box.setCurrentIndex(0)
        self.filter_box.blockSignals(False)
        self.filter_box.setEnabled(available)
        self.filter_box.setToolTip(
            "" if available else "This output has no IR/Raman activity analysis"
        )

    def _populate(self) -> None:
        """Rebuild the list under the current filter, keeping the selected mode.

        Reselecting is by *mode index*, not row, so switching filters doesn't
        jump to a different mode when the one in view survives the filter.
        """
        keep = self.current_mode_index()
        predicate = _FILTERS[self.filter_box.currentIndex()][1]

        self._rows = []
        self.mode_list.blockSignals(True)
        self.mode_list.clear()
        if self._modes is not None:
            for i, mode in enumerate(self._modes):
                if predicate is not None and not predicate(mode):
                    continue
                self._rows.append(i)
                self.mode_list.addItem(_mode_label(i, mode))
        self.mode_list.blockSignals(False)

        if not self._rows:
            self._stop()  # nothing left to animate under this filter
            return
        row = self._rows.index(keep) if keep in self._rows else 0
        # Select quietly, then drive the change by hand: setCurrentRow emits
        # nothing when the row number happens to be unchanged, even though the
        # mode under it may not be.
        self.mode_list.blockSignals(True)
        self.mode_list.setCurrentRow(row)
        self.mode_list.blockSignals(False)
        self._on_mode_changed(row)

    # ── interaction ─────────────────────────────────────────────────────
    def _on_mode_changed(self, row: int) -> None:
        if self._modes is None or row < 0 or row >= len(self._rows):
            return
        index = self._rows[row]
        self._animator.set_mode(self._equilibrium, self._modes[index])
        self._phase = 0.0
        self._animator.set_frame(0.0)
        self.mode_selected.emit(index)

    def _on_amplitude(self, value: float) -> None:
        self._animator.amplitude = value

    def _on_speed(self, value: float) -> None:
        self._speed = value

    def stop(self) -> None:
        """Public stop — called when the user starts editing the structure."""
        self._stop()

    def invalidate_on_edit(self, positions: np.ndarray) -> None:
        """Reconcile the panel after a structural edit.

        Stops a running animation. If the atom count changed, the loaded modes
        are stale and dropped. Otherwise the animation is re-anchored to the
        *edited* geometry — crucially WITHOUT resetting atoms to the old
        equilibrium, which would snap them back and undo the edit just made
        (e.g. an atom dropped after a drag reappearing at its original spot).
        """
        if self._modes is None:
            return
        positions = np.asarray(positions, dtype=float)
        # Stop the timer only — do NOT animator.reset() to the stale equilibrium.
        self._timer.stop()
        if self._equilibrium is None or len(self._equilibrium) != len(positions):
            self.clear()  # atom count changed -> modes no longer apply
            return
        self._equilibrium = positions
        index = self.current_mode_index()
        if index is not None:
            self._animator.set_mode(self._equilibrium, self._modes[index])

    def _play(self) -> None:
        if self._modes is not None and self.has_mode():
            self._timer.start()

    def _stop(self) -> None:
        self._timer.stop()
        # Back to phase 0 as well as to the equilibrium geometry, so the next
        # Play starts from rest instead of jumping into mid-cycle.
        self._phase = 0.0
        self._animator.reset()

    def _tick(self) -> None:
        self._animator.set_frame(self._phase)
        self._phase = (self._phase + _PHASE_STEP * self._speed) % (2.0 * math.pi)


__all__ = ["PhononPanel"]
