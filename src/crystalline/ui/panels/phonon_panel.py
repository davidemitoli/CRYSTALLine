"""Phonon panel: list vibrational modes and drive the animation.

Owns the ``QTimer`` that advances the animation phase (the animator itself is
Qt-free). Selecting a mode sets it on the :class:`PhononAnimator`; Play/Stop
toggle the timer; an amplitude control scales the displacement.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from crystalline.core.phonons import PhononModes
from crystalline.viz.phonon_animator import PhononAnimator


class PhononPanel(QWidget):
    """Choose a phonon mode and animate it in the viewport."""

    mode_selected = Signal(int)

    def __init__(self, animator: PhononAnimator, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._animator = animator
        self._modes: Optional[PhononModes] = None

        self._phases = PhononAnimator.phase_sequence(60)
        self._frame = 0
        self._timer = QTimer(self)
        self._timer.setInterval(33)  # ~30 fps
        self._timer.timeout.connect(self._tick)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Vibrational modes"))
        self.mode_list = QListWidget()
        self.mode_list.currentRowChanged.connect(self._on_mode_changed)
        layout.addWidget(self.mode_list)

        amp_row = QHBoxLayout()
        amp_row.addWidget(QLabel("Amplitude"))
        self.amp_box = QDoubleSpinBox()
        self.amp_box.setRange(0.0, 5.0)
        self.amp_box.setSingleStep(0.1)
        self.amp_box.setValue(self._animator.amplitude)
        self.amp_box.valueChanged.connect(self._on_amplitude)
        amp_row.addWidget(self.amp_box)
        layout.addLayout(amp_row)

        btn_row = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self._play)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop)
        btn_row.addWidget(self.play_btn)
        btn_row.addWidget(self.stop_btn)
        layout.addLayout(btn_row)

        self.setEnabled(False)

    # ── data ────────────────────────────────────────────────────────────
    def set_modes(self, equilibrium: np.ndarray, modes: PhononModes) -> None:
        self._stop()
        self._modes = modes
        self._equilibrium = np.asarray(equilibrium, dtype=float)
        self.mode_list.clear()
        for i, m in enumerate(modes):
            tag = " (imag)" if m.is_imaginary else ""
            self.mode_list.addItem(f"{i}: {m.frequency:9.2f} cm⁻¹{tag}")
        self.setEnabled(len(modes) > 0)
        if len(modes) > 0:
            self.mode_list.setCurrentRow(0)

    def has_mode(self) -> bool:
        """Whether a mode is currently selected (so it can be animated/exported)."""
        return (
            self._modes is not None
            and self._equilibrium is not None
            and self.mode_list.currentRow() >= 0
        )

    def current_selection(self):
        """``(equilibrium, PhononMode)`` for the selected mode, or ``None``."""
        if not self.has_mode():
            return None
        return self._equilibrium, self._modes[self.mode_list.currentRow()]

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
        self.mode_list.clear()
        self.setEnabled(False)

    # ── interaction ─────────────────────────────────────────────────────
    def _on_mode_changed(self, row: int) -> None:
        if self._modes is None or row < 0:
            return
        self._animator.set_mode(self._equilibrium, self._modes[row])
        self._frame = 0
        self._animator.set_frame(0.0)
        self.mode_selected.emit(row)

    def _on_amplitude(self, value: float) -> None:
        self._animator.amplitude = value

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
        row = self.mode_list.currentRow()
        if row >= 0:
            self._animator.set_mode(self._equilibrium, self._modes[row])

    def _play(self) -> None:
        if self._modes is not None:
            self._timer.start()

    def _stop(self) -> None:
        self._timer.stop()
        self._animator.reset()

    def _tick(self) -> None:
        self._animator.set_frame(float(self._phases[self._frame]))
        self._frame = (self._frame + 1) % len(self._phases)


__all__ = ["PhononPanel"]
