"""Animate a phonon mode by displacing atoms along its eigenvector.

The animator is intentionally *timer-free*: it computes the geometry for a
given phase and pushes it to the renderer via ``update_positions``. The UI owns
the ``QTimer`` and advances the phase, so this stays Qt-independent and testable.
"""

from __future__ import annotations

import numpy as np

from crystalline.core.phonons import PhononMode, displaced_positions
from crystalline.viz.renderer import StructureRenderer

# Default peak displacement in Angstrom. Chosen to match the motion the old
# cell-normalised amplitude gave for a small molecule — the case that looked
# right — now that the scale no longer shrinks with the number of atoms.
DEFAULT_AMPLITUDE = 0.3


class PhononAnimator:
    """Drive a :class:`StructureRenderer` to show a vibrating mode."""

    def __init__(self, renderer: StructureRenderer) -> None:
        self.renderer = renderer
        self._equilibrium: np.ndarray | None = None
        self._mode: PhononMode | None = None
        # Peak displacement (Angstrom) of the most-displaced atom — see
        # core.phonons.displaced_positions. Being per-atom rather than per-cell,
        # this one default reads well for a molecule and for a large cell alike.
        self.amplitude: float = DEFAULT_AMPLITUDE

    def set_mode(self, equilibrium: np.ndarray, mode: PhononMode) -> None:
        """Select the mode to animate around a fixed equilibrium geometry.

        The equilibrium also becomes the renderer's bond reference, so the bond
        network stays the one the molecule actually has instead of being
        re-derived from each displaced frame — which broke up at large amplitude.
        """
        self._equilibrium = np.asarray(equilibrium, dtype=float)
        self._mode = mode
        self.renderer.set_bond_reference(self._equilibrium)

    def set_frame(self, phase: float) -> None:
        """Render one frame at animation ``phase`` (radians)."""
        if self._mode is None or not self._matches_renderer():
            return
        pos = displaced_positions(self._equilibrium, self._mode, self.amplitude, phase)
        self.renderer.update_positions(pos)

    def reset(self) -> None:
        """Return atoms to their equilibrium positions and re-enable live bonding."""
        self.renderer.set_bond_reference(None)
        if self._matches_renderer():
            self.renderer.update_positions(self._equilibrium)

    def _matches_renderer(self) -> bool:
        """Whether the held equilibrium still fits the renderer's atom count.

        The displayed structure can be swapped for a different one — a supercell,
        a different cell view — between when a mode was set and when a frame (or a
        reset from ``PhononPanel._stop``) fires. Pushing the stale, wrong-sized
        equilibrium would raise in ``update_positions``; skip instead.
        """
        return self._equilibrium is not None and len(self._equilibrium) == self.renderer.atom_count

    @staticmethod
    def phase_sequence(n_frames: int = 60) -> np.ndarray:
        """A full 0..2*pi loop suitable for cycling with a timer."""
        return np.linspace(0.0, 2.0 * np.pi, n_frames, endpoint=False)


__all__ = ["DEFAULT_AMPLITUDE", "PhononAnimator"]
