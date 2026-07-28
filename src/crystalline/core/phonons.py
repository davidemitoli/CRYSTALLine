"""Phonon-mode model: frequencies plus per-atom displacement eigenvectors.

A CRYSTAL phonon calculation yields, for each normal mode, a frequency and a
set of atomic displacement vectors (the eigenvector). Animating a mode means
displacing every atom from its equilibrium position along that eigenvector,
scaled by ``amplitude * sin(phase)``.

This module holds the data only; the actual per-frame geometry is produced by
:func:`displaced_positions`, which ``viz.phonon_animator`` calls. Kept Qt- and
PyVista-free so it can be unit-tested directly.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class PhononMode:
    """A single normal mode.

    Attributes
    ----------
    frequency:
        Mode frequency in cm^-1 (negative denotes an imaginary/soft mode).
    eigenvector:
        (N, 3) real displacement vectors, one row per atom.
    ir_active, raman_active:
        Selection-rule activity as reported by CRYSTAL, or ``None`` when the
        output carries no IR/Raman analysis (dispersion runs, older outputs).
    ir_intensity:
        Harmonic IR intensity in km/mol, when available.
    """

    frequency: float
    eigenvector: np.ndarray
    ir_active: Optional[bool] = None
    raman_active: Optional[bool] = None
    ir_intensity: Optional[float] = None

    def __post_init__(self) -> None:
        ev = np.asarray(self.eigenvector, dtype=float)
        if ev.ndim != 2 or ev.shape[1] != 3:
            raise ValueError(f"eigenvector must be (N, 3), got {ev.shape}")
        object.__setattr__(self, "eigenvector", ev)

    @property
    def n_atoms(self) -> int:
        return self.eigenvector.shape[0]

    @property
    def is_imaginary(self) -> bool:
        return self.frequency < 0.0

    @property
    def has_activity(self) -> bool:
        """Whether selection-rule activity is known for this mode."""
        return self.ir_active is not None or self.raman_active is not None

    @property
    def peak_displacement(self) -> float:
        """Largest per-atom displacement in the raw eigenvector.

        Eigenvectors arrive normalised over the *whole* cell (their 3N-component
        norm is 1), so the per-atom motion shrinks like 1/sqrt(N) and a fixed
        amplitude that looks right for a molecule is invisible for a big cell.
        :func:`displaced_positions` divides by this to make the user-facing
        amplitude mean "peak displacement of the most-displaced atom".
        """
        if self.eigenvector.size == 0:
            return 0.0
        peak = float(np.max(np.linalg.norm(self.eigenvector, axis=1)))
        return peak if np.isfinite(peak) else 0.0

    def with_eigenvector(self, eigenvector: np.ndarray) -> "PhononMode":
        """Copy of this mode carrying a different eigenvector (e.g. tiled onto
        a supercell), keeping the frequency and the IR/Raman labels."""
        return replace(self, eigenvector=eigenvector)


class PhononModes:
    """An ordered collection of :class:`PhononMode` sharing one geometry."""

    def __init__(self, modes: list[PhononMode]) -> None:
        self._modes = list(modes)

    def __len__(self) -> int:
        return len(self._modes)

    def __getitem__(self, i: int) -> PhononMode:
        return self._modes[i]

    def __iter__(self):
        return iter(self._modes)

    @property
    def frequencies(self) -> np.ndarray:
        return np.array([m.frequency for m in self._modes])

    @property
    def has_activity(self) -> bool:
        """Whether IR/Raman activity was reported for any mode.

        False for outputs without the selection-rule analysis (dispersion runs),
        where filtering by activity would silently hide everything.
        """
        return any(m.has_activity for m in self._modes)


def displaced_positions(
    equilibrium: np.ndarray,
    mode: PhononMode,
    amplitude: float,
    phase: float,
) -> np.ndarray:
    """Return atom positions for one animation frame.

    The eigenvector is rescaled so that ``amplitude`` is the peak displacement
    of the *most-displaced atom*, in Angstrom. Without this the raw eigenvector
    (normalised over all 3N components) makes the motion fade out as 1/sqrt(N),
    so one amplitude setting cannot serve both a molecule and a large cell.

    Parameters
    ----------
    equilibrium:
        (N, 3) equilibrium cartesian positions.
    mode:
        The mode being animated.
    amplitude:
        Peak displacement of the most-displaced atom, in Angstrom.
    phase:
        Animation phase in radians; the frame factor is ``sin(phase)``.
    """
    eq = np.asarray(equilibrium, dtype=float)
    if eq.shape != mode.eigenvector.shape:
        raise ValueError(
            f"geometry {eq.shape} does not match eigenvector {mode.eigenvector.shape}"
        )
    peak = mode.peak_displacement
    scale = amplitude / peak if peak > 0.0 else 0.0  # a null mode simply doesn't move
    return eq + scale * np.sin(phase) * mode.eigenvector


__all__ = ["PhononMode", "PhononModes", "displaced_positions"]
