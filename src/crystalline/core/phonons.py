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

from dataclasses import dataclass

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
    """

    frequency: float
    eigenvector: np.ndarray

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


def displaced_positions(
    equilibrium: np.ndarray,
    mode: PhononMode,
    amplitude: float,
    phase: float,
) -> np.ndarray:
    """Return atom positions for one animation frame.

    Parameters
    ----------
    equilibrium:
        (N, 3) equilibrium cartesian positions.
    mode:
        The mode being animated.
    amplitude:
        Peak displacement scale (Angstrom-ish; user-tunable in the UI).
    phase:
        Animation phase in radians; the frame factor is ``sin(phase)``.
    """
    eq = np.asarray(equilibrium, dtype=float)
    if eq.shape != mode.eigenvector.shape:
        raise ValueError(
            f"geometry {eq.shape} does not match eigenvector {mode.eigenvector.shape}"
        )
    return eq + amplitude * np.sin(phase) * mode.eigenvector


__all__ = ["PhononMode", "PhononModes", "displaced_positions"]
