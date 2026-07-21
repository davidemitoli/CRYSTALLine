"""Tests for PhononAnimator's resilience when the displayed structure changes.

Uses an off-screen PyVista plotter (the QtInteractor segfaults off-screen, but a
plain ``pyvista.Plotter`` is fine); skips if PyVista/ase are unavailable.
"""

import numpy as np
import pytest

pytest.importorskip("pyvista")
pytest.importorskip("ase")

import pyvista as pv  # noqa: E402
from ase.build import bulk  # noqa: E402

from crystalline.core.phonons import PhononMode  # noqa: E402
from crystalline.core.structure import Structure  # noqa: E402
from crystalline.viz.phonon_animator import PhononAnimator  # noqa: E402
from crystalline.viz.renderer import StructureRenderer  # noqa: E402


def _renderer():
    return StructureRenderer(pv.Plotter(off_screen=True))


def test_stale_equilibrium_does_not_crash_after_supercell_rebuild():
    # Regression: selecting a mode, then rebuilding the renderer with a different
    # atom count (a supercell), then a reset()/set_frame() — as PhononPanel does
    # on the next set_modes — must not raise on the now-stale equilibrium.
    renderer = _renderer()
    animator = PhononAnimator(renderer)

    small = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64))  # 2 atoms
    renderer.set_structure(small)
    animator.set_mode(small.positions, PhononMode(100.0, np.array([[1, 0, 0], [-1, 0, 0]], float)))
    animator.set_frame(np.pi / 2)  # animates fine at the small size

    supercell = Structure.from_ase(bulk("NaCl", "rocksalt", a=5.64).repeat((2, 2, 2)))  # 16
    renderer.set_structure(supercell)
    assert renderer.atom_count == 16

    animator.reset()  # would raise ValueError without the size guard
    animator.set_frame(np.pi / 2)  # ditto

    # once the matching equilibrium is supplied, animation resumes
    animator.set_mode(supercell.positions, PhononMode(100.0, np.ones((16, 3))))
    animator.set_frame(np.pi / 2)
