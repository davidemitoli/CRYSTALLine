"""PyVista-based 3D rendering and animation.

``renderer`` and ``phonon_animator`` operate on a PyVista plotter passed in by
the caller, so they stay independent of *which* Qt interactor embeds it. Qt is
only introduced one layer up, in ``ui.viewport``.
"""

from crystalline.viz.renderer import StructureRenderer
from crystalline.viz.phonon_animator import PhononAnimator

__all__ = ["StructureRenderer", "PhononAnimator"]
