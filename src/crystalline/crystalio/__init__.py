"""Adapters over CRYSTALClear for loading/saving CRYSTAL files.

Named ``crystalio`` (not ``io``) to avoid shadowing Python's stdlib ``io``.
This is the *only* place CRYSTALLine touches CRYSTALClear's API, so if that API
shifts, changes are localised here rather than spread through the UI.
"""

from crystalline.crystalio.loader import (
    LoadedFile,
    has_phonons,
    load,
    load_structure,
    load_phonons,
    output_properties,
    read_atoms,
    save_structure_gui,
    save_structure_cif,
)
from crystalline.crystalio.plotting import (
    PlotKind,
    PlotUnavailable,
    available_plots,
    crystalclear_available,
    output_availability,
)

__all__ = [
    "load",
    "LoadedFile",
    "has_phonons",
    "load_structure",
    "load_phonons",
    "output_properties",
    "read_atoms",
    "save_structure_gui",
    "save_structure_cif",
    "PlotKind",
    "PlotUnavailable",
    "available_plots",
    "crystalclear_available",
    "output_availability",
]
