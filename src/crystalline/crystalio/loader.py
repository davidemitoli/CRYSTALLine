"""Load CRYSTAL files into CRYSTALLine domain objects via CRYSTALClear.

Implementation note — CRYSTALClear's ``convert`` convenience wrappers
(``cry_out2ase``, ``cry_gui2ase``, ``cry_ase2gui``) are unusable in a normal
install of CRYSTALClear 0.2.15: internally they do unqualified imports like
``from convert import cry_out2pmg`` which raise ``ModuleNotFoundError: No module
named 'convert'`` unless the package's own directory happens to be on
``sys.path``. We therefore go through the *properly-qualified* class API
(``Crystal_output.get_geometry``, ``Crystal_gui.read_gui``, ``cry_pmg2gui``) and
do the ase conversion ourselves. See ``docs/CRYSTALClear_notes.md``.

CRYSTALClear is imported lazily (inside function bodies) so ``core`` and the
test suite stay importable without it.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
from ase import Atoms

from crystalline.core.phonons import PhononMode, PhononModes
from crystalline.core.structure import Structure

# CRYSTALClear reports phonon frequencies in THz; the domain model stores cm^-1.
# 1 THz = 1e12 Hz -> nu[cm^-1] = 1e12 / c[cm/s] = 1e12 / 2.99792458e10.
THZ_TO_CM = 33.356409519815204

# Lines that mark a CRYSTAL frequency (FREQCALC) run — the same header
# CRYSTALClear itself keys on when parsing phonons.
_FREQ_MARKERS = (
    re.compile(r"\+\+\+\s*SYMMETRY ADAPTION OF VIBRATIONAL MODES"),
    re.compile(r"MODES\s+EIGV\s+FREQUENCIES\s+IRREP"),
)


@dataclass
class LoadedFile:
    """Result of :func:`load`: a structure and, if present, its phonon modes."""

    structure: Structure
    modes: Optional[PhononModes] = None

    @property
    def has_phonons(self) -> bool:
        return self.modes is not None and len(self.modes) > 0


def has_phonons(path: str) -> bool:
    """Whether a CRYSTAL ``.out`` file contains a vibrational calculation.

    Returns False for ``.gui`` files (geometry only) and for outputs without a
    frequency table. Scans the text for CRYSTAL's frequency-run markers.
    """
    if _is_gui(path):
        return False
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            if any(m.search(line) for m in _FREQ_MARKERS):
                return True
    return False


def load(path: str, initial: bool = False) -> LoadedFile:
    """Load a CRYSTAL file, auto-detecting whether it holds phonon modes.

    This is the single entry point the GUI uses: it returns the structure and,
    if the output is a frequency calculation, the phonon modes as well.
    """
    if not _is_gui(path) and not _is_cif(path) and has_phonons(path):
        structure, modes = load_phonons(path)
        return LoadedFile(structure=structure, modes=modes)
    return LoadedFile(structure=load_structure(path, initial=initial), modes=None)


def load_structure(path: str, initial: bool = False) -> Structure:
    """Load a structure from a CRYSTAL ``.out``/``.gui`` file or a ``.cif``.

    Parameters
    ----------
    path:
        A CRYSTAL ``.out`` output, a ``.gui``/``fort.34`` geometry file, or a
        crystallographic ``.cif``.
    initial:
        For ``.out`` files, read the initial geometry instead of the last one
        (relevant after a geometry optimisation). Ignored for ``.gui``/``.cif``.
    """
    if _is_gui(path):
        return Structure.from_ase(_gui_to_ase(path))
    if _is_cif(path):
        return Structure.from_ase(_cif_to_ase(path))
    return Structure.from_ase(_out_to_ase(path, initial=initial))


def load_phonons(
    path: str,
    qpoint: int = 0,
    keep_imaginary: bool = True,
) -> tuple[Structure, PhononModes]:
    """Load equilibrium geometry + phonon modes from a CRYSTAL ``.out`` file.

    Returns the structure the modes are defined on, alongside the modes for the
    requested q-point (Gamma by default). Imaginary/soft modes are kept by
    default so they remain animatable.

    Parameters
    ----------
    path:
        CRYSTAL ``.out`` file from a frequency calculation.
    qpoint:
        Index of the q-point to extract (0 = Gamma).
    keep_imaginary:
        If True, negative-frequency modes are retained (CRYSTALClear would
        otherwise NaN them out).
    """
    from CRYSTALClear.crystal_io import Crystal_output

    out = Crystal_output(path)
    out.get_phonon(read_eigvt=True, rm_imaginary=not keep_imaginary)

    structure = Structure.from_ase(_pmg_to_ase(out.get_geometry(initial=False)))
    natom = len(structure)

    # frequency: (nqpoint, nmode) in THz -> converted to cm^-1 below.
    # eigenvector at a q-point comes back per mode either flattened as
    # (3*natom,) or already (natom, 3), and may be complex (Gamma-point modes
    # are real up to a global phase -> take the real part).
    freqs = np.atleast_2d(np.asarray(out.frequency))[qpoint]
    eigvecs = np.asarray(out.eigenvector)[qpoint]

    # IR/Raman selection rules are only printed for the Gamma point.
    nmode = len(freqs)
    if qpoint == 0:
        ir = _per_mode(getattr(out, "IR", None), nmode)
        raman = _per_mode(getattr(out, "Raman", None), nmode)
        intens = _per_mode(getattr(out, "intens", None), nmode, cast=float)
    else:
        ir = raman = intens = [None] * nmode

    modes = [
        PhononMode(
            frequency=float(np.real(freqs[i])) * THZ_TO_CM,
            eigenvector=np.real(eigvecs[i]).reshape(natom, 3),
            ir_active=ir[i],
            raman_active=raman[i],
            ir_intensity=intens[i],
        )
        for i in range(nmode)
    ]

    return structure, PhononModes(modes)


def _per_mode(values, nmode: int, cast=bool) -> list:
    """Normalise a CRYSTALClear per-mode array to a length-``nmode`` list.

    ``out.IR`` / ``out.Raman`` / ``out.intens`` are empty when the output has no
    IR/Raman analysis (dispersion runs, or FREQCALC without intensities), and
    the modes then simply carry ``None`` rather than a wrong ``False``. A
    length mismatch is treated the same way — better unlabelled than misaligned.
    """
    if values is None:
        return [None] * nmode
    arr = np.asarray(values).ravel()
    if arr.size != nmode:
        return [None] * nmode
    out = []
    for v in arr:
        value = cast(v)
        out.append(None if isinstance(value, float) and not np.isfinite(value) else value)
    return out


def save_structure_gui(structure: Structure, gui_file: str, symmetry: bool = True) -> None:
    """Write a :class:`Structure` to a CRYSTAL ``.gui`` file.

    A periodic cell is written at its own dimensionality (3D bulk / 2D slab /
    1D polymer); a non-periodic one becomes a 0D (molecule) gui. If symmetry
    can't be determined, a ``P1`` gui is written rather than failing the save.

    ``pbc`` is always passed explicitly: left to itself, ``cry_pmg2gui``
    round-trips a molecule through ``Molecule(species, coords, charge,
    site_properties)`` — positional arguments that no longer line up with
    current pymatgen, so it raises.
    """
    from CRYSTALClear.convert import cry_pmg2gui  # qualified path: safe

    if _has_lattice(structure):
        obj = _to_pymatgen(structure)
        pbc = tuple(bool(p) for p in structure.pbc)
    else:
        obj = _to_pymatgen_molecule(structure)
        pbc = (False, False, False)

    try:
        cry_pmg2gui(obj, gui_file=gui_file, pbc=pbc, symmetry=symmetry)
    except Exception:  # noqa: BLE001 - symmetry undeterminable: write an unreduced P1 gui
        cry_pmg2gui(obj, gui_file=gui_file, pbc=pbc, symmetry=False)


def save_structure_cif(structure: Structure, cif_file: str, symmetry: bool = True) -> None:
    """Write a :class:`Structure` to a CIF file.

    For a periodic cell this uses pymatgen's ``CifWriter``; with ``symmetry`` it
    detects the space group (``symprec`` search) and writes the symmetry-reduced
    CIF, otherwise a plain ``P1`` listing. A non-periodic system (a molecule)
    has no lattice for a crystallographic CIF, so it falls back to ASE's writer.
    If symmetry can't be determined, a ``P1`` CIF is written rather than failing
    the save.
    """
    if not _has_lattice(structure):
        from ase.io import write  # non-periodic: no lattice for a crystallographic CIF

        write(cif_file, structure.to_ase(), format="cif")
        return

    from pymatgen.io.cif import CifWriter

    pmg = _to_pymatgen(structure)
    try:
        CifWriter(pmg, symprec=0.01 if symmetry else None).write_file(cif_file)
    except Exception:  # noqa: BLE001 - symmetry undeterminable: write an unreduced P1 CIF
        CifWriter(pmg, symprec=None).write_file(cif_file)


def read_atoms(path: str) -> Atoms:
    """Read atoms from an ``.xyz`` / ``.pdb`` / ``.cif`` file for importing.

    Returns a bare ``ase.Atoms`` (symbols + cartesian positions); the caller
    appends these into the current structure. The source file's own cell (if any)
    is intentionally ignored — the atoms are added to the structure already loaded.
    A multi-frame file (e.g. an XYZ trajectory) yields its first frame.
    """
    from ase.io import read

    result = read(path, index=0)  # first frame if the file holds several
    if isinstance(result, list):  # some formats/return an explicit list
        result = result[0]
    return result


def output_properties(path: str) -> dict:
    """Extract CRYSTAL-computed properties from a ``.out`` file, as (label, value).

    Returns an ordered ``{label: value_string}`` of whatever this run reported —
    total energy, band gap, Fermi energy. Empty for ``.gui`` files. Each getter
    is guarded independently, so one that CRYSTALClear can't parse (some runs)
    just omits its row rather than failing the whole panel.
    """
    if _is_gui(path):
        return {}
    from CRYSTALClear.crystal_io import Crystal_output

    out = Crystal_output(path)
    props: dict = {}

    def _try(label: str, fn, fmt) -> None:
        try:
            value = fn()
        except Exception:  # noqa: BLE001 - a getter failing must not break the rest
            return
        if value is None:
            return
        try:
            props[label] = fmt(value)
        except Exception:  # noqa: BLE001 - unexpected shape; skip this row
            return

    _try("Total energy (eV)", out.get_final_energy, lambda v: f"{float(v):.6f}")
    _try("Band gap (eV)", out.get_band_gap, lambda v: f"{float(np.ravel(v)[0]):.4f}")
    _try("Fermi energy (eV)", out.get_fermi_energy, lambda v: f"{float(np.ravel(v)[0]):.4f}")
    return props


# ── shared by the savers ──────────────────────────────────────────────────
def _has_lattice(structure: Structure) -> bool:
    """Whether ``structure`` should be written as a periodic cell at all.

    A structure with no periodic direction — or with a degenerate (all-zero)
    cell — is a molecule: there is no lattice for pymatgen to build on.
    """
    return bool(structure.is_periodic) and not np.allclose(structure.cell, 0.0)


def _to_pymatgen(structure: Structure):
    """A pymatgen ``Structure``, with boundary-completion duplicates merged out.

    The structure handed to a saver is the *displayed* cell, which may be
    "packed" (boundary atoms duplicated as periodic images). pymatgen's symmetry
    finder raises on those coincident sites, so they are collapsed first
    (``merge_sites`` returns a new Structure in current pymatgen; older versions
    mutate in place and return ``None``).
    """
    from pymatgen.io.ase import AseAtomsAdaptor

    pmg = AseAtomsAdaptor().get_structure(structure.to_ase())
    merged = pmg.merge_sites(tol=0.01, mode="delete")
    return pmg if merged is None else merged


def _to_pymatgen_molecule(structure: Structure):
    """A pymatgen ``Molecule`` — for systems with no lattice to speak of."""
    from pymatgen.core.structure import Molecule

    atoms = structure.to_ase()
    return Molecule(atoms.get_chemical_symbols(), atoms.get_positions())


# ── internal conversions (avoid CRYSTALClear.convert's broken wrappers) ────
def _is_gui(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext == ".gui" or os.path.basename(path).startswith("fort.34")


def _is_cif(path: str) -> bool:
    return os.path.splitext(path)[1].lower() == ".cif"


def _out_to_ase(path: str, initial: bool) -> Atoms:
    from CRYSTALClear.crystal_io import Crystal_output

    geom = Crystal_output(path).get_geometry(initial=initial)
    return _pmg_to_ase(geom)


def _cif_to_ase(path: str) -> Atoms:
    """Read a CIF into ase.Atoms via pymatgen (applies the file's symmetry).

    pymatgen's ``CifParser`` expands the asymmetric unit by the CIF's symmetry
    operations, so the full conventional cell is returned — matching what a
    crystallographer expects, and what the CRYSTAL readers already produce.
    """
    from pymatgen.io.cif import CifParser

    structures = CifParser(path).parse_structures(primitive=False)
    if not structures:
        raise ValueError("no structure found in the CIF file")
    return _pmg_to_ase(structures[0])


def _pmg_to_ase(pmg_obj) -> Atoms:
    """Convert a pymatgen Structure *or* Molecule to ase.Atoms."""
    from pymatgen.io.ase import AseAtomsAdaptor

    return AseAtomsAdaptor().get_atoms(pmg_obj)


def _gui_to_ase(path: str) -> Atoms:
    """Build ase.Atoms directly from a CRYSTAL .gui via Crystal_gui.read_gui."""
    from CRYSTALClear.crystal_io import Crystal_gui

    g = Crystal_gui().read_gui(path)
    numbers = [int(z) for z in g.atom_number]
    positions = np.asarray(g.atom_positions, dtype=float)  # cartesian Angstrom
    cell = np.asarray(g.lattice, dtype=float)
    dim = int(g.dimensionality)
    pbc = (dim >= 1, dim >= 2, dim >= 3)  # CRYSTAL: 3=bulk,2=slab,1=polymer,0=molecule
    return Atoms(numbers=numbers, positions=positions, cell=cell, pbc=pbc)


__all__ = [
    "load_structure",
    "load_phonons",
    "save_structure_gui",
    "save_structure_cif",
    "read_atoms",
    "output_properties",
]
