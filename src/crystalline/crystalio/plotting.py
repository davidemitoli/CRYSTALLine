"""Adapter over ``CRYSTALClear.plot``: turn CRYSTAL results into matplotlib
figures the UI can embed.

Two kinds of source, distinguished by :attr:`PlotKind.source`:

* ``"output"`` — read straight from the CRYSTAL ``.out`` via ``Crystal_output``
  (IR/Raman spectra, elastic surfaces, equation of state). These need no extra
  files, so the UI reuses the already-loaded output.
* ``"data"`` — need a PROPERTIES/dispersion data file (``BAND.DAT``, ``DOSS.DAT``,
  ``fort.25``, ``*SPEC.DAT``): electron/phonon bands & DOS, XRD.

Kept Qt-free (returns a bare ``matplotlib.figure.Figure``) so the mapping from
"kind of plot" → "reader + plotter" lives in one testable place, mirroring how
:mod:`crystalline.crystalio.loader` is the single point of contact for the
geometry/phonon side of CRYSTALClear.

Each :class:`PlotKind` pairs a menu label (and, for data plots, a file filter)
with a ``build(path) -> Figure`` callable. Adding a plot is one entry in
:func:`available_plots`; the UI builds its menu straight from that list.

CRYSTALClear (and matplotlib) are imported lazily inside the builders so the
rest of the package stays importable — and unit-testable — without them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional


class PlotUnavailable(RuntimeError):
    """CRYSTALClear (or a plotting dependency) isn't installed."""


@dataclass(frozen=True)
class PlotKind:
    """One entry in the Plot menu: how to turn a file into a figure.

    Attributes
    ----------
    key:      stable identifier (also handy in tests).
    label:    menu text.
    source:   ``"output"`` (a CRYSTAL ``.out``) or ``"data"`` (a PROPERTIES
              data file). Output plots reuse the loaded output; data plots prompt.
    caption:  the file dialog's title (used when a file must be chosen).
    file_filter: Qt ``getOpenFileName`` filter for the expected file.
    build:    ``path -> matplotlib.figure.Figure``.
    group:    optional submenu name (e.g. "Elastic properties").
    """

    key: str
    label: str
    source: str
    caption: str
    file_filter: str
    build: Callable[[str], "object"]
    group: Optional[str] = None
    # For ``source="output"``: ``(getter, attribute)`` used to test whether the
    # loaded output actually contains this plot's data (see output_availability).
    probe: Optional[tuple] = None


# ── figure extraction ──────────────────────────────────────────────────────
def _to_figure(result):
    """Normalise a CRYSTALClear plot function's return into a single Figure.

    The plot functions are inconsistent: most return ``(fig, ax)``, some return
    a bare ``fig``, a few draw onto the pyplot state and return ``None``, and
    ``plot_cry_ela`` returns ``(fig_list, ax_list, plt_list)``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    if isinstance(result, Figure):
        return result
    if isinstance(result, tuple) and result:
        first = result[0]
        if isinstance(first, Figure):
            return first
        if isinstance(first, (list, tuple)) and first and isinstance(first[0], Figure):
            return first[0]
    return plt.gcf()  # plt-based plotters (e.g. plot_cry_xrd) draw on the current fig


def _require_crystalclear():
    try:
        from CRYSTALClear import crystal_io, plot  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - surface a clear, actionable message
        raise PlotUnavailable(
            "CRYSTALClear is required for plotting but could not be imported "
            f"({exc}). Install it with `pip install CRYSTALClear`."
        ) from exc


def _plot_module():
    """Import CRYSTALClear.plot with a non-interactive matplotlib backend.

    Agg keeps CRYSTALClear's internal ``plt.show()`` (e.g. in ``plot_cry_xrd``)
    from popping a stray window; the figure is embedded in Qt afterwards, which
    is backend-independent.
    """
    _require_crystalclear()
    import matplotlib

    matplotlib.use("Agg")
    from CRYSTALClear import plot as CCplt

    return CCplt


# ── builders: from a CRYSTAL .out via Crystal_output ────────────────────────
def _spectrum_builder(getter: str, attribute: str) -> Callable[[str], object]:
    """IR/Raman: run ``Crystal_output.getter()`` then broaden the stick spectrum.

    ``getter`` populates ``attribute`` (an ``(N, 2)`` freq/intensity array) that
    ``plot_cry_spec`` turns into a lineshape.
    """

    def build(path: str):
        CCplt = _plot_module()
        from CRYSTALClear.crystal_io import Crystal_output

        co = Crystal_output(path)
        getattr(co, getter)()
        transitions = getattr(co, attribute)
        return _to_figure(CCplt.plot_cry_spec(transitions))

    return build


def _elastic_builder(choose: str, ndeg: int = 100) -> Callable[[str], object]:
    """Elastic surface (Young/compressibility/shear/Poisson) from the elastic tensor."""

    def build(path: str):
        CCplt = _plot_module()
        from CRYSTALClear.crystal_io import Crystal_output

        co = Crystal_output(path)
        co.get_elatensor()
        return _to_figure(CCplt.plot_cry_ela(co, choose, ndeg))

    return build


def _elastic_2d_builder(choose: str, ndeg: int = 200) -> Callable[[str], object]:
    """Elastic polar sections through the xy/xz/yz planes (the 2D analogue).

    Same properties as the 3D surfaces, drawn as one polar figure overlaying the
    three principal planes. Needs a CRYSTALClear providing ``plot_cry_ela_2D``
    (see :func:`_has_plot_function`), which is why the menu entries are
    registered conditionally.
    """

    def build(path: str):
        CCplt = _plot_module()
        from CRYSTALClear.crystal_io import Crystal_output

        co = Crystal_output(path)
        co.get_elatensor()
        return _to_figure(CCplt.plot_cry_ela_2D(co, choose, ndeg))

    return build


def _has_plot_function(name: str) -> bool:
    """Whether the installed CRYSTALClear exposes ``plot.<name>``.

    Lets the menu offer plots that only newer CRYSTALClear builds provide,
    without breaking against one that doesn't have them — a missing function
    simply means the entry isn't listed.
    """
    try:
        from CRYSTALClear import plot as CCplt
    except Exception:  # noqa: BLE001 - CRYSTALClear missing/broken: no extras
        return False
    return hasattr(CCplt, name)


def _eos_builder() -> Callable[[str], object]:
    """Equation-of-state E(V) fit from an EOS run."""

    def build(path: str):
        CCplt = _plot_module()
        from CRYSTALClear.crystal_io import Crystal_output

        co = Crystal_output(path)
        co.get_EOS()
        return _to_figure(CCplt.plot_cry_EOS(co))

    return build


# ── builders: from a PROPERTIES/dispersion data file ────────────────────────
def _prop_builder(read_method: str, plot_func: str) -> Callable[[str], object]:
    """Read with ``Properties_output`` and plot with ``plot_func``."""

    def build(path: str):
        CCplt = _plot_module()
        from CRYSTALClear.crystal_io import Properties_output

        reader = Properties_output()
        data = getattr(reader, read_method)(path)
        obj = data if data is not None else reader  # readers may return self/None
        return _to_figure(getattr(CCplt, plot_func)(obj))

    return build


def _external_builder(read_method: str, plot_func: str) -> Callable[[str], object]:
    """Read with ``External_unit`` and plot with ``plot_func``."""

    def build(path: str):
        CCplt = _plot_module()
        from CRYSTALClear.crystal_io import External_unit

        reader = External_unit()
        data = getattr(reader, read_method)(path)
        obj = data if data is not None else reader
        return _to_figure(getattr(CCplt, plot_func)(obj))

    return build


# ── registry ────────────────────────────────────────────────────────────────
_OUT = "CRYSTAL output (*.out *.outp);;All files (*)"
_DAT = "Data files (*.DAT *.dat *.f25 fort.25);;All files (*)"


def available_plots() -> List[PlotKind]:
    """The plot kinds CRYSTALLine offers, in menu order.

    Output-file plots (IR/Raman/elastic/EOS) read the loaded ``.out`` directly;
    data-file plots (bands/DOS/XRD) prompt for the matching PROPERTIES file.

    The 2D elastic sections are listed only when the installed CRYSTALClear can
    draw them (it needs ``plot_cry_ela_2D``), so an older build just shows the
    3D surfaces instead of offering menu entries that would fail.
    """
    return [
        # ── from the CRYSTAL output file ──
        PlotKind("ir", "IR spectrum", "output", "Open CRYSTAL output (.out)", _OUT,
                 _spectrum_builder("get_IR", "IR_HO_0K"),
                 probe=("get_IR", "IR_HO_0K")),
        PlotKind("raman", "Raman spectrum", "output", "Open CRYSTAL output (.out)", _OUT,
                 _spectrum_builder("get_Raman", "Ram_HO_0K_tot"),
                 probe=("get_Raman", "Ram_HO_0K_tot")),
        PlotKind("ela_young", "Young's modulus", "output", "Open CRYSTAL output (.out)", _OUT,
                 _elastic_builder("young"), group="Elastic properties",
                 probe=("get_elatensor", "elatensor")),
        PlotKind("ela_comp", "Linear compressibility", "output", "Open CRYSTAL output (.out)", _OUT,
                 _elastic_builder("comp"), group="Elastic properties",
                 probe=("get_elatensor", "elatensor")),
        PlotKind("ela_shear", "Shear modulus (avg)", "output", "Open CRYSTAL output (.out)", _OUT,
                 _elastic_builder("shear avg"), group="Elastic properties",
                 probe=("get_elatensor", "elatensor")),
        PlotKind("ela_poisson", "Poisson's ratio (avg)", "output", "Open CRYSTAL output (.out)", _OUT,
                 _elastic_builder("poisson avg"), group="Elastic properties",
                 probe=("get_elatensor", "elatensor")),
        *_elastic_2d_plots(),
        PlotKind("eos", "Equation of state", "output", "Open CRYSTAL output (.out)", _OUT,
                 _eos_builder(), probe=("get_EOS", "VvsE")),
        # ── from a PROPERTIES / dispersion data file ──
        PlotKind("electron_band", "Electronic band structure…", "data",
                 "Open BAND.DAT / fort.25", _DAT,
                 _prop_builder("read_electron_band", "plot_electron_band")),
        PlotKind("electron_dos", "Electronic density of states…", "data",
                 "Open DOSS.DAT / fort.25", _DAT,
                 _prop_builder("read_electron_dos", "plot_electron_dos")),
        PlotKind("phonon_band", "Phonon band structure…", "data",
                 "Open PHONBAND.DAT / fort.25", _DAT,
                 _external_builder("read_phonon_band", "plot_phonon_band")),
        PlotKind("phonon_dos", "Phonon density of states…", "data",
                 "Open PHONDOSS.DAT / fort.25", _DAT,
                 _external_builder("read_phonon_dos", "plot_phonon_dos")),
        PlotKind("xrd", "Simulated XRD pattern…", "data",
                 "Open XRD spectrum (XRDATO.DAT)", _DAT,
                 _prop_builder("read_cry_xrd_spec", "plot_cry_xrd")),
    ]


_ELASTIC_2D_GROUP = "Elastic properties (2D)"


def _elastic_2d_plots() -> List[PlotKind]:
    """The 2D elastic sections, or nothing if CRYSTALClear can't draw them.

    Mirrors the four 3D surfaces so the two submenus read the same way; the
    ``choose`` strings are exactly those ``plot_cry_ela_2D`` accepts.
    """
    if not _has_plot_function("plot_cry_ela_2D"):
        return []
    probe = ("get_elatensor", "elatensor")
    return [
        PlotKind("ela2d_young", "Young's modulus", "output", "Open CRYSTAL output (.out)", _OUT,
                 _elastic_2d_builder("young"), group=_ELASTIC_2D_GROUP, probe=probe),
        PlotKind("ela2d_comp", "Linear compressibility", "output",
                 "Open CRYSTAL output (.out)", _OUT,
                 _elastic_2d_builder("comp"), group=_ELASTIC_2D_GROUP, probe=probe),
        PlotKind("ela2d_shear", "Shear modulus (avg)", "output",
                 "Open CRYSTAL output (.out)", _OUT,
                 _elastic_2d_builder("shear avg"), group=_ELASTIC_2D_GROUP, probe=probe),
        PlotKind("ela2d_poisson", "Poisson's ratio (avg)", "output",
                 "Open CRYSTAL output (.out)", _OUT,
                 _elastic_2d_builder("poisson avg"), group=_ELASTIC_2D_GROUP, probe=probe),
    ]


def crystalclear_available() -> bool:
    """Whether CRYSTALClear can be imported (so the UI can hint if not)."""
    try:
        _require_crystalclear()
        return True
    except PlotUnavailable:
        return False


def _probe_output(path: str, getter: str, attribute: str) -> bool:
    """Whether ``Crystal_output(path).getter()`` yields non-empty ``attribute``.

    Any parse/extraction failure (the data simply isn't in this run) counts as
    "not available". CRYSTALClear's chatter is swallowed so probing is silent.
    """
    import contextlib
    import io

    from CRYSTALClear.crystal_io import Crystal_output

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            co = Crystal_output(path)
            getattr(co, getter)()
        value = getattr(co, attribute, None)
        return value is not None and (not hasattr(value, "__len__") or len(value) > 0)
    except Exception:  # noqa: BLE001 - absent data raises in various ways
        return False


def output_availability(path: Optional[str]) -> set:
    """Keys of the output-file plots whose data is present in ``path``.

    Returns an empty set when ``path`` is falsy (no output loaded) or CRYSTALClear
    is unavailable. Each distinct ``(getter, attribute)`` probe is run once even
    though several elastic plots share it. Fast (~0.05 s for a typical output).
    """
    if not path:
        return set()
    try:
        _require_crystalclear()
    except PlotUnavailable:
        return set()

    cache: dict = {}
    available = set()
    for kind in available_plots():
        if kind.source != "output" or not kind.probe:
            continue
        if kind.probe not in cache:
            cache[kind.probe] = _probe_output(path, *kind.probe)
        if cache[kind.probe]:
            available.add(kind.key)
    return available


__all__ = [
    "PlotKind",
    "PlotUnavailable",
    "available_plots",
    "crystalclear_available",
    "output_availability",
]
