<p align="center">
  <img src="docs/logo.png" alt="CRYSTALLine" width="440">
</p>

<p align="center">
  A desktop app for building, editing and visualising <b>CRYSTAL</b> structures
  and their <b>vibrational (phonon) modes</b> — built on
  <a href="https://github.com/crystaldevs/CRYSTALClear">CRYSTALClear</a>.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/GUI-PySide6-41cd52" alt="PySide6">
  <img src="https://img.shields.io/badge/3D-PyVista%2FVTK-orange" alt="PyVista">
  <img src="https://img.shields.io/badge/platforms-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey" alt="Cross-platform">
  <img src="https://img.shields.io/badge/license-GPLv3-green" alt="GPLv3 license">
</p>

---

## Features

**Structure viewer**
- Interactive 3D view (PyVista/VTK): ball-and-stick atoms, bonds, hydrogen
  bonds (dashed D–H···A contacts), coordination polyhedra (VESTA-style, shown
  by default), the unit-cell wireframe and an a/b/c gizmo.
- One-click view alignment down the **a**, **b** or **c** axis.
- A rich, dockable **Display** panel: atom size/opacity, per-element colours,
  bond radius/tolerance, hydrogen bonds, cell & axes, polyhedra, measurement
  colours, background colour, projection, an orientation marker and element labels.
- Crystallographic **Info** panel, dimensionality-aware: space group(3D) or
  layer group(2D slabs), point group, lattice parameters, cell volume/area,
  density and formula — recomputed live as you edit — plus CRYSTAL-computed
  properties (total energy, band gap, Fermi energy) read from the output.

**Geometry & measurements**
- Measure a selection: distance (2 atoms), angle (3), dihedral (4) or
  a least-squares plane (3+); mark single-atom points.
- Overlay measurements in 3D and colour them per item or by type default.

**Phonons**
- Loads vibrational modes automatically when the CRYSTAL output has them.
- Animate any mode in place — bonds, polyhedra and hydrogen bonds follow the
  motion; export the animation as GIF / MP4 / PNG frames with configurable
  resolution, frame count and frame rate.

**Editing**
- Select atoms (click / Ctrl-click), drag them in 3D (periodic images move
  together), drag a whole selection as one piece, or nudge with the arrow keys.
- Add, delete, duplicate, translate and re-element atoms — with a
  visual periodic-table, element picker — and full undo / redo.
- Cell tools: conventional cell, supercells, boundary completion and editable
  lattice parameters.

**Import / export**
- Open CRYSTAL `.out` / `.gui` / `.f34` files and `.cif`** structures.
- Import atoms from `.xyz` / `.pdb` / `.cif` into the current structure.
- Save the structure as `.gui` or `.cif` (symmetry-reduced).
- Export the 3D view as an image (PNG/JPEG/TIFF/SVG/PDF/EPS) with resolution and
  transparency options.

**Input builder**
- Write a ready-to-run CRYSTAL `.d12` deck for the current structure, with a live
  preview of the exact input before you save it.
- Geometry is derived from the structure — space group and asymmetric unit for a
  crystal, and the right coordinate convention for slabs, polymers and molecules.
- Choose the method (HF or DFT, one functional keyword or separate exchange and
  correlation), basis set, SCF settings and the calculation: single point,
  geometry optimisation, frequencies with IR/Raman, phonon dispersion, QHA,
  equation of state, elastic constants, CPHF, anharmonic runs and spin–orbit
  coupling.

**Property plots** (via CRYSTALClear, shown in a dockable tabbed panel)
- IR & Raman spectra, elastic surfaces (Young's modulus, linear compressibility,
  shear, Poisson), equation of state.
- Electronic & phonon band structures and densities of states, simulated XRD.

## Screenshots

<p align="center">
  <img src="docs/screen1.png" alt="Main window" width="820"><br>
  <em>The main window — crystallographic info, the interactive 3D view and the
  phonon-mode list (brucite, Mg(OH)₂).</em>
</p>

<p align="center">
  <img src="docs/screen2.png" alt="Raman spectrum and animation export" width="820"><br>
  <em>A computed Raman spectrum beside the structure, with the phonon-animation
  export dialog (thiourea).</em>
</p>

<p align="center">
  <img src="docs/screen3.png" alt="Display panel, polyhedra and elastic surface" width="820"><br>
  <em>The Display panel and coordination polyhedra on a 2×2×2 supercell, with a
  Young's modulus elastic surface.</em>
</p>

## Architecture

The package is deliberately layered so the domain logic stays independent of the
Qt UI — and therefore unit-testable without a display:

```
src/crystalline/
├── core/        domain model — Structure, phonons, cells, bonds, undo (no Qt)
├── crystalio/   thin adapter over CRYSTALClear (load/save, property plots)
├── viz/         PyVista/VTK rendering, phonon animation, image/movie export (no Qt)
├── ui/          PySide6 widgets: 3D viewport, dockable panels, main window
└── resources/   bundled assets (logo)
```

Only `ui/` (and the viewport that embeds the VTK interactor) imports Qt. Adding a
new property (a new plot, panel, …) is a matter of dropping a widget into
`ui/panels/` and wiring its signals in `MainWindow`.

## Usage

```sh
python -m crystalline
# or, after install, just:
crystalline
```

Use **File → Open** to load a CRYSTAL `.out`/`.gui`/`.34` file or a `.cif`. If a
CRYSTAL output contains a vibrational calculation, the phonon modes are loaded
too — pick one in the **Phonons** panel and press **Play**. Otherwise the
geometry is shown on its own. Tweak the look from the **Display** panel, measure
geometry from the **Geometry** panel, and build property plots from the **Plot**
menu.

Enable **Edit → Editing mode** (`Ctrl+E`) to edit atoms: click to select, drag or
**arrow-key** the selection to move it, `Del` to delete, and pick elements from
the visual periodic table.

CRYSTALLine runs on **Linux, macOS and Windows** — anywhere PySide6 and a working
OpenGL/VTK stack are available.

## Testing

```sh
pytest
```

The suite runs headless and exercises the Qt-free core, the renderer (via an
off-screen PyVista plotter), the plotting adapter and the file I/O.

## License

[GNU General Public License v3.0 or later](LICENSE).

## Acknowledgements

Built on the [CRYSTALClear](https://github.com/crystaldevs/CRYSTALClear) I/O and
plotting framework for the [CRYSTAL](https://www.crystal.unito.it/) quantum
chemistry code.

This software was developed with the assistance of
[Claude](https://www.anthropic.com/claude) (Anthropic), using
[Claude Code](https://www.claude.com/product/claude-code).
