# CRYSTALClear integration notes

Notes on quirks of the CRYSTALClear dependency that shape our
[`crystalio` adapter](../src/crystalline/crystalio/loader.py).

## 1. Broken `convert` convenience wrappers (v0.2.15)

The high-level helpers in `CRYSTALClear/convert.py` — `cry_out2ase`,
`cry_gui2ase`, `cry_out2pmg`, `cry_gui2pmg`, `cry_ase2gui`, `cry_out2xyz`, … —
use **unqualified imports** of sibling modules, e.g. line 278:

```python
def cry_out2ase(output, ...):
    from convert import cry_out2pmg      # ← should be: from CRYSTALClear.convert import ...
```

In a normal `pip install CRYSTALClear`, the package directory is *not* on
`sys.path`, so these raise:

```
ModuleNotFoundError: No module named 'convert'
```

They only "work" when a script is run from *inside* the CRYSTALClear source
tree. Affected files include `convert.py` (≈10 sites), `thermodynamics.py`, and
`unit_test.py` (`from convert import …`, `from crystal_io import …`,
`from units import …`, `from geometry import …`).

**Our workaround:** never call the `convert` wrappers. Use the properly-qualified
class API instead and do the ase conversion ourselves:

| Need | Route we use |
|---|---|
| `.out` → structure | `Crystal_output(path).get_geometry()` → pymatgen → `AseAtomsAdaptor` |
| `.gui` → structure | `Crystal_gui().read_gui(path)` → build `ase.Atoms` from `atom_number` / `atom_positions` / `lattice` |
| structure → `.gui`  | `AseAtomsAdaptor` → pymatgen → `cry_pmg2gui` (this one is import-safe) |

**Upstream fix** (for the CRYSTALClear repo, not this project): make those
imports absolute, e.g. `from CRYSTALClear.convert import cry_out2pmg`.

## 2. Phonon eigenvector shape & dtype

`Crystal_output.get_phonon(read_eigvt=True)` sets:

* `frequency` — shape `(nqpoint, nmode)`, units **THz**.
* `eigenvector` — per q-point, per mode the displacement comes back **flattened
  as `(3*natom,)`** (not `(natom, 3)`) and is **complex**.

For animating Gamma-point modes we take the **real part** and `reshape(natom, 3)`.
`natom` is taken from the parsed geometry (`3*natom == nmode` at Gamma).
