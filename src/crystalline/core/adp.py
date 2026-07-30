"""Atomic displacement parameters: the thermal ellipsoid of each atom.

An ADP is the mean-square displacement tensor ``U = <u u^T>`` of an atom about
its site, in Angstrom squared. Drawn as the surface enclosing a chosen
probability of finding the atom, it is the ellipsoid every published crystal
structure carries — and the one quantity a calculation and a diffraction
refinement can be compared on directly.

Two coordinate conventions matter and are easy to confuse:

* **cartesian** ``U`` — what CRYSTAL prints and what a renderer needs, since
  the ellipsoid is drawn in real space;
* **crystallographic** ``U_ij`` — what a CIF stores, defined through the
  structure factor as ``T(h) = exp(-2 pi^2 sum_ij U_ij h_i h_j a*_i a*_j)``.

They agree only for a cell whose axes are orthogonal and aligned with the
cartesian frame, so for anything else the conversion in
:func:`to_crystallographic` is not optional. ``U_eq``, being a trace, is the
same in both.

Qt- and PyVista-free, like the rest of ``core``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

# Crystallographic default: the surface enclosing half the probability density.
DEFAULT_PROBABILITY = 0.5


@dataclass(frozen=True)
class ADPSet:
    """ADP tensors for every atom, at each temperature the run reported.

    Attributes
    ----------
    temperatures:
        ``(ntemp,)`` in K, in the order the output printed them.
    tensors:
        ``(ntemp, natom, 3, 3)`` cartesian ADP tensors in Angstrom squared.
    """

    temperatures: np.ndarray
    tensors: np.ndarray

    def __post_init__(self) -> None:
        temperatures = np.asarray(self.temperatures, dtype=float).ravel()
        tensors = np.asarray(self.tensors, dtype=float)
        if tensors.ndim != 4 or tensors.shape[0] != len(temperatures) or tensors.shape[2:] != (3, 3):
            raise ValueError(
                f"tensors must be (ntemp, natom, 3, 3) with ntemp={len(temperatures)}, "
                f"got {tensors.shape}"
            )
        object.__setattr__(self, "temperatures", temperatures)
        object.__setattr__(self, "tensors", tensors)

    def __len__(self) -> int:
        return len(self.temperatures)

    @property
    def n_atoms(self) -> int:
        return self.tensors.shape[1]

    def at(self, index: int) -> np.ndarray:
        """The ``(natom, 3, 3)`` tensors at temperature ``index`` (clamped)."""
        if len(self) == 0:
            return np.empty((0, 3, 3))
        return self.tensors[int(np.clip(index, 0, len(self) - 1))]

    def nearest(self, temperature: float) -> int:
        """Index of the reported temperature closest to ``temperature`` (K)."""
        if len(self) == 0:
            return 0
        return int(np.argmin(np.abs(self.temperatures - float(temperature))))

    def label(self, index: int) -> str:
        """``"300 K"`` for the temperature at ``index`` — for a picker."""
        if len(self) == 0:
            return ""
        value = self.temperatures[int(np.clip(index, 0, len(self) - 1))]
        return f"{value:g} K"


def to_crystallographic(u_cart: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Cartesian ADP tensor(s) -> the ``U_ij`` a CIF stores.

    ``cell`` holds the lattice vectors as **rows** (the ase/CRYSTALLine
    convention). Accepts a single ``(3, 3)`` tensor or a stack of them, and
    returns the same shape.

    The derivation, once, so the transposes can be checked rather than trusted.
    With ``A`` the matrix whose *columns* are the lattice vectors (so ``A =
    cell.T``), a cartesian displacement is ``u = A f`` for fractional ``f``,
    hence ``U_frac = A^-1 U A^-T``. Matching the CIF's exponent against
    ``-1/2 Q^T U Q`` with ``Q = 2 pi A^-T h`` gives ``N U_ij N = U_frac`` where
    ``N = diag(a*, b*, c*)``, so ``U_ij = N^-1 U_frac N^-1``. For an orthogonal
    cell aligned with the cartesian axes every factor cancels and ``U_ij ==
    U_cart``, which is the cheapest way to check an implementation.

    A degenerate (non-invertible) cell has no fractional basis to convert into,
    and raises.
    """
    cell = np.asarray(cell, dtype=float)
    if cell.shape != (3, 3) or abs(np.linalg.det(cell)) < 1e-12:
        raise ValueError("a non-degenerate 3x3 cell is required to convert ADPs")

    inverse = np.linalg.inv(cell)          # columns are the reciprocal vectors
    reciprocal_lengths = np.linalg.norm(inverse, axis=0)
    # U_frac = A^-1 U A^-T with A = cell.T, i.e. inverse.T @ U @ inverse.
    scale = np.outer(reciprocal_lengths, reciprocal_lengths)
    return _apply(u_cart, lambda u: (inverse.T @ u @ inverse) / scale)


def to_cartesian(u_ij: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """The inverse of :func:`to_crystallographic` — CIF ``U_ij`` -> cartesian.

    Needed to draw the ellipsoids of a structure *read* from a CIF alongside
    computed ones.
    """
    cell = np.asarray(cell, dtype=float)
    if cell.shape != (3, 3) or abs(np.linalg.det(cell)) < 1e-12:
        raise ValueError("a non-degenerate 3x3 cell is required to convert ADPs")

    inverse = np.linalg.inv(cell)
    reciprocal_lengths = np.linalg.norm(inverse, axis=0)
    scale = np.outer(reciprocal_lengths, reciprocal_lengths)
    return _apply(u_ij, lambda u: cell.T @ (u * scale) @ cell)


def equivalent_isotropic(u_cart: np.ndarray) -> np.ndarray:
    """``U_eq = trace(U)/3`` — the isotropic ADP equivalent to the tensor.

    A trace, so it is the same whichever orthonormal frame ``U`` is written in,
    and it is what a structure report quotes as a single number per atom.
    """
    u_cart = np.asarray(u_cart, dtype=float)
    return np.trace(u_cart, axis1=-2, axis2=-1) / 3.0


def probability_scale(probability: float = DEFAULT_PROBABILITY) -> float:
    """How far out to draw the ellipsoid to enclose ``probability``.

    The displacement is a trivariate Gaussian with covariance ``U``, so
    ``u^T U^-1 u`` follows chi-squared with three degrees of freedom and the
    surface enclosing probability ``p`` sits at ``sqrt(chi2.ppf(p, 3))``. The
    crystallographic 50% gives 1.5382 — the number ORTEP and VESTA use.
    """
    from scipy.stats import chi2

    probability = float(probability)
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be strictly between 0 and 1")
    return float(np.sqrt(chi2.ppf(probability, 3)))


def ellipsoid_axes(
    u_cart: np.ndarray, probability: float = DEFAULT_PROBABILITY
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """``(radii, axes, positive_definite)`` for one atom's ellipsoid.

    ``radii`` are the three semi-axis lengths in Angstrom at the requested
    probability; ``axes`` has the corresponding principal directions as **rows**
    (a proper rotation, so it can be used as a drawing transform directly).

    A tensor with a non-positive eigenvalue is not a physical displacement
    distribution — crystallographers call such an atom "NPD", and it happens
    with under-converged sampling as readily as with bad data. Rather than
    raising or drawing an imaginary axis, the offending eigenvalue is clamped
    to zero (a flat ellipsoid, which is what the number is saying) and the
    returned flag is False so a caller can mark it.
    """
    u_cart = np.asarray(u_cart, dtype=float)
    if u_cart.shape != (3, 3):
        raise ValueError(f"expected a (3, 3) tensor, got {u_cart.shape}")

    # eigh needs symmetry; CRYSTAL prints it symmetric, but a tensor that has
    # been converted between bases can pick up rounding asymmetry.
    eigenvalues, eigenvectors = np.linalg.eigh((u_cart + u_cart.T) / 2.0)
    positive_definite = bool(np.all(eigenvalues > 0.0))
    radii = probability_scale(probability) * np.sqrt(np.clip(eigenvalues, 0.0, None))

    axes = eigenvectors.T  # eigh returns eigenvectors as columns
    if np.linalg.det(axes) < 0:  # keep it a rotation, not a reflection
        axes[0] = -axes[0]
    return radii, axes, positive_definite


def ellipsoid_radii(
    u_cart: np.ndarray, probability: float = DEFAULT_PROBABILITY
) -> np.ndarray:
    """``(natom, 3)`` ascending semi-axis lengths for a *stack* of ADP tensors.

    The batched counterpart of :func:`ellipsoid_axes`'s first return value, for
    callers that need every atom's ellipsoid *size* but not its orientation —
    deciding which atoms are big enough to draw, above all. Equivalent to
    ``[ellipsoid_axes(u, probability)[0] for u in u_cart]`` and used in its place
    because that loop is not cheap at scale: it re-solves the chi-squared
    quantile and a 3x3 eigenproblem per atom, and the renderer runs it on every
    animation frame. ``numpy`` diagonalises the whole stack at once, which
    measured ~80x faster at 2000 atoms (89 ms -> 1.1 ms).

    Non-positive eigenvalues are clamped to zero exactly as
    :func:`ellipsoid_axes` does, so an NPD atom reports a flat ellipsoid rather
    than an imaginary axis.
    """
    tensors = np.asarray(u_cart, dtype=float)
    if tensors.ndim != 3 or tensors.shape[1:] != (3, 3):
        raise ValueError(f"expected (natom, 3, 3) tensors, got {tensors.shape}")
    if len(tensors) == 0:
        return np.empty((0, 3), dtype=float)

    # Symmetrise for the same reason ellipsoid_axes does: a tensor converted
    # between bases can pick up rounding asymmetry, and eigvalsh assumes none.
    symmetric = (tensors + np.swapaxes(tensors, -1, -2)) / 2.0
    eigenvalues = np.linalg.eigvalsh(symmetric)  # ascending, per tensor
    return probability_scale(probability) * np.sqrt(np.clip(eigenvalues, 0.0, None))


def _apply(tensors: np.ndarray, transform) -> np.ndarray:
    """Run ``transform`` over a single ``(3, 3)`` tensor or a stack of them."""
    tensors = np.asarray(tensors, dtype=float)
    if tensors.shape == (3, 3):
        return transform(tensors)
    if tensors.ndim < 2 or tensors.shape[-2:] != (3, 3):
        raise ValueError(f"expected (..., 3, 3) tensors, got {tensors.shape}")
    flat = tensors.reshape(-1, 3, 3)
    return np.stack([transform(u) for u in flat]).reshape(tensors.shape)


__all__ = [
    "ADPSet",
    "DEFAULT_PROBABILITY",
    "ellipsoid_axes",
    "ellipsoid_radii",
    "equivalent_isotropic",
    "probability_scale",
    "to_cartesian",
    "to_crystallographic",
]
