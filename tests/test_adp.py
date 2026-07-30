"""Atomic displacement parameters: coordinate conventions and ellipsoid geometry.

The cartesian/crystallographic conversion is the part with no visible failure
mode — a wrong transpose gives plausible-looking numbers — so it is pinned three
independent ways: an identity that must hold for orthogonal cells, an exact
round trip, and the crystallographic ``U_eq`` formula agreeing with the trace.
"""

import numpy as np
import pytest

from crystalline.core.adp import (
    ADPSet,
    ellipsoid_axes,
    ellipsoid_radii,
    equivalent_isotropic,
    probability_scale,
    to_cartesian,
    to_crystallographic,
)

# A hexagonal cell (ZnO's), where the crystallographic basis really differs from
# the cartesian one. Rows are the lattice vectors, as everywhere else.
_HEX = np.array(
    [[2.85292, -1.647134, 0.0], [0.0, 3.294269, 0.0], [0.0, 0.0, 5.270251]]
)
# An arbitrary, definitely-anisotropic tensor with off-diagonal terms.
_U = np.array(
    [[0.0121, -0.0038, -0.0003], [-0.0038, 0.0122, -0.0014], [-0.0003, -0.0014, 0.0031]]
)


def test_an_orthogonal_cell_leaves_the_tensor_alone():
    """The identity that catches a transposed conversion: when the lattice
    vectors are orthogonal and along the cartesian axes, every factor in the
    crystallographic definition cancels."""
    for cell in (np.eye(3), np.diag([4.0, 7.0, 11.0])):
        assert np.allclose(to_crystallographic(_U, cell), _U)


def test_a_hexagonal_cell_really_does_change_it():
    """...and the conversion is not quietly a no-op for a cell that needs it."""
    assert not np.allclose(to_crystallographic(_U, _HEX), _U)


def test_the_conversion_round_trips():
    assert np.allclose(to_cartesian(to_crystallographic(_U, _HEX), _HEX), _U, atol=1e-15)


def test_u_eq_agrees_with_the_crystallographic_formula():
    """``trace/3`` must equal the ``U_eq`` a refinement quotes,
    ``(1/3) sum_ij U_ij a*_i a*_j (a_i . a_j)`` — an independent route through
    the converted tensor, so the two only agree if the conversion is right."""
    u_ij = to_crystallographic(_U, _HEX)
    reciprocal = np.linalg.norm(np.linalg.inv(_HEX), axis=0)
    metric = _HEX @ _HEX.T  # a_i . a_j
    from_cif = sum(
        u_ij[i, j] * reciprocal[i] * reciprocal[j] * metric[i, j]
        for i in range(3)
        for j in range(3)
    ) / 3.0

    assert equivalent_isotropic(_U) == pytest.approx(from_cif)


def test_conversions_accept_a_stack_of_tensors():
    stack = np.stack([_U, _U * 2, np.diag([0.01, 0.02, 0.03])])

    converted = to_crystallographic(stack, _HEX)

    assert converted.shape == stack.shape
    assert np.allclose(converted[0], to_crystallographic(_U, _HEX))
    assert equivalent_isotropic(stack).shape == (3,)


def test_a_degenerate_cell_cannot_define_the_conversion():
    flat = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 0]])
    for convert in (to_crystallographic, to_cartesian):
        with pytest.raises(ValueError):
            convert(_U, flat)


def test_probability_scale_matches_the_crystallographic_convention():
    """50% is ORTEP's, and the number every structure report is drawn at."""
    assert probability_scale(0.50) == pytest.approx(1.5382, abs=1e-4)
    assert probability_scale(0.90) == pytest.approx(2.5003, abs=1e-4)
    assert probability_scale(0.99) == pytest.approx(3.3682, abs=1e-4)
    for bad in (0.0, 1.0, -0.5, 2.0):
        with pytest.raises(ValueError):
            probability_scale(bad)


def test_ellipsoid_axes_reconstruct_the_tensor():
    """The drawn geometry must be the tensor and nothing else: scaling the
    radii back down and rebuilding from the axes has to return U exactly."""
    radii, axes, positive_definite = ellipsoid_axes(_U, 0.5)

    assert positive_definite
    assert np.allclose(axes @ axes.T, np.eye(3))          # orthonormal
    assert np.linalg.det(axes) == pytest.approx(1.0)      # a rotation, not a flip
    eigenvalues = (radii / probability_scale(0.5)) ** 2
    assert np.allclose(axes.T @ np.diag(eigenvalues) @ axes, _U)


def test_the_radii_scale_with_the_probability():
    at_50 = ellipsoid_axes(_U, 0.50)[0]
    at_99 = ellipsoid_axes(_U, 0.99)[0]

    ratio = probability_scale(0.99) / probability_scale(0.50)
    assert np.allclose(at_99, at_50 * ratio)


def test_a_non_positive_definite_tensor_is_flagged_not_drawn_imaginary():
    """Under-converged sampling produces these; a NaN radius would poison the
    mesh, so the offending axis is flattened and the caller told."""
    radii, _axes, positive_definite = ellipsoid_axes(np.diag([0.01, 0.005, -1e-4]))

    assert positive_definite is False
    assert not np.isnan(radii).any()
    assert radii.min() == 0.0


def test_adp_set_indexes_temperatures():
    tensors = np.stack([np.tile(np.eye(3) * t, (2, 1, 1)) for t in (0.001, 0.002, 0.003)])
    adps = ADPSet(temperatures=[10.0, 150.0, 300.0], tensors=tensors)

    assert len(adps) == 3
    assert adps.n_atoms == 2
    assert adps.label(2) == "300 K"
    assert adps.nearest(140.0) == 1
    assert adps.nearest(1e6) == 2       # clamped to the hottest reported
    assert np.allclose(adps.at(1), tensors[1])
    assert np.allclose(adps.at(99), tensors[2])  # out of range clamps, never raises


def test_adp_set_rejects_tensors_that_do_not_match_the_temperatures():
    with pytest.raises(ValueError):
        ADPSet(temperatures=[10.0, 300.0], tensors=np.zeros((3, 2, 3, 3)))
    with pytest.raises(ValueError):
        ADPSet(temperatures=[10.0], tensors=np.zeros((1, 2, 3)))


def test_batched_radii_agree_with_the_per_atom_axes():
    """``ellipsoid_radii`` exists only to be faster than looping ``ellipsoid_axes``
    (the renderer runs it per animation frame), so what it must not do is differ.
    Checked on positive-definite *and* NPD tensors, where the clamp applies."""
    rng = np.random.default_rng(1)
    factors = rng.normal(size=(24, 3, 3)) * 0.05
    tensors = factors @ np.swapaxes(factors, -1, -2)  # positive definite by construction
    tensors[::4] -= np.eye(3) * 0.3  # and some that are not, so the clamp is exercised
    assert not all(ellipsoid_axes(u)[2] for u in tensors)  # NPD cases really present

    for probability in (0.5, 0.9, 0.99):
        looped = np.array([ellipsoid_axes(u, probability)[0] for u in tensors])
        assert np.allclose(ellipsoid_radii(tensors, probability), looped)


def test_batched_radii_handle_an_empty_stack_and_reject_a_single_tensor():
    """A structure can legitimately have no atoms; a bare (3, 3) tensor, though,
    is a caller confusing this with ``ellipsoid_axes``."""
    assert ellipsoid_radii(np.empty((0, 3, 3))).shape == (0, 3)
    with pytest.raises(ValueError):
        ellipsoid_radii(_U)
