import numpy as np
import pytest

from radarsim.geometry import cartesian_to_polar, polar_to_cartesian, radial_velocity


@pytest.mark.parametrize(("xyz", "expected"), [
    ((1, 0, 0), (1, 0, 0)),
    ((0, 1, 0), (1, -90, 0)),
    ((0, -1, 0), (1, 90, 0)),
    ((0, 0, 1), (1, 0, 90)),
])
def test_axis_conventions(xyz, expected):
    assert cartesian_to_polar(np.asarray(xyz)) == pytest.approx(expected)


def test_polar_cartesian_round_trip():
    xyz = polar_to_cartesian(1250, -32, 14)
    assert cartesian_to_polar(xyz) == pytest.approx((1250, -32, 14))


def test_zero_range_has_no_defined_angles():
    with pytest.raises(ValueError, match="zero range"):
        cartesian_to_polar(np.zeros(3))


def test_radial_velocity_is_positive_when_receding():
    assert radial_velocity(np.array([100, 0, 0]), np.array([12, 3, 0])) == 12

@pytest.mark.parametrize(("xyz", "expected"), [((-1, 1, 0), (2**0.5, -135, 0)), ((-1, -1, 0), (2**0.5, 135, 0))])
def test_inverse_quadrant_conventions(xyz, expected):
    assert cartesian_to_polar(np.asarray(xyz)) == pytest.approx(expected)

def test_large_finite_vectors_use_stable_norm_and_projection():
    xyz = np.array([1e308, 1e308, 0.0])
    assert cartesian_to_polar(xyz) == pytest.approx((np.sqrt(2) * 1e308, -45, 0))
    assert radial_velocity(xyz, np.array([1e308, 0.0, 0.0])) == pytest.approx(7.071067811865475e307)

def test_unrepresentable_range_is_rejected():
    with pytest.raises(ValueError, match="range"):
        cartesian_to_polar(np.array([1.7e308, 1.7e308, 0.0]))

def test_unrepresentable_range_is_rejected_for_radial_velocity():
    with pytest.raises(ValueError, match="range"):
        radial_velocity(np.array([1.7e308, 1.7e308, 0.0]), np.array([10.0, 0.0, 0.0]))

def test_numeric_string_geometry_inputs_are_rejected():
    with pytest.raises(ValueError, match="xyz_m"):
        cartesian_to_polar(["1", 0, 0])
    with pytest.raises(ValueError, match="range_m"):
        polar_to_cartesian("1", 0, 0)
