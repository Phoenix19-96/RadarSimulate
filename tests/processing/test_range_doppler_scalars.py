import numpy as np
import pytest

from radarsim.models import RangeDopplerResult


def result_with(**overrides):
    values = {
        "spectrum": np.ones((1, 1, 1, 1), complex),
        "power_w": np.ones((1, 1, 1, 1)),
        "range_m": np.array([0.0]),
        "velocity_mps": np.array([0.0]),
        "range_resolution_m": 1.0,
        "velocity_resolution_mps": 1.0,
        "max_unambiguous_range_m": 1.0,
        "max_unambiguous_velocity_mps": 1.0,
    }
    values.update(overrides)
    return RangeDopplerResult(**values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("range_resolution_m", np.nan),
        ("range_resolution_m", True),
        ("velocity_resolution_mps", np.nan),
        ("velocity_resolution_mps", False),
        ("velocity_resolution_mps", -np.inf),
        ("velocity_resolution_mps", "1"),
        ("max_unambiguous_range_m", np.nan),
        ("max_unambiguous_range_m", True),
        ("max_unambiguous_velocity_mps", np.inf),
        ("max_unambiguous_velocity_mps", "1"),
    ],
)
def test_range_doppler_result_rejects_invalid_scalar_contracts(field, value):
    """Break caught: scalar contracts accepted NaN, bool, or invalid infinity."""
    with pytest.raises(ValueError, match=field):
        result_with(**{field: value})


def test_range_doppler_result_allows_positive_infinite_velocity_resolution_only():
    result = result_with(velocity_resolution_mps=np.inf)

    assert np.isposinf(result.velocity_resolution_mps)
