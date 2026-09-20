import numpy as np
import pytest

from radarsim.models import SignalCube


def test_signal_cube_accepts_named_four_dimensional_data():
    cube = SignalCube(
        np.zeros((2, 4, 1, 8), complex),
        ("frame", "slow_time", "rx", "fast_time"),
        np.arange(8) / 1e6,
        np.arange(4) / 1e3,
        np.zeros((2, 4, 8)),
        {"waveform_kind": "fmcw"},
    )
    assert cube.data.shape == (2, 4, 1, 8)


def test_signal_cube_accepts_transmit_dimension():
    cube = SignalCube(
        np.zeros((2, 4, 1, 8), complex),
        ("frame", "slow_time", "tx", "fast_time"),
        np.arange(8), np.arange(4), np.zeros((2, 4, 8)), {},
    )
    assert cube.dimensions[2] == "tx"


def test_signal_cube_rejects_non_four_dimensional_data():
    with pytest.raises(ValueError, match="four-dimensional"):
        SignalCube(
            np.zeros((4, 8), complex),
            ("slow_time", "fast_time"),
            np.arange(8),
            np.arange(4),
            np.zeros((1, 4, 8)),
            {},
        )


def _cube(**overrides):
    values = {
        "data": np.zeros((2, 4, 1, 8), complex),
        "dimensions": ("frame", "slow_time", "rx", "fast_time"),
        "fast_time_s": np.arange(8),
        "slow_time_s": np.arange(4),
        "sample_times_s": np.zeros((2, 4, 8)),
        "metadata": {},
    }
    values.update(overrides)
    return SignalCube(**values)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("dimensions", ("frame", "rx", "slow_time", "fast_time"), "dimensions"),
        ("fast_time_s", np.arange(7), "fast_time_s"),
        ("slow_time_s", np.arange(3), "slow_time_s"),
        ("sample_times_s", np.zeros((2, 4, 7)), "sample_times_s"),
    ],
)
def test_signal_cube_rejects_invalid_metadata_shapes_and_order(field, value, message):
    with pytest.raises(ValueError, match=message):
        _cube(**{field: value})


def test_signal_cube_uses_identity_equality_and_safe_hashing():
    first = _cube()
    second = _cube()
    assert first is not second
    assert first != second
    assert isinstance(hash(first), int)
