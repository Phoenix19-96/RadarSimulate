import numpy as np
import pytest

from radarsim.config import RadarConfig
from radarsim.waveforms.fmcw import FMCWConfig, FMCWWaveform


def test_fmcw_cube_shape_and_absolute_times():
    cfg = FMCWConfig(2e6, 8e-6, 10e-6, 1e6, 4, 2)
    cube = FMCWWaveform(cfg).build_tx(RadarConfig())
    assert cube.data.shape == (2, 4, 1, 8)
    assert cube.dimensions == ("frame", "slow_time", "tx", "fast_time")
    assert cube.sample_times_s[1, 0, 0] == pytest.approx(40e-6)


def test_fmcw_phase_matches_linear_chirp():
    cfg = FMCWConfig(2e6, 8e-6, 10e-6, 1e6, 2, 1)
    times = np.array([1e-6, 2e-6])
    expected = np.exp(1j * np.pi * cfg.slope_hz_per_s * times**2)
    assert FMCWWaveform(cfg).sample_at(times) == pytest.approx(expected)


def test_fmcw_is_zero_during_idle_time():
    cfg = FMCWConfig(2e6, 8e-6, 10e-6, 1e6, 2, 1)
    assert FMCWWaveform(cfg).sample_at(np.array([9e-6]))[0] == 0j


def test_fmcw_build_uses_local_fast_time_at_repetition_and_frame_boundaries():
    waveform = FMCWWaveform(FMCWConfig(2e6, 8e-6, 10e-6, 1e6, 4, 2))
    cube = waveform.build_tx(RadarConfig())
    assert cube.data[0, 1, 0, 0] == pytest.approx(1 + 0j)
    assert cube.data[1, 0, 0, 0] == pytest.approx(1 + 0j)
    assert np.count_nonzero(cube.data[1, :, 0, :]) == 32


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "2e6"])
def test_fmcw_rejects_nonfinite_or_non_numeric_physical_values(value):
    with pytest.raises(ValueError, match="bandwidth_hz"):
        FMCWConfig(value, 8e-6, 10e-6, 1e6, 2)


@pytest.mark.parametrize("field", ["chirps_per_frame", "frame_count"])
@pytest.mark.parametrize("value", [0, -1, 1.0, True])
def test_fmcw_rejects_invalid_counts(field, value):
    kwargs = {field: value}
    kwargs.setdefault("chirps_per_frame", 2)
    with pytest.raises(ValueError, match=field):
        FMCWConfig(2e6, 8e-6, 10e-6, 1e6, **kwargs)


def test_fmcw_rejects_subsample_chirp_and_is_bounded_in_time():
    with pytest.raises(ValueError, match="sample count"):
        FMCWConfig(2e6, 1e-15, 10e-6, 1e6, 1)
    waveform = FMCWWaveform(FMCWConfig(2e6, 8e-6, 10e-6, 1e6, 2, 1))
    times = np.array([-1e-12, 0.0, 10e-6, 19.999999e-6, 20e-6, 20.001e-6])
    values = waveform.sample_at(times)
    assert values[0] == 0j
    assert values[1] == pytest.approx(1 + 0j)
    assert values[2] == pytest.approx(1 + 0j)
    assert values[3] == 0j
    assert values[4] == 0j
    assert values[5] == 0j


def test_fmcw_one_femtosecond_probes_preserve_physical_boundaries():
    waveform = FMCWWaveform(FMCWConfig(2e6, 8e-6, 10e-6, 1e6, 2, 1))
    eps = 1e-15
    values = waveform.sample_at(np.array([
        8e-6 - eps, 8e-6 + eps, 10e-6 - eps, 10e-6 + eps,
        20e-6 - eps, 20e-6 + eps,
    ]))
    assert values[0] != 0j
    assert values[1] == 0j
    assert values[2] == 0j
    assert values[3] != 0j
    assert values[4] == 0j
    assert values[5] == 0j


def test_fmcw_long_support_does_not_snap_one_femtosecond_boundaries():
    waveform = FMCWWaveform(FMCWConfig(2e6, 8e-6, 10e-6, 1e6, 100000, 1))
    eps = 1e-15
    values = waveform.sample_at(np.array([1.0 - eps, 1.0 - 2e-6 - eps, 1.0 - 2e-6 + eps]))
    assert values[0] == 0j
    assert values[1] != 0j
    assert values[2] == 0j


@pytest.mark.parametrize("clock", [False, True])
def test_fmcw_all_100000_nominal_starts_reset_phase(clock):
    waveform = FMCWWaveform(FMCWConfig(2e6, 8e-6, 10e-6, 1e6, 100000))
    times = np.arange(100000) * 10 / 1e6 if clock else np.arange(100000) * 10e-6
    np.testing.assert_array_equal(waveform.sample_at(times), 1 + 0j)
    np.testing.assert_array_equal(waveform.sample_at(times - 1e-15), 0j)
    np.testing.assert_array_equal(waveform.sample_at(times + 1e-15) != 0j, True)
    ends = (np.arange(100000) * 10 + 8) / 1e6 if clock else times + 8e-6
    np.testing.assert_array_equal(waveform.sample_at(ends), 0j)
    np.testing.assert_array_equal(waveform.sample_at(ends - 1e-15) != 0j, True)
    np.testing.assert_array_equal(waveform.sample_at(ends + 1e-15), 0j)


def test_fmcw_200_repetition_absolute_samples_match_tx_amplitude_and_phase():
    waveform = FMCWWaveform(FMCWConfig(2e6, 8e-6, 10e-6, 1e6, 100, 2))
    cube = waveform.build_tx(RadarConfig())
    np.testing.assert_array_equal(waveform.sample_at(cube.sample_times_s[:, :, 0]), 1 + 0j)
    expected = np.exp(1j * np.pi * 2.5e11 * (np.arange(8) / 1e6) ** 2)
    np.testing.assert_allclose(cube.data[:, :, 0], np.broadcast_to(expected, (2, 100, 8)), rtol=0, atol=1e-12)
    sampled = waveform.sample_at(cube.sample_times_s)
    np.testing.assert_array_equal(np.count_nonzero(sampled, axis=-1), 8)
    np.testing.assert_allclose(sampled, cube.data[:, :, 0], rtol=0, atol=1e-10)
    assert waveform.sample_at(np.array([50e-6]))[0] == 1 + 0j


@pytest.mark.parametrize("count", [2, 100000])
def test_fmcw_femtosecond_offsets_at_all_envelope_edges(count):
    waveform = FMCWWaveform(FMCWConfig(2e6, 8e-6, 10e-6, 1e6, count))
    start = (count - 1) * 10e-6
    edges = np.array([0.0, start, start + 8e-6, count * 10e-6])
    offsets = np.array([-1e-15, 0.0, 1e-15])
    times = edges[:, None] + offsets
    assert np.all(times[:, 0] < edges) and np.all(times[:, 2] > edges)
    np.testing.assert_array_equal(waveform.sample_at(times) != 0j, [
        [False, True, True], [False, True, True],
        [True, False, False], [False, False, False],
    ])


@pytest.mark.parametrize("count", [2, 100000])
def test_fmcw_without_idle_preserves_last_femtosecond_of_support(count):
    waveform = FMCWWaveform(FMCWConfig(2e6, 10e-6, 10e-6, 1e6, count))
    support = count * 10e-6
    np.testing.assert_array_equal(
        waveform.sample_at(np.array([support - 1e-15, support, support + 1e-15])) != 0j,
        [True, False, False],
    )
