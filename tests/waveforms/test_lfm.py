import numpy as np
import pytest

from radarsim.config import RadarConfig
from radarsim.waveforms.lfm import LFMConfig, LFMWaveform


def test_lfm_cube_covers_pri_and_retains_tx_axis():
    cfg = LFMConfig(2e6, 4e-6, 10e-6, 1e6, 3, 1)
    cube = LFMWaveform(cfg).build_tx(RadarConfig())
    assert cube.data.shape == (1, 3, 1, 10)
    assert np.count_nonzero(cube.data[0, 0, 0]) == 4


def test_lfm_reference_contains_only_transmit_pulse():
    waveform = LFMWaveform(LFMConfig(2e6, 4e-6, 10e-6, 1e6, 3, 1))
    assert waveform.reference().shape == (4,)
    assert np.abs(waveform.reference()) == pytest.approx(np.ones(4))


def test_lfm_is_zero_outside_pulse():
    waveform = LFMWaveform(LFMConfig(2e6, 4e-6, 10e-6, 1e6, 3, 1))
    assert waveform.sample_at(np.array([5e-6]))[0] == 0j


def test_lfm_build_has_four_active_samples_on_every_pulse_and_frame():
    waveform = LFMWaveform(LFMConfig(2e6, 4e-6, 10e-6, 1e6, 3, 2))
    cube = waveform.build_tx(RadarConfig())
    assert [np.count_nonzero(cube.data[f, s, 0]) for f in range(2) for s in range(3)] == [4] * 6
    assert cube.data[1, 0, 0, 0] == pytest.approx(1 + 0j)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "2e6"])
def test_lfm_rejects_nonfinite_or_non_numeric_physical_values(value):
    with pytest.raises(ValueError, match="bandwidth_hz"):
        LFMConfig(value, 4e-6, 10e-6, 1e6, 2)


@pytest.mark.parametrize("field", ["pulses_per_frame", "frame_count"])
@pytest.mark.parametrize("value", [0, -1, 1.0, True])
def test_lfm_rejects_invalid_counts(field, value):
    kwargs = {field: value}
    kwargs.setdefault("pulses_per_frame", 2)
    with pytest.raises(ValueError, match=field):
        LFMConfig(2e6, 4e-6, 10e-6, 1e6, **kwargs)


def test_lfm_rejects_subsample_pulse_and_is_bounded_in_time():
    with pytest.raises(ValueError, match="sample count"):
        LFMConfig(2e6, 1e-15, 10e-6, 1e6, 1)
    waveform = LFMWaveform(LFMConfig(2e6, 4e-6, 10e-6, 1e6, 2, 1))
    times = np.array([-1e-12, 0.0, 10e-6, 13.999999e-6, 14e-6, 20e-6])
    values = waveform.sample_at(times)
    assert values[0] == 0j
    assert values[1] == pytest.approx(1 + 0j)
    assert values[2] == pytest.approx(1 + 0j)
    assert values[3] != 0j
    assert values[4] == 0j
    assert values[5] == 0j


def test_lfm_one_femtosecond_probes_preserve_physical_boundaries():
    waveform = LFMWaveform(LFMConfig(2e6, 4e-6, 10e-6, 1e6, 2, 1))
    eps = 1e-15
    values = waveform.sample_at(np.array([
        4e-6 - eps, 4e-6 + eps, 10e-6 - eps, 10e-6 + eps,
        20e-6 - eps, 20e-6 + eps,
    ]))
    assert values[0] != 0j
    assert values[1] == 0j
    assert values[2] == 0j
    assert values[3] != 0j
    assert values[4] == 0j
    assert values[5] == 0j


def test_lfm_long_support_does_not_snap_one_femtosecond_boundaries():
    waveform = LFMWaveform(LFMConfig(2e6, 4e-6, 10e-6, 1e6, 100000, 1))
    eps = 1e-15
    values = waveform.sample_at(np.array([1.0 - eps, 1.0 - 6e-6 - eps, 1.0 - 6e-6 + eps]))
    assert values[0] == 0j
    assert values[1] != 0j
    assert values[2] == 0j


@pytest.mark.parametrize("clock", [False, True])
def test_lfm_all_100000_nominal_starts_reset_phase(clock):
    waveform = LFMWaveform(LFMConfig(2e6, 4e-6, 10e-6, 1e6, 100000))
    times = np.arange(100000) * 10 / 1e6 if clock else np.arange(100000) * 10e-6
    np.testing.assert_array_equal(waveform.sample_at(times), 1 + 0j)
    np.testing.assert_array_equal(waveform.sample_at(times - 1e-15), 0j)
    np.testing.assert_array_equal(waveform.sample_at(times + 1e-15) != 0j, True)
    ends = (np.arange(100000) * 10 + 4) / 1e6 if clock else times + 4e-6
    np.testing.assert_array_equal(waveform.sample_at(ends), 0j)
    np.testing.assert_array_equal(waveform.sample_at(ends - 1e-15) != 0j, True)
    np.testing.assert_array_equal(waveform.sample_at(ends + 1e-15), 0j)


def test_lfm_200_repetition_absolute_samples_match_tx_amplitude_and_phase():
    waveform = LFMWaveform(LFMConfig(2e6, 4e-6, 10e-6, 1e6, 100, 2))
    cube = waveform.build_tx(RadarConfig())
    np.testing.assert_array_equal(waveform.sample_at(cube.sample_times_s[:, :, 0]), 1 + 0j)
    expected = np.zeros(10, dtype=complex)
    expected[:4] = np.exp(1j * np.pi * 5e11 * (np.arange(4) / 1e6) ** 2)
    np.testing.assert_allclose(cube.data[:, :, 0], np.broadcast_to(expected, (2, 100, 10)), rtol=0, atol=1e-12)
    sampled = waveform.sample_at(cube.sample_times_s)
    np.testing.assert_array_equal(np.count_nonzero(sampled, axis=-1), 4)
    np.testing.assert_allclose(sampled, cube.data[:, :, 0], rtol=0, atol=1e-10)
    assert waveform.sample_at(np.array([50e-6]))[0] == 1 + 0j


@pytest.mark.parametrize("count", [2, 100000])
def test_lfm_femtosecond_offsets_at_all_envelope_edges(count):
    waveform = LFMWaveform(LFMConfig(2e6, 4e-6, 10e-6, 1e6, count))
    start = (count - 1) * 10e-6
    edges = np.array([0.0, start, start + 4e-6, count * 10e-6])
    offsets = np.array([-1e-15, 0.0, 1e-15])
    times = edges[:, None] + offsets
    assert np.all(times[:, 0] < edges) and np.all(times[:, 2] > edges)
    np.testing.assert_array_equal(waveform.sample_at(times) != 0j, [
        [False, True, True], [False, True, True],
        [True, False, False], [False, False, False],
    ])
