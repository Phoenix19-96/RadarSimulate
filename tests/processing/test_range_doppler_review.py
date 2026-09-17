import numpy as np
import pytest

from radarsim.config import ProcessingConfig, RadarConfig
from radarsim.constants import C_MPS
from radarsim.models import LFMFrontendResult, RangeDopplerResult, SignalCube
from radarsim.processing.range_doppler import process_fmcw, process_lfm
from radarsim.waveforms.fmcw import FMCWConfig
from radarsim.waveforms.lfm import LFMConfig, LFMWaveform


def fmcw_cube(radar, waveform, *, metadata=None):
    fast_count = round(waveform.chirp_duration_s * waveform.sample_rate_hz)
    fast = np.arange(fast_count) / waveform.sample_rate_hz
    slow = np.arange(waveform.chirps_per_frame) * waveform.chirp_repetition_interval_s
    data = np.ones((waveform.frame_count, waveform.chirps_per_frame, 1, fast_count), complex)
    times = np.arange(waveform.frame_count)[:, None, None] * (waveform.chirps_per_frame * waveform.chirp_repetition_interval_s) + slow[None, :, None] + fast[None, None, :]
    return SignalCube(data, ("frame", "slow_time", "rx", "fast_time"), fast, slow, times,
                      {"waveform_kind": "fmcw", "frontend": "fmcw_dechirp", **(metadata or {})})


def lfm_frontend(config, *, metadata=None, pulse_phases=None):
    waveform = LFMWaveform(config)
    reference = waveform.reference()
    fast = np.arange(round(config.pulse_repetition_interval_s * config.sample_rate_hz)) / config.sample_rate_hz
    slow = np.arange(config.pulses_per_frame) * config.pulse_repetition_interval_s
    phases = np.ones(config.pulses_per_frame, complex) if pulse_phases is None else pulse_phases
    raw = np.zeros((config.frame_count, config.pulses_per_frame, 1, fast.size), complex)
    raw[:, :, 0, :reference.size] = phases[None, :, None] * reference[None, None, :]
    times = np.arange(config.frame_count)[:, None, None] * (config.pulses_per_frame * config.pulse_repetition_interval_s) + slow[None, :, None] + fast[None, None, :]
    cube = SignalCube(raw, ("frame", "slow_time", "rx", "fast_time"), fast, slow, times,
                      {"waveform_kind": "lfm", **(metadata or {})})
    return LFMFrontendResult(cube, np.conj(reference[::-1]))


def test_fmcw_accepts_one_chirp_and_one_fast_sample_with_unavailable_velocity_resolution():
    radar = RadarConfig()
    waveform = FMCWConfig(1e6, 1e-6, 2e-6, 1e6, 1, 1)
    result = process_fmcw(fmcw_cube(radar, waveform), radar, waveform, ProcessingConfig(range_window="boxcar", doppler_window="boxcar"))

    assert result.spectrum.shape == (1, 1, 1, 1)
    np.testing.assert_array_equal(result.range_m, [0.0])
    np.testing.assert_array_equal(result.velocity_mps, [0.0])
    assert np.isinf(result.velocity_resolution_mps)


def test_lfm_accepts_one_pulse_with_zero_velocity_and_unavailable_resolution():
    radar = RadarConfig()
    config = LFMConfig(1e6, 1e-6, 2e-6, 1e6, 1, 1)
    result = process_lfm(lfm_frontend(config), radar, config, ProcessingConfig(doppler_window="boxcar"))

    assert result.spectrum.shape[2] == 1
    np.testing.assert_array_equal(result.velocity_mps, [0.0])
    assert np.isinf(result.velocity_resolution_mps)


def test_max_unambiguous_range_is_theoretical_for_odd_fmcw_fft_and_lfm_pri():
    radar = RadarConfig()
    fmcw = FMCWConfig(2e6, 2e-6, 3e-6, 2e6, 2, 1)
    odd = process_fmcw(fmcw_cube(radar, fmcw), radar, fmcw,
                       ProcessingConfig(range_window="boxcar", doppler_window="boxcar", range_fft_size=5))
    even = process_fmcw(fmcw_cube(radar, fmcw), radar, fmcw,
                        ProcessingConfig(range_window="boxcar", doppler_window="boxcar", range_fft_size=4))
    lfm = LFMConfig(1e6, 1e-6, 4e-6, 1e6, 2, 1)
    pulsed = process_lfm(lfm_frontend(lfm), radar, lfm, ProcessingConfig(doppler_window="boxcar"))

    assert odd.max_unambiguous_range_m == pytest.approx(C_MPS * fmcw.sample_rate_hz / (4 * fmcw.slope_hz_per_s))
    assert even.max_unambiguous_range_m == pytest.approx(C_MPS * fmcw.sample_rate_hz / (4 * fmcw.slope_hz_per_s))
    assert pulsed.max_unambiguous_range_m == pytest.approx(C_MPS * lfm.pulse_repetition_interval_s / 2)


def test_range_doppler_result_validates_and_defensively_freezes_arrays():
    spectrum = np.ones((1, 1, 2, 2), complex)
    power = np.ones((1, 1, 2, 2))
    ranges = np.array([0.0, 1.0])
    velocity = np.array([-1.0, 1.0])
    result = RangeDopplerResult(spectrum, power, ranges, velocity, 1.0, 1.0, 2.0, 1.0)
    spectrum[:] = 7
    power[:] = 8

    assert result != RangeDopplerResult(np.ones((1, 1, 2, 2), complex), np.ones((1, 1, 2, 2)), ranges, velocity, 1.0, 1.0, 2.0, 1.0)
    assert result.spectrum[0, 0, 0, 0] == 1
    assert not result.spectrum.flags.writeable
    assert not result.power_w.flags.writeable
    with pytest.raises(ValueError, match="range_m"):
        RangeDopplerResult(np.ones((1, 1, 2, 2), complex), np.ones((1, 1, 2, 2)), np.array([0.0]), velocity, 1.0, 1.0, 2.0, 1.0)
    with pytest.raises(ValueError, match="finite"):
        RangeDopplerResult(np.ones((1, 1, 2, 2), complex), np.full((1, 1, 2, 2), np.nan), ranges, velocity, 1.0, 1.0, 2.0, 1.0)


def test_processors_reject_same_timing_waveforms_with_wrong_physical_parameters():
    radar = RadarConfig()
    fmcw = FMCWConfig(2e6, 2e-6, 3e-6, 2e6, 2, 1)
    beat = fmcw_cube(radar, fmcw, metadata={"slope_hz_per_s": fmcw.slope_hz_per_s * 2, "bandwidth_hz": fmcw.bandwidth_hz * 2, "chirp_duration_s": fmcw.chirp_duration_s})
    with pytest.raises(ValueError, match="slope"):
        process_fmcw(beat, radar, fmcw, ProcessingConfig())

    actual = LFMConfig(1e6, 2e-6, 4e-6, 1e6, 2, 1)
    wrong = LFMConfig(2e6, 2e-6, 4e-6, 1e6, 2, 1)
    with pytest.raises(ValueError, match="reference"):
        process_lfm(lfm_frontend(actual), radar, wrong, ProcessingConfig())


def test_lfm_doppler_window_changes_the_slow_time_transform():
    radar = RadarConfig()
    config = LFMConfig(1e6, 1e-6, 4e-6, 1e6, 4, 1)
    phases = np.exp(1j * np.array([0.0, 0.4, 1.1, 2.0]))
    frontend = lfm_frontend(config, pulse_phases=phases)
    boxcar = process_lfm(frontend, radar, config, ProcessingConfig(doppler_window="boxcar"))
    hann = process_lfm(frontend, radar, config, ProcessingConfig(doppler_window="hann"))

    assert not np.allclose(boxcar.spectrum, hann.spectrum)
