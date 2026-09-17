import numpy as np
import pytest

from radarsim.config import ProcessingConfig, RadarConfig
from radarsim.models import SignalCube
from radarsim.processing.range_doppler import process_fmcw
from radarsim.waveforms.fmcw import FMCWConfig


def synthetic_fmcw_beat(
    radar: RadarConfig,
    waveform: FMCWConfig,
    range_m: float,
    velocity_mps: float,
    *,
    frames: int | None = None,
    receivers: int = 1,
) -> SignalCube:
    """A dechirped point target using the project's receding-positive convention."""
    frame_count = waveform.frame_count if frames is None else frames
    fast_count = round(waveform.chirp_duration_s * waveform.sample_rate_hz)
    fast = np.arange(fast_count) / waveform.sample_rate_hz
    slow = np.arange(waveform.chirps_per_frame) * waveform.chirp_repetition_interval_s
    beat_hz = 2 * waveform.slope_hz_per_s * range_m / 299_792_458
    doppler_hz = -2 * velocity_mps / radar.wavelength_m
    data = np.exp(-1j * 2 * np.pi * beat_hz * fast)[None, None, None, :]
    data = data * np.exp(1j * 2 * np.pi * doppler_hz * slow)[None, :, None, None]
    data = np.broadcast_to(
        data, (frame_count, waveform.chirps_per_frame, receivers, fast_count)
    ).copy()
    frame_span = waveform.chirps_per_frame * waveform.chirp_repetition_interval_s
    times = (
        np.arange(frame_count)[:, None, None] * frame_span
        + slow[None, :, None]
        + fast[None, None, :]
    )
    return SignalCube(
        data,
        ("frame", "slow_time", "rx", "fast_time"),
        fast,
        slow,
        times,
        {"waveform_kind": "fmcw", "frontend": "fmcw_dechirp"},
    )


def test_fmcw_axes_and_peak_use_receding_positive_velocity():
    """Break caught: reversing Doppler sign or using the wrong FMCW beat bins."""
    radar = RadarConfig(carrier_hz=10e9)
    waveform = FMCWConfig(20e6, 40e-6, 50e-6, 10e6, 64, 1)
    cube = synthetic_fmcw_beat(radar, waveform, 30, 5)

    result = process_fmcw(
        cube,
        radar,
        waveform,
        ProcessingConfig(
            range_window="boxcar",
            doppler_window="boxcar",
            range_fft_size=512,
            doppler_fft_size=64,
        ),
    )

    doppler_bin, range_bin = np.unravel_index(
        np.argmax(result.power_w[0, 0]), result.power_w[0, 0].shape
    )
    assert result.spectrum.shape == (1, 1, 64, result.range_m.size)
    assert result.range_m[range_bin] == pytest.approx(
        30, abs=result.range_resolution_m
    )
    assert result.velocity_mps[doppler_bin] == pytest.approx(
        5, abs=result.velocity_resolution_mps
    )


def test_fmcw_rejects_nonfinite_dechirped_samples():
    """Break caught: nonfinite FFT inputs silently contaminate range-Doppler output."""
    radar = RadarConfig()
    waveform = FMCWConfig(20e6, 40e-6, 50e-6, 10e6, 8, 1)
    cube = synthetic_fmcw_beat(radar, waveform, 30, 0)
    cube.data[0, 0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        process_fmcw(cube, radar, waveform, ProcessingConfig())


def test_fmcw_uses_fft_bin_spacing_without_misstating_resolution():
    """Break caught: confusing zero-padded bin spacing with physical resolution."""
    radar = RadarConfig()
    waveform = FMCWConfig(20e6, 40e-6, 50e-6, 10e6, 16, 1)
    result = process_fmcw(
        synthetic_fmcw_beat(radar, waveform, 20, 0),
        radar,
        waveform,
        ProcessingConfig(range_window="boxcar", doppler_window="boxcar", range_fft_size=1024),
    )

    assert result.range_m[1] - result.range_m[0] == pytest.approx(
        299_792_458 / (2 * waveform.slope_hz_per_s * 1024 / waveform.sample_rate_hz)
    )
    assert result.range_resolution_m == pytest.approx(299_792_458 / (2 * waveform.bandwidth_hz))
    assert result.range_resolution_m > result.range_m[1] - result.range_m[0]


def test_fmcw_preserves_frames_and_receivers_and_orders_shifted_velocity_axis():
    """Break caught: dropping a frame/Rx axis or leaving fftshift velocity descending."""
    radar = RadarConfig()
    waveform = FMCWConfig(20e6, 40e-6, 50e-6, 10e6, 8, 2)
    result = process_fmcw(
        synthetic_fmcw_beat(radar, waveform, 30, -3, receivers=2),
        radar,
        waveform,
        ProcessingConfig(range_window="boxcar", doppler_window="boxcar", doppler_fft_size=16),
    )

    assert result.spectrum.shape[:2] == (2, 2)
    assert np.all(np.diff(result.velocity_mps) > 0)
    assert np.allclose(result.power_w, np.abs(result.spectrum) ** 2)


def test_fmcw_range_window_changes_the_fast_time_transform():
    """Break caught: ignoring the configured FMCW range window."""
    radar = RadarConfig()
    waveform = FMCWConfig(20e6, 40e-6, 50e-6, 10e6, 8, 1)
    cube = synthetic_fmcw_beat(radar, waveform, 0, 0)
    boxcar = process_fmcw(cube, radar, waveform, ProcessingConfig(range_window="boxcar", doppler_window="boxcar"))
    hann = process_fmcw(cube, radar, waveform, ProcessingConfig(range_window="hann", doppler_window="boxcar"))

    zero_velocity = np.argmin(np.abs(boxcar.velocity_mps))
    assert boxcar.power_w[0, 0, zero_velocity, 0] == pytest.approx(10_240_000)
    assert hann.power_w[0, 0, zero_velocity, 0] == pytest.approx(2_560_000)


def test_fmcw_rejects_fft_sizes_smaller_than_input_counts():
    """Break caught: accepting truncating range or Doppler FFTs."""
    radar = RadarConfig()
    waveform = FMCWConfig(20e6, 40e-6, 50e-6, 10e6, 8, 1)
    cube = synthetic_fmcw_beat(radar, waveform, 30, 0)

    with pytest.raises(ValueError, match="range_fft_size"):
        process_fmcw(cube, radar, waveform, ProcessingConfig(range_fft_size=399))
    with pytest.raises(ValueError, match="doppler_fft_size"):
        process_fmcw(cube, radar, waveform, ProcessingConfig(doppler_fft_size=7))


def test_lfm_match_filter_aligns_full_convolution_delay_and_velocity_axis():
    """Break caught: omitting full-convolution delay offset or reversing velocity."""
    from radarsim.models import LFMFrontendResult
    from radarsim.processing.range_doppler import process_lfm
    from radarsim.waveforms.lfm import LFMConfig, LFMWaveform

    radar = RadarConfig(carrier_hz=10e9)
    config = LFMConfig(2e6, 8e-6, 80e-6, 4e6, 32, 1)
    waveform = LFMWaveform(config)
    reference = waveform.reference()
    delay_samples = 40
    velocity_mps = 3.0
    slow = np.arange(config.pulses_per_frame) * config.pulse_repetition_interval_s
    fast = np.arange(round(config.pulse_repetition_interval_s * config.sample_rate_hz)) / config.sample_rate_hz
    raw = np.zeros((1, config.pulses_per_frame, 1, fast.size), complex)
    raw[0, :, 0, delay_samples:delay_samples + reference.size] = reference[None, :] * np.exp(
        1j * 2 * np.pi * (-2 * velocity_mps / radar.wavelength_m) * slow
    )[:, None]
    cube = SignalCube(raw, ("frame", "slow_time", "rx", "fast_time"), fast, slow,
                      slow[None, :, None] + fast[None, None, :], {"waveform_kind": "lfm"})

    result = process_lfm(LFMFrontendResult(cube, np.conj(reference[::-1])), radar, config,
                         ProcessingConfig(doppler_window="boxcar", doppler_fft_size=32))
    doppler_bin, range_bin = np.unravel_index(np.argmax(result.power_w[0, 0]), result.power_w[0, 0].shape)
    assert range_bin == delay_samples
    assert result.range_m[range_bin] == pytest.approx(
        delay_samples / config.sample_rate_hz * 299_792_458 / 2,

    )
    assert result.velocity_mps[doppler_bin] == pytest.approx(
        velocity_mps, abs=result.velocity_resolution_mps
    )


def test_lfm_preserves_multiple_frames_and_receivers():
    """Break caught: collapsing LFM frame or receiver axes during compression."""
    from radarsim.models import LFMFrontendResult
    from radarsim.processing.range_doppler import process_lfm
    from radarsim.waveforms.lfm import LFMConfig, LFMWaveform

    radar = RadarConfig()
    config = LFMConfig(2e6, 2e-6, 20e-6, 2e6, 4, 2)
    waveform = LFMWaveform(config)
    fast = np.arange(round(config.pulse_repetition_interval_s * config.sample_rate_hz)) / config.sample_rate_hz
    slow = np.arange(config.pulses_per_frame) * config.pulse_repetition_interval_s
    raw = np.zeros((2, 4, 2, fast.size), complex)
    raw[..., :waveform.reference().size] = waveform.reference()
    cube = SignalCube(raw, ("frame", "slow_time", "rx", "fast_time"), fast, slow,
                      np.arange(2)[:, None, None] * (4 * config.pulse_repetition_interval_s)
                      + slow[None, :, None] + fast[None, None, :], {"waveform_kind": "lfm"})

    result = process_lfm(LFMFrontendResult(cube, np.conj(waveform.reference()[::-1])), radar, config,
                         ProcessingConfig(doppler_window="boxcar"))
    assert result.spectrum.shape[:2] == (2, 2)


def test_processors_reject_wrong_frontend_metadata():
    """Break caught: processing raw/wrong-waveform cubes as FMCW dechirped data."""
    radar = RadarConfig()
    waveform = FMCWConfig(20e6, 40e-6, 50e-6, 10e6, 8, 1)
    cube = synthetic_fmcw_beat(radar, waveform, 30, 0)
    cube.metadata["frontend"] = "raw"
    with pytest.raises(ValueError, match="dechirped"):
        process_fmcw(cube, radar, waveform, ProcessingConfig())
