import numpy as np
from scipy.signal import fftconvolve, get_window

from radarsim.constants import C_MPS
from radarsim.models import RangeDopplerResult
from radarsim.waveforms.lfm import LFMWaveform


def _validate_received_cube(cube, frame_count, slow_count, fast_count,
                            sample_rate_hz, repetition_s, name):
    """Check that processing receives the canonical, finite frontend product."""
    if cube.dimensions != ("frame", "slow_time", "rx", "fast_time"):
        raise ValueError(f"{name} requires an Rx SignalCube")
    if cube.data.shape[:2] != (frame_count, slow_count) or cube.data.shape[-1] != fast_count:
        raise ValueError(f"{name} data shape does not match waveform")
    if cube.data.shape[2] < 1 or slow_count < 1 or fast_count < 1:
        raise ValueError(f"{name} requires at least one receiver, slow-time sample, and fast-time sample")
    arrays = (cube.data.real, cube.data.imag, cube.fast_time_s,
              cube.slow_time_s, cube.sample_times_s)
    if not all(np.all(np.isfinite(values)) for values in arrays):
        raise ValueError(f"{name} data and axes must be finite")
    expected_fast = np.arange(fast_count, dtype=float) / sample_rate_hz
    expected_slow = np.arange(slow_count, dtype=float) * repetition_s
    if not np.allclose(cube.fast_time_s, expected_fast, rtol=0.0, atol=1e-15):
        raise ValueError(f"{name} fast_time_s does not match waveform")
    if not np.allclose(cube.slow_time_s, expected_slow, rtol=0.0, atol=1e-15):
        raise ValueError(f"{name} slow_time_s does not match waveform")


def _validate_metadata(metadata, expected, name):
    for key, value in expected.items():
        supplied = metadata.get(key)
        if supplied is not None and (not np.isscalar(supplied) or not np.isclose(supplied, value, rtol=1e-12, atol=1e-15)):
            raise ValueError(f"{name} metadata {key} does not match waveform")

def _fft_size(configured, required, name):
    size = required if configured is None else configured
    if size < required:
        raise ValueError(f"{name} must be at least {required}")
    return size


def _slow_time_fft(range_data, radar, repetition_s, window_name, fft_size):
    slow_count = range_data.shape[1]
    size = _fft_size(fft_size, slow_count, "doppler_fft_size")
    window = get_window(window_name, slow_count)
    spectrum = np.fft.fftshift(
        np.fft.fft(
            range_data * window[None, :, None, None], n=size, axis=1
        ),
        axes=1,
    )
    doppler_hz = np.fft.fftshift(np.fft.fftfreq(size, repetition_s))
    velocity = -doppler_hz * radar.wavelength_m / 2
    order = np.argsort(velocity)
    return np.transpose(spectrum[:, order], (0, 2, 1, 3)), velocity[order]


def process_fmcw(beat, radar, waveform, processing):
    if beat.metadata.get("frontend") != "fmcw_dechirp":
        raise ValueError("process_fmcw requires dechirped FMCW data")
    fast_count = beat.data.shape[-1]
    if beat.metadata.get("waveform_kind") != "fmcw":
        raise ValueError("process_fmcw requires FMCW waveform metadata")
    _validate_metadata(beat.metadata, {
        "sample_rate_hz": waveform.sample_rate_hz,
        "slope_hz_per_s": waveform.slope_hz_per_s,
        "bandwidth_hz": waveform.bandwidth_hz,
        "chirp_duration_s": waveform.chirp_duration_s,
        "repetition_interval_s": waveform.chirp_repetition_interval_s,
    }, "FMCW")
    _validate_received_cube(
        beat, waveform.frame_count, waveform.chirps_per_frame,
        round(waveform.chirp_duration_s * waveform.sample_rate_hz),
        waveform.sample_rate_hz, waveform.chirp_repetition_interval_s, "FMCW",
    )
    range_size = _fft_size(
        processing.range_fft_size, fast_count, "range_fft_size"
    )
    window = get_window(processing.range_window, fast_count)
    fast_spectrum = np.fft.fft(
        beat.data * window[None, None, None, :], n=range_size, axis=-1
    )
    frequencies = np.fft.fftfreq(range_size, 1 / waveform.sample_rate_hz)
    keep = frequencies <= 0
    ranges = -frequencies[keep] * C_MPS / (2 * waveform.slope_hz_per_s)
    order = np.argsort(ranges)
    range_data = fast_spectrum[..., keep][..., order]
    ranges = ranges[order]
    spectrum, velocity = _slow_time_fft(
        range_data,
        radar,
        waveform.chirp_repetition_interval_s,
        processing.doppler_window,
        processing.doppler_fft_size,
    )
    range_resolution = C_MPS / (2 * waveform.bandwidth_hz)
    velocity_resolution = (np.inf if waveform.chirps_per_frame == 1 else radar.wavelength_m / (
        2 * waveform.chirps_per_frame * waveform.chirp_repetition_interval_s
    ))
    return RangeDopplerResult(
        spectrum,
        np.abs(spectrum) ** 2,
        ranges,
        velocity,
        range_resolution,
        velocity_resolution,
        C_MPS * waveform.sample_rate_hz / (4 * waveform.slope_hz_per_s),
        radar.wavelength_m / (4 * waveform.chirp_repetition_interval_s),
    )


def process_lfm(frontend, radar, waveform, processing):
    rx = frontend.received
    frames, slow, channels, fast = rx.data.shape
    if rx.metadata.get("waveform_kind") != "lfm":
        raise ValueError("process_lfm requires LFM frontend data")
    _validate_metadata(rx.metadata, {
        "sample_rate_hz": waveform.sample_rate_hz,
        "bandwidth_hz": waveform.bandwidth_hz,
        "pulse_width_s": waveform.pulse_width_s,
        "slope_hz_per_s": waveform.slope_hz_per_s,
        "repetition_interval_s": waveform.pulse_repetition_interval_s,
    }, "LFM")
    _validate_received_cube(
        rx, waveform.frame_count, waveform.pulses_per_frame,
        round(waveform.pulse_repetition_interval_s * waveform.sample_rate_hz),
        waveform.sample_rate_hz, waveform.pulse_repetition_interval_s, "LFM",
    )
    reference_size = frontend.matched_filter_reference.size
    expected_reference = np.conj(LFMWaveform(waveform).reference()[::-1])
    if reference_size != expected_reference.size or not np.allclose(frontend.matched_filter_reference, expected_reference, rtol=1e-12, atol=1e-15):
        raise ValueError("LFM matched_filter_reference does not match waveform")
    compressed = np.empty(
        (frames, slow, channels, fast + reference_size - 1), complex
    )
    for frame in range(frames):
        for pulse in range(slow):
            for channel in range(channels):
                compressed[frame, pulse, channel] = fftconvolve(
                    rx.data[frame, pulse, channel],
                    frontend.matched_filter_reference,
                    mode="full",
                )
    delay_samples = np.arange(compressed.shape[-1]) - (reference_size - 1)
    valid = (delay_samples >= 0) & (
        delay_samples / waveform.sample_rate_hz
        < waveform.pulse_repetition_interval_s
    )
    ranges = delay_samples[valid] / waveform.sample_rate_hz * C_MPS / 2
    spectrum, velocity = _slow_time_fft(
        compressed[..., valid],
        radar,
        waveform.pulse_repetition_interval_s,
        processing.doppler_window,
        processing.doppler_fft_size,
    )
    return RangeDopplerResult(
        spectrum,
        np.abs(spectrum) ** 2,
        ranges,
        velocity,
        C_MPS / (2 * waveform.bandwidth_hz),
        (np.inf if waveform.pulses_per_frame == 1 else radar.wavelength_m
         / (2 * waveform.pulses_per_frame * waveform.pulse_repetition_interval_s)),
        C_MPS * waveform.pulse_repetition_interval_s / 2,
        radar.wavelength_m / (4 * waveform.pulse_repetition_interval_s),
    )
