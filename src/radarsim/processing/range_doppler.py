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
    expected_times = (
        np.arange(frame_count)[:, None, None] * (slow_count * repetition_s)
        + expected_slow[None, :, None] + expected_fast[None, None, :]
    )
    if not np.allclose(cube.sample_times_s, expected_times, rtol=0.0, atol=1e-15):
        raise ValueError(f"{name} sample_times_s does not match waveform")


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


def _fmcw_acquisition_delay_s(waveform):
    count = round(waveform.chirp_duration_s * waveform.sample_rate_hz)
    last_sample = (count - 1) / waveform.sample_rate_hz
    previous_available = waveform.chirps_per_frame * waveform.frame_count > 1
    # Current-chirp delays extend through the last acquired sample. Previous
    # chirps fill the rest of [0,T) only if their support overlaps that interval.
    if (previous_available and waveform.chirp_repetition_interval_s
            - waveform.chirp_duration_s <= last_sample):
        return waveform.chirp_repetition_interval_s
    # Preserve the one-sample processor's degenerate zero-range cell.
    return last_sample if count > 1 else waveform.chirp_duration_s


def _fmcw_max_delay_s(waveform):
    return min(waveform.sample_rate_hz / (2 * waveform.slope_hz_per_s),
               _fmcw_acquisition_delay_s(waveform))


def _fmcw_delay_grid(waveform, range_size):
    acquisition_limit = _fmcw_acquisition_delay_s(waveform)
    if _fmcw_max_delay_s(waveform) == acquisition_limit:
        # Cell centers cover both ends of the acquired delay interval. For a
        # periodic interval a zero bin would own half the far-end range cell.
        step = waveform.sample_rate_hz / (range_size * waveform.slope_hz_per_s)
        count = max(1, int(np.ceil(acquisition_limit / step)))
        return (np.arange(count) + 0.5) * acquisition_limit / count
    frequencies = np.fft.fftfreq(range_size, 1 / waveform.sample_rate_hz)
    delays = np.sort(-frequencies[frequencies <= 0] / waveform.slope_hz_per_s)
    # Nyquist is a usable discrete bin; the acquisition/repetition endpoint
    # either has no current support or repeats zero delay and is excluded.
    return delays[delays < acquisition_limit]


def _fmcw_delay_kernel(reference, active, window):
    kernel = np.conj(reference) * active * window[None, :]
    energy = np.sum(np.abs(kernel) ** 2, axis=-1, keepdims=True)
    scale = np.sqrt(np.divide(np.sum(window**2), energy,
                              out=np.zeros_like(energy), where=energy > 0))
    return kernel * scale


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
    delays = _fmcw_delay_grid(waveform, range_size)
    fast = beat.fast_time_s[None, :]
    delayed = fast - delays[:, None]
    previous = delayed < 0
    local = np.where(previous, delayed + waveform.chirp_repetition_interval_s, delayed)
    active = (local >= 0) & (local < waveform.chirp_duration_s)
    # Match both -S*tau (current) and +S*(T-tau) (previous) beat branches
    # coherently, including their relative phase and active envelopes. This
    # generalizes the windowed range transform on the configured FFT grid.
    reference = np.exp(1j * np.pi * waveform.slope_hz_per_s * (local**2 - fast**2))
    kernel = _fmcw_delay_kernel(reference, active, window)
    doppler_data, velocity = _slow_time_fft(
        beat.data,
        radar,
        waveform.chirp_repetition_interval_s,
        processing.doppler_window,
        processing.doppler_fft_size,
    )
    # Transform order commutes. Matching after slow FFT lets us remove each
    # Doppler bin's fast-time phase, preventing small near-zero range beats
    # from changing sign and wrapping to the far range boundary.
    doppler_hz = -2 * velocity / radar.wavelength_m
    correction = np.exp(-2j * np.pi * doppler_hz[:, None] * fast)
    spectrum = np.einsum("fcvt,vt,rt->fcvr", doppler_data, correction, kernel)
    # No transmitted chirp exists before the first absolute receive record.
    first_weight = get_window(processing.doppler_window, waveform.chirps_per_frame)[0]
    first_kernel = _fmcw_delay_kernel(reference, active & ~previous, window)
    spectrum[0] -= first_weight * np.einsum(
        "ct,vt,rt->cvr", beat.data[0, 0], correction, kernel - first_kernel,
    )
    ranges = delays * C_MPS / 2
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
        C_MPS * _fmcw_max_delay_s(waveform) / 2,
        radar.wavelength_m / (4 * waveform.chirp_repetition_interval_s),
    )


def process_lfm(frontend, radar, waveform, processing):
    if processing.range_fft_size is not None:
        raise ValueError("range_fft_size is inapplicable to LFM matched filtering")
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
    # Convolve each receiver's continuous record. A matched output at absolute
    # delay pulse_start + delay belongs to that transmitted pulse, even when
    # its receive tail ends in the following PRI/frame. Valid convolution
    # excludes unobserved final tails; no prehistory or final padding is used.
    continuous = rx.data.transpose(2, 0, 1, 3).reshape(channels, -1)
    observed = fftconvolve(
        continuous, frontend.matched_filter_reference[None, :],
        mode="valid", axes=-1,
    )
    compressed = np.zeros_like(continuous)
    compressed[:, :observed.shape[-1]] = observed
    compressed = compressed.reshape(channels, frames, slow, fast).transpose(1, 2, 0, 3)
    range_count = fast if slow > 1 else fast - reference_size + 1
    ranges = np.arange(range_count) / waveform.sample_rate_hz * C_MPS / 2
    spectrum, velocity = _slow_time_fft(
        compressed[..., :range_count],
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
        C_MPS * range_count / (2 * waveform.sample_rate_hz),
        radar.wavelength_m / (4 * waveform.pulse_repetition_interval_s),
    )
