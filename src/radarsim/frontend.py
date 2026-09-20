"""Waveform-specific receive frontends (no range-Doppler processing)."""

import numpy as np

from .models import LFMFrontendResult, SignalCube
from .waveforms.lfm import LFMWaveform


def _finite(data: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(data.real)) or not np.all(np.isfinite(data.imag)):
        raise ValueError(f"{name} data must be finite")


def dechirp_fmcw(rx: SignalCube, tx: SignalCube) -> SignalCube:
    """Mix Rx with conjugate Tx reference while preserving Rx axes."""
    if rx.dimensions != ("frame", "slow_time", "rx", "fast_time"):
        raise ValueError("rx dimensions must be canonical frame/slow_time/rx/fast_time")
    if tx.dimensions != ("frame", "slow_time", "tx", "fast_time"):
        raise ValueError("tx dimensions must be canonical frame/slow_time/tx/fast_time")
    if tx.data.shape[2] != 1:
        raise ValueError("initial FMCW frontend requires one Tx reference")
    if (rx.data.shape[0], rx.data.shape[1], rx.data.shape[3]) != (tx.data.shape[0], tx.data.shape[1], tx.data.shape[3]):
        raise ValueError("Tx and Rx require matching frame, slow-time, and fast-time shapes")
    if not np.array_equal(rx.fast_time_s, tx.fast_time_s):
        raise ValueError("Tx and Rx require matching fast-time axes")
    if not np.array_equal(rx.slow_time_s, tx.slow_time_s):
        raise ValueError("Tx and Rx require matching slow-time axes")
    if rx.sample_times_s.shape != tx.sample_times_s.shape or not np.allclose(
        rx.sample_times_s, tx.sample_times_s, rtol=0.0, atol=1e-15
    ):
        raise ValueError("Tx and Rx require matching sample_times_s")
    if rx.metadata.get("waveform_kind") != "fmcw" or tx.metadata.get("waveform_kind") != "fmcw":
        raise ValueError("FMCW frontend requires FMCW waveform metadata")
    _finite(rx.data, "Rx")
    _finite(tx.data, "Tx")
    data = rx.data * np.conj(tx.data[:, :, :1, :])
    metadata = dict(rx.metadata)
    metadata["frontend"] = "fmcw_dechirp"
    return SignalCube(data, ("frame", "slow_time", "rx", "fast_time"), rx.fast_time_s, rx.slow_time_s, rx.sample_times_s, metadata)


def prepare_lfm(rx: SignalCube, waveform: object) -> LFMFrontendResult:
    """Prepare Rx and a conjugate time-reversed LFM reference."""
    if rx.dimensions != ("frame", "slow_time", "rx", "fast_time"):
        raise ValueError("rx dimensions must be canonical frame/slow_time/rx/fast_time")
    if rx.metadata.get("waveform_kind") != "lfm":
        raise ValueError("LFM frontend requires LFM waveform metadata")
    if not isinstance(waveform, LFMWaveform):
        raise ValueError("waveform must be an LFMWaveform")
    expected = {
        "sample_rate_hz": waveform.sample_rate_hz,
        "bandwidth_hz": waveform.config.bandwidth_hz,
        "pulse_width_s": waveform.config.pulse_width_s,
        "slope_hz_per_s": waveform.config.slope_hz_per_s,
        "repetition_interval_s": waveform.repetition_interval_s,
        "pulses_per_frame": waveform.slow_time_count,
        "frame_count": waveform.frame_count,
    }
    for name, value in expected.items():
        supplied = rx.metadata.get(name)
        if supplied is not None and (not np.isscalar(supplied) or not np.isclose(supplied, value, rtol=1e-12, atol=1e-15)):
            raise ValueError(f"LFM waveform metadata {name} does not match waveform")
    if rx.data.shape[0] != waveform.frame_count or rx.data.shape[1] != waveform.slow_time_count:
        raise ValueError("LFM waveform frame/slow-time shape does not match receive data")
    reference = np.asarray(waveform.reference(), dtype=np.complex128)
    if reference.ndim != 1 or reference.size == 0:
        raise ValueError("LFM reference must be a non-empty one-dimensional array")
    if reference.size > rx.data.shape[-1]:
        raise ValueError("LFM reference cannot exceed receive fast-time length")
    _finite(rx.data, "Rx")
    _finite(reference, "LFM reference")
    return LFMFrontendResult(rx, np.conj(reference[::-1]))
