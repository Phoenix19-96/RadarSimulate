"""Explicit monostatic free-space propagation for complex baseband signals."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import C_MPS
from .models import SignalCube


def radar_received_power_w(
    radar: object, range_m: ArrayLike, rcs_m2: ArrayLike,
) -> float | NDArray[np.float64]:
    """Return monostatic received power from the free-space radar equation.

    ``range_m`` and ``rcs_m2`` may be scalars or broadcast-compatible arrays.
    Each must be finite and strictly positive; accepting invalid values would
    otherwise silently create non-finite complex samples downstream.
    """
    ranges = np.asarray(range_m, dtype=float)
    rcs = np.asarray(rcs_m2, dtype=float)
    if not np.all(np.isfinite(ranges)) or np.any(ranges <= 0):
        raise ValueError("range_m must be finite and positive")
    if not np.all(np.isfinite(rcs)) or np.any(rcs <= 0):
        raise ValueError("rcs_m2 must be finite and positive")

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            gain_tx = np.power(10.0, radar.tx_gain_db / 10.0)
            gain_rx = np.power(10.0, radar.rx_gain_db / 10.0)
            power = (
                radar.transmit_power_w * gain_tx * gain_rx * radar.wavelength_m**2 * rcs
                / ((4.0 * np.pi) ** 3 * ranges**4)
            )
        except FloatingPointError as exc:
            raise ValueError("received power is not finite") from exc
    if not np.all(np.isfinite(power)) or np.any(power < 0):
        raise ValueError("received power is not finite")
    return float(power) if power.ndim == 0 else power


def _sample_tx_modulation(
    tx: SignalCube, waveform: object, query_times_s: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """Interpolate Tx-sample weights while retaining waveform analytic phase.

    The supplied Tx cube is divided by the waveform's nominal samples on the
    transmit grid.  Those complex modulation weights are interpolated only
    inside active samples of the same waveform repetition; the analytic
    waveform remains responsible for fractional-delay chirp phase and support.
    """
    source_times = tx.sample_times_s.reshape(-1)
    if source_times.size < 2 or np.any(source_times[1:] <= source_times[:-1]):
        raise ValueError("tx sample_times_s must be strictly increasing")

    nominal = np.asarray(waveform.sample_at(tx.sample_times_s), dtype=np.complex128)
    if nominal.shape != tx.sample_times_s.shape:
        raise ValueError("waveform samples must match tx sample_times_s")
    active = np.abs(nominal) > 0.0
    weights = np.zeros_like(nominal)
    np.divide(tx.data[:, :, 0, :], nominal, out=weights, where=active)

    repetitions_per_frame = tx.data.shape[1]
    total_repetitions = tx.data.shape[0] * repetitions_per_frame
    local_times, _, repetition, valid = waveform._timing_at(query_times_s)
    local_times = local_times.reshape(-1)
    repetition = repetition.reshape(-1)
    valid = valid.reshape(-1)
    if np.any(valid & ((repetition < 0) | (repetition >= total_repetitions))):
        raise ValueError("waveform repetition map must match tx dimensions")
    repetition = np.clip(repetition, 0, total_repetitions - 1)
    frame, slow_time = divmod(repetition, repetitions_per_frame)

    fast_times = tx.fast_time_s
    right = np.searchsorted(fast_times, local_times, side="right")
    left = np.clip(right - 1, 0, fast_times.size - 1)
    right = np.clip(right, 0, fast_times.size - 1)
    sampled = weights[frame, slow_time, left].copy()

    intervals = fast_times[right] - fast_times[left]
    interpolate = (
        valid & active[frame, slow_time, left] & active[frame, slow_time, right]
        & (intervals > 0.0)
    )
    fraction = np.zeros_like(local_times)
    fraction[interpolate] = (
        (local_times[interpolate] - fast_times[left[interpolate]])
        / intervals[interpolate]
    )
    sampled[interpolate] = (
        (1.0 - fraction[interpolate])
        * weights[frame[interpolate], slow_time[interpolate], left[interpolate]]
        + fraction[interpolate]
        * weights[frame[interpolate], slow_time[interpolate], right[interpolate]]
    )
    sampled[~valid] = 0.0
    return sampled.reshape(query_times_s.shape)


class MonostaticChannel:
    """Sum delayed target echoes and deterministic-seed complex white noise."""

    def propagate(self, tx: SignalCube, waveform: object, radar: object,
                  targets: Sequence[object], noise: object) -> SignalCube:
        """Propagate the 1T1R transmit waveform to the receiver sample times.

        Target state is evaluated at every absolute receive time.  The
        time-varying two-way carrier-delay phase is therefore also the Doppler
        model, with receding targets producing negative physical Doppler.
        """
        if tx.dimensions != ("frame", "slow_time", "tx", "fast_time"):
            raise ValueError("tx dimensions must be frame/slow_time/tx/fast_time")
        if tx.data.shape[2] != radar.num_tx:
            raise ValueError("tx channel count must match radar.num_tx")
        if tx.data.shape[2] != 1:
            raise ValueError("only one transmit channel is supported")

        times = np.asarray(tx.sample_times_s, dtype=float)
        if not np.all(np.isfinite(times)):
            raise ValueError("tx sample_times_s must be finite")
        if not np.all(np.isfinite(tx.data.real)) or not np.all(np.isfinite(tx.data.imag)):
            raise ValueError("tx data must be finite")

        try:
            noise_power = noise.resolved_power_w()
        except (FloatingPointError, OverflowError) as exc:
            raise ValueError(
                "noise_figure_db produces unrepresentable noise power"
            ) from exc
        noise_parameter = "noise_power_w" if noise.noise_power_w is not None else "noise_figure_db"
        if not np.isfinite(noise_power) or noise_power < 0:
            raise ValueError(
                f"{noise_parameter} must resolve to finite non-negative noise power"
            )

        # Validate every trajectory before allocating the potentially large Rx cube.
        start_s = float(times.min())
        end_s = float(times.max())
        for target in targets:
            if (target.trajectory.times_s[0] > start_s
                    or target.trajectory.times_s[-1] < end_s):
                raise ValueError("simulation time is outside trajectory coverage")

        received = np.zeros(
            (tx.data.shape[0], tx.data.shape[1], radar.num_rx, tx.data.shape[3]),
            dtype=np.complex128,
        )
        radar_position = np.asarray(radar.position_m, dtype=float)
        for target in targets:
            positions, velocities = target.trajectory.states_at(times)
            if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
                raise ValueError("target states must be finite")
            offsets = positions - radar_position
            ranges = np.linalg.norm(offsets, axis=-1)
            if not np.all(np.isfinite(ranges)) or np.any(ranges <= 0):
                raise ValueError("target range must be finite and non-zero")

            delay_s = 2.0 * ranges / C_MPS
            delayed_times = times - delay_s
            delayed_waveform = waveform.sample_at(delayed_times)
            if (not np.all(np.isfinite(delayed_waveform.real))
                    or not np.all(np.isfinite(delayed_waveform.imag))):
                raise ValueError("waveform delayed samples must be finite")
            delayed_tx = delayed_waveform * _sample_tx_modulation(tx, waveform, delayed_times)
            received_power = radar_received_power_w(radar, ranges, target.rcs_m2)
            phase_argument = -2.0 * np.pi * radar.carrier_hz * delay_s
            if not np.all(np.isfinite(phase_argument)):
                raise ValueError("carrier propagation phase must be finite")
            echo = np.sqrt(received_power) * delayed_tx * np.exp(1j * phase_argument)
            received += echo[:, :, None, :]

        if noise_power > 0:
            generator = np.random.default_rng(noise.seed)
            noise_scale = np.sqrt(noise_power / 2.0)
            received += noise_scale * (
                generator.standard_normal(received.shape)
                + 1j * generator.standard_normal(received.shape)
            )

        return SignalCube(
            received,
            ("frame", "slow_time", "rx", "fast_time"),
            tx.fast_time_s,
            tx.slow_time_s,
            times,
            dict(tx.metadata),
        )
