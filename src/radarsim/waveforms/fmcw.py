from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import NDArray

from radarsim.config import RadarConfig
from radarsim.models import SignalCube
from .base import _repetition_timing


@dataclass(frozen=True)
class FMCWConfig:
    bandwidth_hz: float
    chirp_duration_s: float
    chirp_repetition_interval_s: float
    sample_rate_hz: float
    chirps_per_frame: int
    frame_count: int = 1

    def __post_init__(self) -> None:
        for name, value in (("bandwidth_hz", self.bandwidth_hz), ("chirp_duration_s", self.chirp_duration_s), ("chirp_repetition_interval_s", self.chirp_repetition_interval_s), ("sample_rate_hz", self.sample_rate_hz)):
            if not isinstance(value, Real) or isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        if self.chirp_repetition_interval_s < self.chirp_duration_s:
            raise ValueError("chirp repetition interval must cover chirp duration")
        for name, value in (("chirps_per_frame", self.chirps_per_frame), ("frame_count", self.frame_count)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        sample_count = self.chirp_duration_s * self.sample_rate_hz
        tolerance = 32 * np.finfo(float).eps * max(1.0, abs(sample_count))
        if sample_count < 1 or abs(sample_count - round(sample_count)) > tolerance:
            raise ValueError("FMCW chirp duration and sample_rate_hz must yield an integer sample count")

    @property
    def slope_hz_per_s(self) -> float:
        return self.bandwidth_hz / self.chirp_duration_s


class FMCWWaveform:
    def __init__(self, config: FMCWConfig) -> None:
        self.config = config
        self.sample_rate_hz = config.sample_rate_hz
        self.slow_time_count = config.chirps_per_frame
        self.frame_count = config.frame_count
        self.repetition_interval_s = config.chirp_repetition_interval_s

    def _timing_at(self, times_s: NDArray[np.float64]) -> tuple[
        NDArray[np.float64], NDArray[np.bool_], NDArray[np.intp], NDArray[np.bool_],
    ]:
        times = np.asarray(times_s, dtype=float)
        return _repetition_timing(
            times, self.repetition_interval_s, self.config.chirp_duration_s,
            self.slow_time_count, self.frame_count, self.sample_rate_hz,
        )

    def sample_at(self, times_s: NDArray[np.float64]) -> NDArray[np.complex128]:
        times = np.asarray(times_s, dtype=float)
        local, active, _, _ = self._timing_at(times)
        out = np.zeros(times.shape, dtype=np.complex128)
        out[active] = np.exp(1j * np.pi * self.config.slope_hz_per_s * local[active] ** 2)
        return out

    def reference(self) -> NDArray[np.complex128]:
        count = round(self.config.chirp_duration_s * self.sample_rate_hz)
        times = np.arange(count, dtype=float) / self.sample_rate_hz
        return self.sample_at(times)

    def build_tx(self, radar: RadarConfig) -> SignalCube:
        count = round(self.config.chirp_duration_s * self.sample_rate_hz)
        fast = np.arange(count, dtype=float) / self.sample_rate_hz
        slow = np.arange(self.slow_time_count, dtype=float) * self.repetition_interval_s
        frame_span = self.slow_time_count * self.repetition_interval_s
        times = (
            np.arange(self.frame_count, dtype=float)[:, None, None] * frame_span
            + slow[None, :, None]
            + fast[None, None, :]
        )
        data = np.broadcast_to(self.sample_at(fast), (self.frame_count, self.slow_time_count, count))[:, :, None, :].copy()
        return SignalCube(
            data,
            ("frame", "slow_time", "tx", "fast_time"),
            fast,
            slow,
            times,
            {"waveform_kind": "fmcw", "sample_rate_hz": self.sample_rate_hz,
             "slope_hz_per_s": self.config.slope_hz_per_s,
             "bandwidth_hz": self.config.bandwidth_hz,
             "chirp_duration_s": self.config.chirp_duration_s,
             "chirps_per_frame": self.slow_time_count,
             "frame_count": self.frame_count,
             "repetition_interval_s": self.repetition_interval_s},
        )
