from dataclasses import dataclass
from numbers import Real
from typing import Mapping

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, eq=False)
class SignalCube:
    data: NDArray[np.complex128]
    dimensions: tuple[str, str, str, str]
    fast_time_s: NDArray[np.float64]
    slow_time_s: NDArray[np.float64]
    sample_times_s: NDArray[np.float64]
    metadata: Mapping[str, float | int | str]

    def __post_init__(self) -> None:
        if self.data.ndim != 4:
            raise ValueError("data must be four-dimensional")
        if self.dimensions not in (
            ("frame", "slow_time", "tx", "fast_time"),
            ("frame", "slow_time", "rx", "fast_time"),
        ):
            raise ValueError("dimensions must be canonical frame/slow_time/tx|rx/fast_time")
        frames, slow, _, fast = self.data.shape
        if self.fast_time_s.shape != (fast,):
            raise ValueError("fast_time_s length must match data")
        if self.slow_time_s.shape != (slow,):
            raise ValueError("slow_time_s length must match data")
        if self.sample_times_s.shape != (frames, slow, fast):
            raise ValueError("sample_times_s shape must be [frame, slow_time, fast_time]")


@dataclass(frozen=True, eq=False)
class LFMFrontendResult:
    received: SignalCube
    matched_filter_reference: NDArray[np.complex128]

    def __post_init__(self) -> None:
        if not isinstance(self.received, SignalCube) or self.received.dimensions != ("frame", "slow_time", "rx", "fast_time"):
            raise ValueError("received must be an Rx SignalCube")
        reference = np.asarray(self.matched_filter_reference)
        if reference.ndim != 1 or reference.size == 0:
            raise ValueError("matched_filter_reference must be a non-empty one-dimensional array")
        if not np.iscomplexobj(reference):
            raise ValueError("matched_filter_reference must be complex")
        if not np.all(np.isfinite(reference.real)) or not np.all(np.isfinite(reference.imag)):
            raise ValueError("matched_filter_reference must be finite")
        frozen = np.array(reference, dtype=np.complex128, copy=True)
        frozen.setflags(write=False)
        object.__setattr__(self, "matched_filter_reference", frozen)


@dataclass(frozen=True, eq=False)
class RangeDopplerResult:
    spectrum: NDArray[np.complex128]
    power_w: NDArray[np.float64]
    range_m: NDArray[np.float64]
    velocity_mps: NDArray[np.float64]
    range_resolution_m: float
    velocity_resolution_mps: float
    max_unambiguous_range_m: float
    max_unambiguous_velocity_mps: float

    def __post_init__(self) -> None:
        spectrum = np.asarray(self.spectrum)
        power = np.asarray(self.power_w)
        ranges = np.asarray(self.range_m)
        velocity = np.asarray(self.velocity_mps)
        if spectrum.ndim != 4 or not np.iscomplexobj(spectrum):
            raise ValueError("spectrum must be a four-dimensional complex array")
        if power.ndim != 4 or power.shape != spectrum.shape or np.iscomplexobj(power):
            raise ValueError("power_w must be a real array matching spectrum")
        if min(spectrum.shape) < 1:
            raise ValueError("spectrum dimensions must be non-empty")
        if ranges.ndim != 1 or ranges.size != spectrum.shape[-1]:
            raise ValueError("range_m must match spectrum range bins")
        if velocity.ndim != 1 or velocity.size != spectrum.shape[-2]:
            raise ValueError("velocity_mps must match spectrum Doppler bins")
        if not all(np.all(np.isfinite(item)) for item in (spectrum.real, spectrum.imag, power, ranges, velocity)):
            raise ValueError("range-Doppler arrays must be finite")
        if np.any(power < 0) or np.any(ranges < 0):
            raise ValueError("power_w must be non-negative and range_m non-negative")
        if ranges.size > 1 and np.any(np.diff(ranges) <= 0):
            raise ValueError("range_m must be strictly increasing")
        if velocity.size > 1 and np.any(np.diff(velocity) <= 0):
            raise ValueError("velocity_mps must be strictly increasing")
        scalar_fields = (
            ("range_resolution_m", self.range_resolution_m, False),
            ("velocity_resolution_mps", self.velocity_resolution_mps, True),
            ("max_unambiguous_range_m", self.max_unambiguous_range_m, False),
            ("max_unambiguous_velocity_mps", self.max_unambiguous_velocity_mps, False),
        )
        for name, value, allow_positive_infinity in scalar_fields:
            valid_infinity = (allow_positive_infinity and isinstance(value, Real)
                              and not isinstance(value, (bool, np.bool_)) and np.isposinf(value))
            if (not isinstance(value, Real) or isinstance(value, (bool, np.bool_))
                    or value <= 0 or (not np.isfinite(value) and not valid_infinity)):
                suffix = "positive (or +inf when Doppler resolution is unavailable)" if allow_positive_infinity else "finite and positive"
                raise ValueError(f"{name} must be {suffix}")
        for name, value, dtype in (("spectrum", spectrum, np.complex128), ("power_w", power, np.float64), ("range_m", ranges, np.float64), ("velocity_mps", velocity, np.float64)):
            copied = np.array(value, dtype=dtype, copy=True)
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)
