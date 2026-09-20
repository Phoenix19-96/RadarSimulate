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


@dataclass(frozen=True, eq=False)
class CFARResult:
    detections: NDArray[np.bool_]
    threshold_w: NDArray[np.float64]
    noise_w: NDArray[np.float64]

    def __post_init__(self) -> None:
        detections = np.asarray(self.detections)
        threshold = np.asarray(self.threshold_w)
        noise = np.asarray(self.noise_w)
        if detections.ndim != 2 or detections.dtype != np.bool_:
            raise ValueError("detections must be a two-dimensional boolean array")
        if threshold.shape != detections.shape or noise.shape != detections.shape:
            raise ValueError("CFAR maps must have matching two-dimensional shapes")
        if not np.all(np.isfinite(threshold) | np.isnan(threshold)) or not np.all(np.isfinite(noise) | np.isnan(noise)):
            raise ValueError("CFAR maps may contain only finite values or NaN for invalid cells")
        if np.any(threshold[np.isfinite(threshold)] < 0) or np.any(noise[np.isfinite(noise)] < 0):
            raise ValueError("CFAR maps must be non-negative")
        for name, value, dtype in (("detections", detections, np.bool_), ("threshold_w", threshold, np.float64), ("noise_w", noise, np.float64)):
            copied = np.array(value, dtype=dtype, copy=True)
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)


@dataclass(frozen=True)
class Detection:
    frame_index: int
    channel_index: int
    range_m: float
    velocity_mps: float
    power_w: float
    noise_w: float
    snr_db: float
    doppler_index: int = 0
    range_index: int = 0

    @property
    def rx_index(self) -> int:
        return self.channel_index

    @property
    def doppler_bin_index(self) -> int:
        return self.doppler_index

    @property
    def range_bin_index(self) -> int:
        return self.range_index


@dataclass(frozen=True)
class TruthRecord:
    frame_index: int
    target_id: str
    time_s: float
    x_m: float
    y_m: float
    z_m: float
    range_m: float
    radial_velocity_mps: float
    azimuth_deg: float
    elevation_deg: float
    rcs_dbsm: float


@dataclass(frozen=True)
class SimulationResult:
    tx: SignalCube
    rx: SignalCube
    frontend: SignalCube | LFMFrontendResult
    range_doppler: RangeDopplerResult
    cfar: tuple[tuple[CFARResult, ...], ...]
    detections: tuple[Detection, ...]
    truth: tuple[TruthRecord, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.tx, SignalCube) or self.tx.dimensions != (
            "frame", "slow_time", "tx", "fast_time",
        ):
            raise ValueError("tx must be a Tx SignalCube")
        if not isinstance(self.rx, SignalCube) or self.rx.dimensions != (
            "frame", "slow_time", "rx", "fast_time",
        ):
            raise ValueError("rx must be an Rx SignalCube")
        if (self.tx.data.shape[0], self.tx.data.shape[1], self.tx.data.shape[3]) != (
            self.rx.data.shape[0], self.rx.data.shape[1], self.rx.data.shape[3],
        ):
            raise ValueError("tx and rx frame, slow-time, and fast-time shapes must match")
        frontend_rx = (
            self.frontend if isinstance(self.frontend, SignalCube)
            else self.frontend.received if isinstance(self.frontend, LFMFrontendResult)
            else None
        )
        if frontend_rx is None or frontend_rx.dimensions != (
            "frame", "slow_time", "rx", "fast_time",
        ):
            raise ValueError("frontend must retain an Rx SignalCube")
        if (frontend_rx.data.shape != self.rx.data.shape
                or not np.array_equal(frontend_rx.fast_time_s, self.rx.fast_time_s)
                or not np.array_equal(frontend_rx.slow_time_s, self.rx.slow_time_s)
                or not np.array_equal(frontend_rx.sample_times_s, self.rx.sample_times_s)):
            raise ValueError("frontend axes and shape must match rx")
        if (isinstance(self.frontend, LFMFrontendResult)
                and not np.array_equal(frontend_rx.data, self.rx.data)):
            raise ValueError("frontend received data must match rx")
        if not isinstance(self.range_doppler, RangeDopplerResult):
            raise ValueError("range_doppler must be a RangeDopplerResult")
        frames, channels = self.range_doppler.power_w.shape[:2]
        if (frames, channels) != self.rx.data.shape[:3:2]:
            raise ValueError("range_doppler frame and channel dimensions must match rx")
        if len(self.cfar) != frames or any(len(row) != channels for row in self.cfar):
            raise ValueError("cfar results must be nested by range-Doppler frame and channel")
        for row in self.cfar:
            for result in row:
                if (not isinstance(result, CFARResult)
                        or result.detections.shape != self.range_doppler.power_w.shape[-2:]):
                    raise ValueError("CFAR result shape must match range-Doppler map")
        if not all(isinstance(detection, Detection) for detection in self.detections):
            raise ValueError("detections must contain Detection values")
        if not all(isinstance(record, TruthRecord) for record in self.truth):
            raise ValueError("truth must contain TruthRecord values")
