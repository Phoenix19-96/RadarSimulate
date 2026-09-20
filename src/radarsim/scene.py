from dataclasses import dataclass
import numbers
from typing import Sequence
import numpy as np
from numpy.typing import NDArray
from .geometry import polar_to_cartesian

@dataclass(frozen=True)
class CartesianWaypoint:
    time_s: float
    position_m: tuple[float, float, float]

@dataclass(frozen=True)
class PolarWaypoint:
    time_s: float
    range_m: float
    azimuth_deg: float
    elevation_deg: float

@dataclass(frozen=True, eq=False)
class TargetState:
    position_m: NDArray[np.float64]
    velocity_mps: NDArray[np.float64]

class Trajectory:
    def __init__(self, times_s, positions_m):
        try:
            self.times_s = np.array(times_s, dtype=float, copy=True)
            self.positions_m = np.array(positions_m, dtype=float, copy=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("waypoint times and positions must be numeric") from exc
        if self.times_s.ndim != 1 or self.times_s.size < 2: raise ValueError("trajectory requires two waypoints")
        if self.positions_m.shape != (self.times_s.size, 3): raise ValueError("positions must have shape [waypoint, xyz]")
        if not np.all(np.isfinite(self.times_s)) or not np.all(np.isfinite(self.positions_m)): raise ValueError("waypoint times and positions must be finite")
        if np.any(self.times_s[1:] <= self.times_s[:-1]): raise ValueError("waypoint times must be strictly increasing")
        slopes = np.empty((self.times_s.size - 1, 3), dtype=float)
        for index, (p0, p1, t0, t1) in enumerate(zip(self.positions_m[:-1], self.positions_m[1:], self.times_s[:-1], self.times_s[1:])):
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                direct_delta = p1 - p0
                direct_dt = t1 - t0
                if np.all(np.isfinite(direct_delta)) and np.isfinite(direct_dt):
                    slopes[index] = direct_delta / direct_dt
                else:
                    position_scale = max(float(np.max(np.abs(p0))), float(np.max(np.abs(p1))))
                    time_scale = max(abs(float(t0)), abs(float(t1)))
                    normalized_delta = (p1 / position_scale - p0 / position_scale) if position_scale else np.zeros(3)
                    normalized_dt = t1 / time_scale - t0 / time_scale if time_scale else 0.0
                    slopes[index] = (normalized_delta / normalized_dt) * (position_scale / time_scale)
        if not np.all(np.isfinite(slopes)):
            raise ValueError("trajectory segment velocity is not finite")
        self._slopes = slopes
        self.times_s.setflags(write=False); self.positions_m.setflags(write=False)
        self._slopes.setflags(write=False)

    @classmethod
    def from_cartesian(cls, points: Sequence[CartesianWaypoint]):
        for point in points:
            if not isinstance(point.time_s, numbers.Real) or isinstance(point.time_s, bool):
                raise ValueError("time_s must be numeric")
            try:
                position = np.asarray(point.position_m)
            except (TypeError, ValueError) as exc:
                raise ValueError("position_m must be numeric") from exc
            if position.shape != (3,) or position.dtype.kind in "OUS":
                raise ValueError("position_m must contain three numeric coordinates")
        return cls([p.time_s for p in points], [p.position_m for p in points])

    @classmethod
    def from_polar(cls, points: Sequence[PolarWaypoint]):
        for point in points:
            for name, value in (("time_s", point.time_s), ("range_m", point.range_m), ("azimuth_deg", point.azimuth_deg), ("elevation_deg", point.elevation_deg)):
                if not isinstance(value, numbers.Real) or isinstance(value, bool):
                    raise ValueError(f"{name} must be numeric")
        return cls([p.time_s for p in points], [polar_to_cartesian(p.range_m, p.azimuth_deg, p.elevation_deg) for p in points])

    def state_at(self, time_s: float) -> TargetState:
        try:
            valid_time = np.isscalar(time_s) and np.isfinite(time_s)
        except TypeError as exc:
            raise ValueError("time_s must be a finite scalar") from exc
        if not valid_time: raise ValueError("time_s must be a finite scalar")
        time_s = float(time_s)
        if not self.times_s[0] <= time_s <= self.times_s[-1]: raise ValueError("time_s is outside trajectory coverage")
        i = min(int(np.searchsorted(self.times_s, time_s, side="right") - 1), self.times_s.size - 2)
        t0 = self.times_s[i]; p0 = self.positions_m[i]
        velocity = self._slopes[i]
        if time_s == self.times_s[i + 1]:
            return TargetState(self.positions_m[i + 1].copy(), velocity.copy())
        return TargetState((p0 + (time_s-t0)*velocity).copy(), velocity.copy())

    def states_at(self, times_s):
        times = np.asarray(times_s, dtype=float)
        states = [self.state_at(float(t)) for t in times.ravel()]
        shape = times.shape + (3,)
        if not states:
            empty = np.empty(shape, dtype=float); return empty.copy(), empty.copy()
        return np.stack([s.position_m for s in states]).reshape(shape), np.stack([s.velocity_mps for s in states]).reshape(shape)

@dataclass(frozen=True)
class Target:
    target_id: str
    name: str
    rcs_dbsm: float
    trajectory: Trajectory

    def __post_init__(self) -> None:
        if not isinstance(self.target_id, str) or not self.target_id: raise ValueError("target_id must be non-empty")
        if not isinstance(self.name, str): raise ValueError("name must be a string")
        if not isinstance(self.rcs_dbsm, numbers.Real) or isinstance(self.rcs_dbsm, bool) or not np.isfinite(self.rcs_dbsm): raise ValueError("rcs_dbsm must be a finite numeric scalar")
        try:
            if 10.0 ** (self.rcs_dbsm / 10.0) <= 0.0 or not np.isfinite(10.0 ** (self.rcs_dbsm / 10.0)):
                raise ValueError("rcs_dbsm conversion is not representable")
        except OverflowError as exc:
            raise ValueError("rcs_dbsm conversion is not representable") from exc
        if not isinstance(self.trajectory, Trajectory): raise TypeError("trajectory must be a Trajectory")

    @property
    def rcs_m2(self) -> float:
        return float(10.0 ** (self.rcs_dbsm / 10.0))
