from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from radarsim.config import RadarConfig
from radarsim.models import SignalCube


def _repetition_time(
    times: NDArray[np.float64], interval: float, width: float,
    slow_count: int, frame_count: int, sample_rate: float,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Map absolute queries to a finite train of rectangular chirps.

    Nominal boundaries use n * interval, the frame/slow-time sum in build_tx,
    or integer sample ticks / sample_rate when the timing fits that clock.
    Only exact equality to these floats identifies a boundary. An input that
    rounds to the same float as a nominal boundary is indistinguishable from
    that boundary and takes its half-open envelope semantics, even if it was
    obtained by shifting another representation of that boundary. All other
    inputs use n * interval as the local-time origin. Finite support takes
    precedence over aliases; no elapsed-time tolerance is used.
    """
    support = frame_count * slow_count * interval
    valid = (times >= 0) & (times < support)
    bounded = np.where(valid, times, 0.0)
    nearest = np.rint(bounded / interval)
    frame_span = slow_count * interval
    nominal = nearest * interval
    framed = (nearest // slow_count) * frame_span + (nearest % slow_count) * interval
    at_start = (bounded == nominal) | (bounded == framed)
    interval_ticks = round(interval * sample_rate)
    width_ticks = round(width * sample_rate)
    on_clock = interval == interval_ticks / sample_rate and width == width_ticks / sample_rate
    if on_clock:
        at_start |= bounded == nearest * interval_ticks / sample_rate
    repetition = np.where((bounded < nominal) & ~at_start, nearest - 1, nearest)
    start = repetition * interval
    frame_start = (repetition // slow_count) * frame_span + (repetition % slow_count) * interval
    local = np.where(at_start, 0.0, bounded - start)
    at_end = (bounded == start + width) | (bounded == frame_start + width)
    if on_clock:
        at_end |= bounded == (repetition * interval_ticks + width_ticks) / sample_rate
    active = valid & ~at_end & (local >= 0) & (local < width)
    return local, active


class Waveform(Protocol):
    sample_rate_hz: float
    slow_time_count: int
    frame_count: int
    repetition_interval_s: float

    def build_tx(self, radar: RadarConfig) -> SignalCube: ...

    def sample_at(self, times_s: NDArray[np.float64]) -> NDArray[np.complex128]: ...

    def reference(self) -> NDArray[np.complex128]: ...
