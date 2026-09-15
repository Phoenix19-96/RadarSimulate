from dataclasses import dataclass
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
