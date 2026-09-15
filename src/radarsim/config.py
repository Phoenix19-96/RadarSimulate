from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Literal

from .constants import BOLTZMANN_J_PER_K, C_MPS


@dataclass(frozen=True)
class RadarConfig:
    position_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    boresight_az_deg: float = 0.0
    boresight_el_deg: float = 0.0
    carrier_hz: float = 10e9
    transmit_power_w: float = 1.0
    tx_gain_db: float = 0.0
    rx_gain_db: float = 0.0
    num_tx: int = 1
    num_rx: int = 1

    def __post_init__(self) -> None:
        if type(self.position_m) is not tuple or len(self.position_m) != 3 or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in self.position_m):
            raise ValueError("position_m must contain three finite scalars")
        for name, value in (("boresight_az_deg", self.boresight_az_deg), ("boresight_el_deg", self.boresight_el_deg), ("carrier_hz", self.carrier_hz), ("transmit_power_w", self.transmit_power_w), ("tx_gain_db", self.tx_gain_db), ("rx_gain_db", self.rx_gain_db)):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.carrier_hz <= 0:
            raise ValueError("carrier_hz must be positive")
        if self.transmit_power_w < 0:
            raise ValueError("transmit_power_w must be non-negative")
        if type(self.num_tx) is not int or self.num_tx <= 0 or self.num_tx != 1:
            raise ValueError("num_tx must equal 1 in the initial release")
        if type(self.num_rx) is not int or self.num_rx <= 0 or self.num_rx != 1:
            raise ValueError("num_rx must equal 1 in the initial release")

    @property
    def wavelength_m(self) -> float:
        return C_MPS / self.carrier_hz


@dataclass(frozen=True)
class NoiseConfig:
    noise_power_w: float | None = 0.0
    temperature_k: float | None = None
    noise_figure_db: float | None = None
    bandwidth_hz: float | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        for name, value in (("noise_power_w", self.noise_power_w), ("temperature_k", self.temperature_k), ("noise_figure_db", self.noise_figure_db), ("bandwidth_hz", self.bandwidth_hz)):
            if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)):
                raise ValueError(f"{name} must be finite")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        thermal = (self.temperature_k, self.noise_figure_db, self.bandwidth_hz)
        if self.noise_power_w is not None and any(v is not None for v in thermal):
            raise ValueError("direct and thermal noise settings are mutually exclusive")
        if self.noise_power_w is None and any(v is None for v in thermal):
            raise ValueError("all thermal noise settings are required")
        if self.noise_power_w is not None and self.noise_power_w < 0:
            raise ValueError("noise_power_w must be non-negative")
        if self.temperature_k is not None and self.temperature_k <= 0:
            raise ValueError("temperature_k must be positive")
        if self.bandwidth_hz is not None and self.bandwidth_hz <= 0:
            raise ValueError("bandwidth_hz must be positive")

    def resolved_power_w(self) -> float:
        if self.noise_power_w is not None:
            return self.noise_power_w
        return BOLTZMANN_J_PER_K * self.temperature_k * 10 ** (self.noise_figure_db / 10) * self.bandwidth_hz


@dataclass(frozen=True)
class CFARConfig:
    training_cells: tuple[int, int] = (8, 4)
    guard_cells: tuple[int, int] = (2, 1)
    false_alarm_rate: float = 1e-5

    def __post_init__(self) -> None:
        for name, cells in (("training_cells", self.training_cells), ("guard_cells", self.guard_cells)):
            if type(cells) is not tuple or len(cells) != 2 or any(type(v) is not int or v < 0 for v in cells):
                raise ValueError(f"{name} must contain exactly two non-negative integers")
        if self.training_cells == (0, 0):
            raise ValueError("CFAR requires training cells")
        if not 0 < self.false_alarm_rate < 1:
            raise ValueError("false_alarm_rate must be between 0 and 1")


@dataclass(frozen=True)
class ProcessingConfig:
    range_window: Literal["hann", "hamming", "blackman", "boxcar"] = "hann"
    doppler_window: Literal["hann", "hamming", "blackman", "boxcar"] = "hann"
    range_fft_size: int | None = None
    doppler_fft_size: int | None = None
    cfar: CFARConfig = field(default_factory=CFARConfig)


@dataclass(frozen=True)
class OutputConfig:
    experiment_name: str
    output_root: Path = Path("outputs")
    save_raw_data: bool = True
