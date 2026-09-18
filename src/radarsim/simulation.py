"""End-to-end assembly of the explicit radar signal chain."""

from dataclasses import dataclass
import warnings

import numpy as np

from .channel import MonostaticChannel
from .config import NoiseConfig, ProcessingConfig, RadarConfig
from .frontend import dechirp_fmcw, prepare_lfm
from .geometry import cartesian_to_polar, radial_velocity
from .models import SimulationResult, TruthRecord
from .processing.cfar import ca_cfar_2d, extract_detections
from .processing.range_doppler import process_fmcw, process_lfm
from .scene import Target
from .waveforms.fmcw import FMCWConfig, FMCWWaveform
from .waveforms.lfm import LFMConfig, LFMWaveform


@dataclass(frozen=True)
class ExperimentConfig:
    radar: RadarConfig
    waveform: FMCWConfig | LFMConfig
    targets: tuple[Target, ...]
    noise: NoiseConfig
    processing: ProcessingConfig

    def __post_init__(self) -> None:
        if not isinstance(self.radar, RadarConfig):
            raise TypeError("radar must be a RadarConfig")
        if not isinstance(self.waveform, (FMCWConfig, LFMConfig)):
            raise TypeError("waveform must be FMCWConfig or LFMConfig")
        if type(self.targets) is not tuple or not all(
            isinstance(target, Target) for target in self.targets
        ):
            raise TypeError("targets must be a tuple of Target values")
        if not isinstance(self.noise, NoiseConfig):
            raise TypeError("noise must be a NoiseConfig")
        if not isinstance(self.processing, ProcessingConfig):
            raise TypeError("processing must be a ProcessingConfig")


def _waveform(config: FMCWConfig | LFMConfig) -> FMCWWaveform | LFMWaveform:
    if isinstance(config, FMCWConfig):
        return FMCWWaveform(config)
    if isinstance(config, LFMConfig):
        return LFMWaveform(config)
    raise TypeError("waveform must be FMCWConfig or LFMConfig")


def _validate_processing(
    waveform: FMCWWaveform | LFMWaveform, processing: ProcessingConfig,
) -> None:
    if processing.range_window not in {"hann", "hamming", "blackman", "boxcar"}:
        raise ValueError("range_window must be a supported window")
    if processing.doppler_window not in {"hann", "hamming", "blackman", "boxcar"}:
        raise ValueError("doppler_window must be a supported window")
    if isinstance(waveform, FMCWWaveform):
        fast_count = round(
            waveform.config.chirp_duration_s * waveform.sample_rate_hz
        )
        range_fft_size = (
            fast_count if processing.range_fft_size is None
            else processing.range_fft_size
        )
        range_bins = range_fft_size // 2 + 1
    else:
        fast_count = round(
            waveform.config.pulse_repetition_interval_s * waveform.sample_rate_hz
        )
        range_fft_size = fast_count
        range_bins = fast_count
    doppler_fft_size = (
        waveform.slow_time_count if processing.doppler_fft_size is None
        else processing.doppler_fft_size
    )
    for name, value, minimum in (
        ("range_fft_size", range_fft_size, fast_count),
        ("doppler_fft_size", doppler_fft_size, waveform.slow_time_count),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
    train_doppler, train_range = processing.cfar.training_cells
    guard_doppler, guard_range = processing.cfar.guard_cells
    if (doppler_fft_size <= 2 * (train_doppler + guard_doppler)
            or range_bins <= 2 * (train_range + guard_range)):
        raise ValueError("CFAR window does not fit the range-Doppler map")


def _validate_trajectory_coverage(
    config: ExperimentConfig, waveform: FMCWWaveform | LFMWaveform,
) -> None:
    if isinstance(waveform, FMCWWaveform):
        fast_count = round(
            waveform.config.chirp_duration_s * waveform.sample_rate_hz
        )
    else:
        fast_count = round(
            waveform.config.pulse_repetition_interval_s * waveform.sample_rate_hz
        )
    frame_span = waveform.slow_time_count * waveform.repetition_interval_s
    sample_end_s = (
        (waveform.frame_count - 1) * frame_span
        + (waveform.slow_time_count - 1) * waveform.repetition_interval_s
        + (fast_count - 1) / waveform.sample_rate_hz
    )
    required_end_s = max(
        sample_end_s,
        (waveform.frame_count - 0.5) * frame_span,
    )
    for target in config.targets:
        if (target.trajectory.times_s[0] > 0.0
                or target.trajectory.times_s[-1] < required_end_s):
            raise ValueError(
                "trajectory coverage must include simulation samples and truth times"
            )


def _validate_fmcw_metadata(rx, tx, waveform: FMCWWaveform) -> None:
    expected = {
        "waveform_kind": "fmcw",
        "sample_rate_hz": waveform.sample_rate_hz,
        "bandwidth_hz": waveform.config.bandwidth_hz,
        "slope_hz_per_s": waveform.config.slope_hz_per_s,
        "chirp_duration_s": waveform.config.chirp_duration_s,
        "repetition_interval_s": waveform.repetition_interval_s,
        "chirps_per_frame": waveform.slow_time_count,
        "frame_count": waveform.frame_count,
    }
    for label, cube in (("Rx", rx), ("Tx", tx)):
        for name, expected_value in expected.items():
            supplied = cube.metadata.get(name)
            if name == "waveform_kind":
                matches = supplied == expected_value
            else:
                try:
                    matches = (np.isscalar(supplied) and np.isfinite(supplied)
                               and np.isclose(supplied, expected_value,
                                              rtol=1e-12, atol=1e-15))
                except TypeError:
                    matches = False
            if not matches:
                raise ValueError(f"{label} metadata {name} does not match waveform")


def _validate_unique_target_ids(targets: tuple[Target, ...]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for target in targets:
        if target.target_id in seen:
            duplicates.add(target.target_id)
        seen.add(target.target_id)
    if duplicates:
        raise ValueError("duplicate target IDs: " + ", ".join(sorted(duplicates)))


def _validate_noise_power(noise: NoiseConfig) -> None:
    parameter = "noise_power_w" if noise.noise_power_w is not None else "noise_figure_db"
    try:
        power_w = noise.resolved_power_w()
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError(f"{parameter} produces unrepresentable noise power") from exc
    if not np.isfinite(power_w) or power_w < 0:
        raise ValueError(f"{parameter} must resolve to finite non-negative noise power")



def run_processing(rx, tx, waveform, radar, processing):
    """Process receiver samples without access to scene or target truth."""
    if isinstance(waveform, FMCWWaveform):
        _validate_fmcw_metadata(rx, tx, waveform)
        frontend = dechirp_fmcw(rx, tx)
        range_doppler = process_fmcw(frontend, radar, waveform.config, processing)
    elif isinstance(waveform, LFMWaveform):
        frontend = prepare_lfm(rx, waveform)
        range_doppler = process_lfm(frontend, radar, waveform.config, processing)
    else:
        raise TypeError("waveform must be an FMCWWaveform or LFMWaveform")
    cfar = tuple(
        tuple(
            ca_cfar_2d(range_doppler.power_w[frame, channel], processing.cfar)
            for channel in range(range_doppler.power_w.shape[1])
        )
        for frame in range(range_doppler.power_w.shape[0])
    )
    return frontend, range_doppler, cfar, extract_detections(range_doppler, cfar)


def _truth_records(config: ExperimentConfig, tx) -> tuple[TruthRecord, ...]:
    records: list[TruthRecord] = []
    frame_span = tx.sample_times_s.shape[1] * float(
        tx.metadata["repetition_interval_s"]
    )
    radar_position = np.asarray(config.radar.position_m, dtype=float)
    for frame in range(tx.data.shape[0]):
        time_s = frame * frame_span + frame_span / 2
        for target in config.targets:
            state = target.trajectory.state_at(time_s)
            relative = state.position_m - radar_position
            range_m, azimuth_deg, elevation_deg = cartesian_to_polar(relative)
            records.append(TruthRecord(
                frame, target.target_id, time_s, *state.position_m, range_m,
                radial_velocity(relative, state.velocity_mps), azimuth_deg,
                elevation_deg, target.rcs_dbsm,
            ))
    return tuple(records)


def _warn_for_ambiguity(truth: tuple[TruthRecord, ...], range_doppler) -> None:
    if any(record.range_m > range_doppler.max_unambiguous_range_m for record in truth):
        warnings.warn(
            "target range exceeds maximum unambiguous range", RuntimeWarning,
            stacklevel=2,
        )
    if any(
        abs(record.radial_velocity_mps)
        > range_doppler.max_unambiguous_velocity_mps
        for record in truth
    ):
        warnings.warn(
            "target speed exceeds maximum unambiguous velocity", RuntimeWarning,
            stacklevel=2,
        )


def run_simulation(config: ExperimentConfig) -> SimulationResult:
    """Run waveform generation, propagation, frontend, processing, and CFAR."""
    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig")
    waveform = _waveform(config.waveform)
    _validate_processing(waveform, config.processing)
    _validate_trajectory_coverage(config, waveform)
    _validate_unique_target_ids(config.targets)
    _validate_noise_power(config.noise)
    tx = waveform.build_tx(config.radar)
    rx = MonostaticChannel().propagate(
        tx, waveform, config.radar, config.targets, config.noise,
    )
    frontend, range_doppler, cfar, detections = run_processing(
        rx, tx, waveform, config.radar, config.processing,
    )
    truth = _truth_records(config, tx)
    _warn_for_ambiguity(truth, range_doppler)
    return SimulationResult(
        tx, rx, frontend, range_doppler, cfar, detections, truth,
    )


def run_and_save(config: ExperimentConfig, output):
    """Run the pure simulation, then explicitly persist its artifacts."""
    from .output import save_simulation

    result = run_simulation(config)
    return result, save_simulation(result, config, output)
