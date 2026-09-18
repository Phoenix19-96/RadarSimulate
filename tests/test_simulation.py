import inspect
from dataclasses import replace

import numpy as np
import pytest
from radarsim.channel import MonostaticChannel

from radarsim.config import CFARConfig, NoiseConfig, ProcessingConfig, RadarConfig
from radarsim.models import SignalCube
from radarsim.scene import CartesianWaypoint, Target, Trajectory
from radarsim.simulation import ExperimentConfig, run_processing, run_simulation
from radarsim.waveforms.fmcw import FMCWConfig, FMCWWaveform
from radarsim.waveforms.lfm import LFMConfig


def target_at(range_m, speed_mps=0):
    trajectory = Trajectory.from_cartesian([
        CartesianWaypoint(0, (range_m, 0, 0)),
        CartesianWaypoint(1, (range_m + speed_mps, 0, 0)),
    ])
    return Target(f"t{range_m}", f"target-{range_m}", 20, trajectory)


def test_fmcw_simulation_preserves_axes_and_returns_finite_results():
    config = ExperimentConfig(
        radar=RadarConfig(carrier_hz=10e9, transmit_power_w=1e6),
        waveform=FMCWConfig(20e6, 40e-6, 50e-6, 10e6, 16, 1),
        targets=(target_at(30, 2),),
        noise=NoiseConfig(0),
        processing=ProcessingConfig(
            range_fft_size=512, doppler_fft_size=16,
            cfar=CFARConfig((2, 2), (1, 1), 1e-3),
        ),
    )
    result = run_simulation(config)
    assert result.tx.data.shape[:3] == (1, 16, 1)
    assert result.rx.data.shape[:3] == (1, 16, 1)
    assert result.range_doppler.power_w.shape[:3] == (1, 1, 16)
    assert np.isfinite(result.range_doppler.power_w).all()
    assert len(result.truth) == 1


def test_lfm_simulation_runs_the_same_result_contract():
    config = ExperimentConfig(
        RadarConfig(carrier_hz=10e9, transmit_power_w=1e6),
        LFMConfig(2e6, 8e-6, 80e-6, 4e6, 16, 1),
        (target_at(300, -1),),
        NoiseConfig(0),
        ProcessingConfig(
            doppler_fft_size=16,
            cfar=CFARConfig((2, 2), (1, 1), 1e-3),
        ),
    )
    result = run_simulation(config)
    assert result.rx.dimensions == ("frame", "slow_time", "rx", "fast_time")
    assert result.range_doppler.spectrum.ndim == 4


def test_truth_outside_unambiguous_range_emits_warning():
    config = ExperimentConfig(
        RadarConfig(transmit_power_w=1e6),
        FMCWConfig(20e6, 40e-6, 50e-6, 1e6, 8, 1),
        (target_at(10_000),),
        NoiseConfig(0),
        ProcessingConfig(
            doppler_fft_size=8,
            cfar=CFARConfig((2, 2), (1, 1), 1e-3),
        ),
    )
    with pytest.warns(RuntimeWarning, match="range"):
        run_simulation(config)


def test_processing_boundary_does_not_accept_target_truth():
    assert tuple(inspect.signature(run_processing).parameters) == (
        "rx", "tx", "waveform", "radar", "processing",
    )


def test_trajectory_coverage_is_checked_before_tx_build(monkeypatch):
    short_trajectory_target = Target(
        "short", "short", 20, Trajectory.from_cartesian([
            CartesianWaypoint(0, (30, 0, 0)),
            CartesianWaypoint(1e-6, (30, 0, 0)),
        ]),
    )
    config = ExperimentConfig(
        RadarConfig(transmit_power_w=1e6),
        FMCWConfig(2e6, 10e-6, 20e-6, 4e6, 16, 1),
        (short_trajectory_target,),
        NoiseConfig(0),
        ProcessingConfig(
            cfar=CFARConfig((2, 2), (1, 1), 1e-3),
        ),
    )
    monkeypatch.setattr(
        FMCWWaveform, "build_tx",
        lambda *_: (_ for _ in ()).throw(AssertionError("Tx must not be built")),
    )
    with pytest.raises(ValueError, match="trajectory coverage"):
        run_simulation(config)


def test_simulation_result_rejects_frontend_inconsistent_with_rx():
    result = run_simulation(ExperimentConfig(
        RadarConfig(transmit_power_w=1e6),
        FMCWConfig(2e6, 10e-6, 20e-6, 4e6, 16, 1),
        (target_at(30),),
        NoiseConfig(0),
        ProcessingConfig(cfar=CFARConfig((2, 2), (1, 1), 1e-3)),
    ))
    mismatched_frontend = SignalCube(
        result.rx.data[..., :-1], result.rx.dimensions,
        result.rx.fast_time_s[:-1], result.rx.slow_time_s,
        result.rx.sample_times_s[..., :-1], result.rx.metadata,
    )
    with pytest.raises(ValueError, match="frontend"):
        replace(result, frontend=mismatched_frontend)


def test_lfm_simulation_result_requires_its_received_intermediate():
    result = run_simulation(ExperimentConfig(
        RadarConfig(transmit_power_w=1e6),
        LFMConfig(2e6, 8e-6, 80e-6, 4e6, 16, 1),
        (target_at(300),),
        NoiseConfig(0),
        ProcessingConfig(cfar=CFARConfig((2, 2), (1, 1), 1e-3)),
    ))
    altered_received = SignalCube(
        result.rx.data * 2, result.rx.dimensions, result.rx.fast_time_s,
        result.rx.slow_time_s, result.rx.sample_times_s, result.rx.metadata,
    )
    altered_frontend = replace(result.frontend, received=altered_received)
    with pytest.raises(ValueError, match="frontend"):
        replace(result, frontend=altered_frontend)

def test_fmcw_processing_rejects_tx_metadata_from_a_different_waveform():
    radar = RadarConfig(transmit_power_w=1e6)
    selected_config = FMCWConfig(2e6, 10e-6, 20e-6, 4e6, 16, 1)
    selected_waveform = FMCWWaveform(selected_config)
    expected_tx = selected_waveform.build_tx(radar)
    rx = MonostaticChannel().propagate(
        expected_tx, selected_waveform, radar, (target_at(30),), NoiseConfig(0),
    )
    mismatched_tx = FMCWWaveform(FMCWConfig(
        4e6, 10e-6, 20e-6, 4e6, 16, 1,
    )).build_tx(radar)
    with pytest.raises(ValueError, match="Tx metadata bandwidth_hz"):
        run_processing(
            rx, mismatched_tx, selected_waveform, radar,
            ProcessingConfig(cfar=CFARConfig((2, 2), (1, 1), 1e-3)),
        )


def test_duplicate_target_ids_are_rejected_before_tx_build(monkeypatch):
    duplicate_target = Target("t30", "duplicate", 20, Trajectory.from_cartesian([
        CartesianWaypoint(0, (40, 0, 0)),
        CartesianWaypoint(1, (40, 0, 0)),
    ]))
    config = ExperimentConfig(
        RadarConfig(transmit_power_w=1e6),
        FMCWConfig(2e6, 10e-6, 20e-6, 4e6, 16, 1),
        (target_at(30), duplicate_target),
        NoiseConfig(0),
        ProcessingConfig(cfar=CFARConfig((2, 2), (1, 1), 1e-3)),
    )
    monkeypatch.setattr(
        FMCWWaveform, "build_tx",
        lambda *_: (_ for _ in ()).throw(AssertionError("Tx must not be built")),
    )
    with pytest.raises(ValueError, match="t30"):
        run_simulation(config)


def test_unrepresentable_thermal_noise_is_rejected_before_tx_build(monkeypatch):
    config = ExperimentConfig(
        RadarConfig(transmit_power_w=1e6),
        FMCWConfig(2e6, 10e-6, 20e-6, 4e6, 16, 1),
        (target_at(30),),
        NoiseConfig(
            noise_power_w=None, temperature_k=1e308, noise_figure_db=308,
            bandwidth_hz=1e308,
        ),
        ProcessingConfig(cfar=CFARConfig((2, 2), (1, 1), 1e-3)),
    )
    monkeypatch.setattr(
        FMCWWaveform, "build_tx",
        lambda *_: (_ for _ in ()).throw(AssertionError("Tx must not be built")),
    )
    with pytest.raises(ValueError, match="noise_figure_db"):
        run_simulation(config)
