"""Tests for opt-in simulation artifact integration."""

from radarsim.config import CFARConfig, NoiseConfig, OutputConfig, ProcessingConfig, RadarConfig
from radarsim.scene import CartesianWaypoint, Target, Trajectory
from radarsim.simulation import ExperimentConfig, run_and_save
from radarsim.waveforms.fmcw import FMCWConfig


def test_run_and_save_returns_result_and_explicit_artifact_directory(tmp_path):
    """Break caught: explicit integration loses its result or artifact directory."""
    trajectory = Trajectory.from_cartesian([
        CartesianWaypoint(0, (30, 0, 0)),
        CartesianWaypoint(1, (31, 0, 0)),
    ])
    config = ExperimentConfig(
        RadarConfig(transmit_power_w=1e6),
        FMCWConfig(10e6, 20e-6, 25e-6, 5e6, 8, 1),
        (Target("t1", "target", 10, trajectory),),
        NoiseConfig(0),
        ProcessingConfig(
            range_fft_size=128, doppler_fft_size=8,
            cfar=CFARConfig((2, 2), (1, 1), 1e-3),
        ),
    )

    result, directory = run_and_save(config, OutputConfig("integrated", tmp_path))

    assert result.tx.dimensions == ("frame", "slow_time", "tx", "fast_time")
    assert directory == tmp_path / "integrated"
    assert (directory / "config.json").exists()
