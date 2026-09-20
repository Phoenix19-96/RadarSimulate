"""Behavioral tests for explicit, reproducible simulation artifacts."""

import json

import numpy as np
import pytest

from radarsim.config import (
    CFARConfig,
    NoiseConfig,
    OutputConfig,
    ProcessingConfig,
    RadarConfig,
)
from radarsim.output import save_simulation
from radarsim.scene import CartesianWaypoint, Target, Trajectory
from radarsim.simulation import ExperimentConfig, run_simulation
from radarsim.waveforms.fmcw import FMCWConfig


def small_result():
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
            range_fft_size=128,
            doppler_fft_size=8,
            cfar=CFARConfig((2, 2), (1, 1), 1e-3),
        ),
    )
    return config, run_simulation(config)


def test_save_simulation_writes_complete_reproducible_manifest(tmp_path):
    """Break caught: omitting or corrupting a persisted result product."""
    config, result = small_result()

    directory = save_simulation(
        result, config, OutputConfig("smoke", tmp_path, save_raw_data=True),
    )

    assert directory == tmp_path / "smoke"
    assert {path.name for path in directory.iterdir()} == {
        "config.json",
        "truth.csv",
        "detections.csv",
        "raw_data.npz",
        "range_profile.png",
        "range_doppler.png",
        "cfar_map.png",
    }
    snapshot = json.loads((directory / "config.json").read_text("utf-8"))
    assert snapshot["waveform"]["bandwidth_hz"] == 10e6
    assert snapshot["output"] == {
        "experiment_name": "smoke",
        "output_root": str(tmp_path),
        "save_raw_data": True,
    }
    with np.load(directory / "raw_data.npz") as arrays:
        assert arrays.files == [
            "tx", "rx", "range_doppler", "range_m", "velocity_mps",
        ]
        assert arrays["rx"].shape == result.rx.data.shape
        assert arrays["range_doppler"].shape == result.range_doppler.spectrum.shape
        np.testing.assert_array_equal(arrays["range_m"], result.range_doppler.range_m)
    for name in ("range_profile.png", "range_doppler.png", "cfar_map.png"):
        assert (directory / name).stat().st_size > 0


def test_save_simulation_can_skip_raw_data(tmp_path):
    """Break caught: raw archive is written when callers opt out."""
    config, result = small_result()

    directory = save_simulation(
        result, config, OutputConfig("no-raw", tmp_path, save_raw_data=False),
    )

    assert not (directory / "raw_data.npz").exists()
    assert (directory / "range_doppler.png").stat().st_size > 0


def test_save_simulation_refuses_to_overwrite_an_existing_experiment(tmp_path):
    """Break caught: a repeated experiment name silently replaces artifacts."""
    config, result = small_result()
    output = OutputConfig("repeat", tmp_path)
    save_simulation(result, config, output)

    with pytest.raises(FileExistsError, match="repeat"):
        save_simulation(result, config, output)


def test_run_simulation_does_not_write_artifacts_by_default(tmp_path, monkeypatch):
    """Break caught: pure simulation gains an implicit filesystem side effect."""
    config, _ = small_result()
    monkeypatch.chdir(tmp_path)

    run_simulation(config)

    assert list(tmp_path.iterdir()) == []
