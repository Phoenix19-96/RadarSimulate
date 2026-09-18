"""Editable FMCW experiment parameters used by ``examples/run_fmcw.py``."""

import os
from pathlib import Path

from radarsim.config import CFARConfig, NoiseConfig, OutputConfig, ProcessingConfig, RadarConfig
from radarsim.scene import CartesianWaypoint, PolarWaypoint, Target, Trajectory
from radarsim.simulation import ExperimentConfig
from radarsim.waveforms.fmcw import FMCWConfig


# Edit this object directly to change the FMCW experiment; no GUI or CLI is used.
CONFIG = ExperimentConfig(
    radar=RadarConfig(carrier_hz=10e9, transmit_power_w=1e9, tx_gain_db=20, rx_gain_db=20),
    waveform=FMCWConfig(20e6, 40e-6, 50e-6, 5e6, 64, 1),
    targets=(
        Target("fmcw-1", "receding", 10, Trajectory.from_cartesian([
            CartesianWaypoint(0, (120, 20, 5)),
            CartesianWaypoint(1, (128, 20, 5)),
        ])),
        Target("fmcw-2", "approaching", 5, Trajectory.from_polar([
            PolarWaypoint(0, 250, -20, 3),
            PolarWaypoint(1, 240, -20, 3),
        ])),
    ),
    noise=NoiseConfig(noise_power_w=1e-14, seed=20260915),
    processing=ProcessingConfig(
        range_fft_size=512, doppler_fft_size=64,
        cfar=CFARConfig((6, 4), (2, 1), 1e-4),
    ),
)

# Set RADARSIM_OUTPUT_ROOT for a one-off destination without altering CONFIG.
OUTPUT = OutputConfig("fmcw_example", Path(os.environ.get("RADARSIM_OUTPUT_ROOT", "outputs")))
