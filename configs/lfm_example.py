"""Editable pulsed-LFM experiment parameters used by ``examples/run_lfm.py``."""

import os
from pathlib import Path

from radarsim.config import CFARConfig, NoiseConfig, OutputConfig, ProcessingConfig, RadarConfig
from radarsim.scene import CartesianWaypoint, PolarWaypoint, Target, Trajectory
from radarsim.simulation import ExperimentConfig
from radarsim.waveforms.lfm import LFMConfig


# Edit this object directly to change the pulsed-LFM experiment; no GUI or CLI is used.
CONFIG = ExperimentConfig(
    radar=RadarConfig(carrier_hz=10e9, transmit_power_w=1e9, tx_gain_db=25, rx_gain_db=25),
    waveform=LFMConfig(5e6, 10e-6, 100e-6, 10e6, 64, 1),
    targets=(
        Target("lfm-1", "receding", 15, Trajectory.from_cartesian([
            CartesianWaypoint(0, (1200, 100, 20)),
            CartesianWaypoint(1, (1210, 100, 20)),
        ])),
        Target("lfm-2", "approaching", 10, Trajectory.from_polar([
            PolarWaypoint(0, 2500, 15, 5),
            PolarWaypoint(1, 2485, 15, 5),
        ])),
    ),
    noise=NoiseConfig(noise_power_w=1e-15, seed=20260915),
    processing=ProcessingConfig(
        doppler_fft_size=64, cfar=CFARConfig((6, 4), (2, 1), 1e-4),
    ),
)

# Set RADARSIM_OUTPUT_ROOT for a one-off destination without altering CONFIG.
OUTPUT = OutputConfig("lfm_example", Path(os.environ.get("RADARSIM_OUTPUT_ROOT", "outputs")))
