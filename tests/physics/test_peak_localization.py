import numpy as np

from radarsim.config import NoiseConfig, ProcessingConfig, RadarConfig
from radarsim.scene import CartesianWaypoint, Target, Trajectory
from radarsim.simulation import ExperimentConfig, run_simulation
from radarsim.waveforms.fmcw import FMCWConfig
from radarsim.waveforms.lfm import LFMConfig


def moving_target(initial_range, speed):
    return Target("single", "single", 20, Trajectory.from_cartesian([
        CartesianWaypoint(0, (initial_range, 0, 0)),
        CartesianWaypoint(1, (initial_range + speed, 0, 0)),
    ]))


def assert_within_one_resolution_cell(result):
    doppler_index, range_index = np.unravel_index(
        np.argmax(result.range_doppler.power_w[0, 0]),
        result.range_doppler.power_w[0, 0].shape,
    )
    truth = result.truth[0]
    measured_range = result.range_doppler.range_m[range_index]
    measured_velocity = result.range_doppler.velocity_mps[doppler_index]
    assert abs(measured_range - truth.range_m) <= result.range_doppler.range_resolution_m
    assert abs(measured_velocity - truth.radial_velocity_mps) <= (
        result.range_doppler.velocity_resolution_mps
    )


def test_fmcw_no_noise_single_target_peak_is_within_one_cell():
    result = run_simulation(ExperimentConfig(
        RadarConfig(carrier_hz=10e9, transmit_power_w=1e9),
        FMCWConfig(50e6, 40e-6, 50e-6, 10e6, 64, 1),
        (moving_target(150, 4),),
        NoiseConfig(0),
        ProcessingConfig(range_fft_size=512, doppler_fft_size=64),
    ))
    assert_within_one_resolution_cell(result)


def test_lfm_no_noise_single_target_peak_is_within_one_cell():
    result = run_simulation(ExperimentConfig(
        RadarConfig(carrier_hz=10e9, transmit_power_w=1e9),
        LFMConfig(5e6, 10e-6, 100e-6, 10e6, 64, 1),
        (moving_target(1200, 3),),
        NoiseConfig(0),
        ProcessingConfig(doppler_fft_size=64),
    ))
    assert_within_one_resolution_cell(result)
