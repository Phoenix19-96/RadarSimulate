"""Regressions for delayed echoes that cross waveform repetition boundaries."""

from dataclasses import replace
import warnings

import numpy as np
import pytest

from radarsim.config import CFARConfig, NoiseConfig, ProcessingConfig, RadarConfig
from radarsim.scene import CartesianWaypoint, Target, Trajectory
from radarsim.simulation import ExperimentConfig, run_simulation
from radarsim.waveforms.fmcw import FMCWConfig, FMCWWaveform
from radarsim.waveforms.lfm import LFMConfig


def experiment(waveform, distance, speed=0):
    target = Target("boundary", "boundary", 20, Trajectory.from_cartesian([
        CartesianWaypoint(0, (distance, 0, 0)),
        CartesianWaypoint(1, (distance + speed, 0, 0)),
    ]))
    return ExperimentConfig(
        RadarConfig(carrier_hz=10e9, transmit_power_w=1e9), waveform,
        (target,), NoiseConfig(0), ProcessingConfig(cfar=CFARConfig((2, 2), (1, 1))),
    )


@pytest.mark.parametrize("speed", [-3.0, 0.0, 3.0])
@pytest.mark.parametrize("distance", [1200.0, 14900.0])
def test_lfm_continuous_echo_peak_within_physical_resolution(distance, speed):
    """Independent PRI filtering mistakes a previous-pulse tail for near range."""
    result = run_simulation(experiment(
        LFMConfig(5e6, 10e-6, 100e-6, 10e6, 32, 2), distance, speed,
    ))
    for frame, truth in enumerate(result.truth):
        rd = result.range_doppler
        d, r = np.unravel_index(np.argmax(rd.power_w[frame, 0]), rd.power_w[frame, 0].shape)
        assert abs(rd.range_m[r] - truth.range_m) <= 29.9792458
        assert abs(rd.velocity_mps[d] - speed) <= rd.velocity_resolution_mps
    assert rd.max_unambiguous_range_m == pytest.approx(14989.6229)


@pytest.mark.parametrize("speed", [-60.0, 0.0, 60.0])
@pytest.mark.parametrize("distance", [5.0, 300.0, 1400.0, 1450.0, 1490.0])
def test_fmcw_both_chirp_branches_peak_within_physical_resolution(distance, speed):
    """Discarding positive previous-chirp beats maps far full-duty echoes to zero."""
    result = run_simulation(experiment(
        FMCWConfig(1e6, 10e-6, 10e-6, 10e6, 32, 2), distance, speed,
    ))
    for frame, truth in enumerate(result.truth):
        rd = result.range_doppler
        d, r = np.unravel_index(np.argmax(rd.power_w[frame, 0]), rd.power_w[frame, 0].shape)
        assert abs(rd.range_m[r] - truth.range_m) <= 149.896229
        assert abs(rd.velocity_mps[d] - speed) <= rd.velocity_resolution_mps
    assert rd.max_unambiguous_range_m == pytest.approx(1498.96229)
    assert np.all(rd.range_m < rd.max_unambiguous_range_m)


def test_fmcw_repetition_alias_warns_before_building_tx(monkeypatch):
    """An IF-only range limit silently accepts the physically ambiguous 2000 m echo."""
    original = FMCWWaveform.build_tx
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        def check_warning_before_build(waveform, radar):
            assert any("range" in str(item.message) for item in caught)
            return original(waveform, radar)

        monkeypatch.setattr(FMCWWaveform, "build_tx", check_warning_before_build)
        result = run_simulation(experiment(
            FMCWConfig(1e6, 10e-6, 10e-6, 10e6, 32), 2000,
        ))
    assert result.range_doppler.max_unambiguous_range_m == pytest.approx(1498.96229)


@pytest.mark.parametrize("distance", [1400.0, 1480.0])
def test_gapped_fmcw_uses_the_observed_partial_chirp_support(distance):
    """Unequal template support must not prefer unobserved echo samples."""
    result = run_simulation(experiment(
        FMCWConfig(1e6, 10e-6, 20e-6, 10e6, 32), distance,
    ))
    rd = result.range_doppler
    _, r = np.unravel_index(np.argmax(rd.power_w[0, 0]), rd.power_w[0, 0].shape)
    assert abs(rd.range_m[r] - distance) <= 149.896229
    assert rd.max_unambiguous_range_m == pytest.approx(1483.9726671)


def test_gapped_fmcw_warns_before_tx_for_echo_after_last_acquired_sample(monkeypatch):
    """A 9.94 us echo cannot be observed when the final sample is at 9.9 us."""
    original = FMCWWaveform.build_tx
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        def check_warning_before_build(waveform, radar):
            assert any("range" in str(item.message) for item in caught)
            return original(waveform, radar)

        monkeypatch.setattr(FMCWWaveform, "build_tx", check_warning_before_build)
        run_simulation(experiment(FMCWConfig(1e6, 10e-6, 20e-6, 10e6, 32), 1490))


@pytest.mark.parametrize("fft_size", [0, -1, 3.5, True])
def test_invalid_fmcw_range_fft_size_reports_configuration_error(fft_size):
    config = experiment(FMCWConfig(1e6, 10e-6, 10e-6, 10e6, 32), 300)
    config = replace(config, processing=replace(config.processing, range_fft_size=fft_size))
    with pytest.raises(ValueError, match="range_fft_size"):
        run_simulation(config)


@pytest.mark.parametrize("fft_size", [999, 1024])
def test_lfm_rejects_inapplicable_range_fft_before_building_tx(fft_size):
    """LFM silently ignored explicit range FFT requests of any size."""
    config = experiment(LFMConfig(5e6, 10e-6, 100e-6, 10e6, 32), 1200)
    config = replace(config, processing=replace(config.processing, range_fft_size=fft_size))
    with pytest.raises(ValueError, match="range_fft_size.*LFM"):
        run_simulation(config)
