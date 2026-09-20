import numpy as np
import pytest

from radarsim.channel import MonostaticChannel, radar_received_power_w
from radarsim.config import NoiseConfig, RadarConfig
from radarsim.scene import CartesianWaypoint, Target, Trajectory
from radarsim.waveforms.fmcw import FMCWConfig, FMCWWaveform
from radarsim.waveforms.lfm import LFMConfig, LFMWaveform


def moving_target(speed_mps: float) -> Target:
    trajectory = Trajectory.from_cartesian([
        CartesianWaypoint(0.0, (100.0, 0.0, 0.0)),
        CartesianWaypoint(1.0, (100.0 + speed_mps, 0.0, 0.0)),
    ])
    return Target("moving", "moving", 0.0, trajectory)


def test_received_power_scales_with_rcs_and_inverse_fourth_range() -> None:
    """The public radar-equation helper must retain its physical scaling."""
    radar = RadarConfig()
    power_at_100_m = radar_received_power_w(radar, 100.0, 1.0)

    assert radar_received_power_w(radar, 100.0, 10.0) / power_at_100_m == pytest.approx(10.0)
    assert power_at_100_m / radar_received_power_w(radar, 200.0, 1.0) == pytest.approx(16.0)


@pytest.mark.parametrize(
    ("waveform", "interval_s"),
    [
        (LFMWaveform(LFMConfig(1e6, 2e-6, 1e-3, 2e6, 8, 1)), 1e-3),
        (FMCWWaveform(FMCWConfig(1e6, 20e-6, 1e-3, 2e6, 8, 1)), 1e-3),
    ],
    ids=("lfm", "fmcw"),
)
def test_slow_time_phase_has_negative_doppler_for_receding_target(
    waveform: LFMWaveform | FMCWWaveform, interval_s: float,
) -> None:
    """Both waveform paths must get negative carrier Doppler when range grows."""
    radar = RadarConfig(carrier_hz=1e9)
    received = MonostaticChannel().propagate(
        waveform.build_tx(radar), waveform, radar, [moving_target(10.0)], NoiseConfig(0.0)
    )
    delay_bin = int(np.ceil((2 * 100 / 299_792_458) * waveform.sample_rate_hz))
    phase_step = np.angle(
        received.data[0, 1, 0, delay_bin] * np.conj(received.data[0, 0, 0, delay_bin])
    )
    expected = -2 * np.pi * (2 * 10 / radar.wavelength_m) * interval_s

    assert phase_step == pytest.approx(np.angle(np.exp(1j * expected)), abs=0.03)


@pytest.mark.parametrize("range_m, rcs_m2", [(0.0, 1.0), (100.0, 0.0), (np.inf, 1.0)])
def test_radar_equation_rejects_nonphysical_parameters(range_m: float, rcs_m2: float) -> None:
    """Invalid range or RCS must not leak infinities into channel amplitudes."""
    with pytest.raises(ValueError):
        radar_received_power_w(RadarConfig(), range_m, rcs_m2)


def test_radar_equation_rejects_gain_that_overflows_power() -> None:
    """A finite config gain still must not create an infinite received power."""
    with pytest.raises(ValueError, match="received power"):
        radar_received_power_w(RadarConfig(tx_gain_db=4_000.0), 100.0, 1.0)
