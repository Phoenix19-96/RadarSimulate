import numpy as np
import pytest

from radarsim.channel import MonostaticChannel
from radarsim.config import NoiseConfig, RadarConfig
from radarsim.models import SignalCube
from radarsim.scene import CartesianWaypoint, Target, Trajectory
from radarsim.waveforms.fmcw import FMCWConfig, FMCWWaveform
from radarsim.waveforms.lfm import LFMConfig, LFMWaveform


def stationary_target(range_m: float = 300.0, rcs_dbsm: float = 0.0) -> Target:
    trajectory = Trajectory.from_cartesian([
        CartesianWaypoint(0.0, (range_m, 0.0, 0.0)),
        CartesianWaypoint(1.0, (range_m, 0.0, 0.0)),
    ])
    return Target("target", "target", rcs_dbsm, trajectory)


def test_channel_returns_rx_cube_and_delays_lfm_echo() -> None:
    """A delayed LFM echo must start at the physical round-trip delay."""
    radar = RadarConfig(carrier_hz=1e9)
    waveform = LFMWaveform(LFMConfig(1e6, 4e-6, 20e-6, 2e6, 2, 1))
    tx = waveform.build_tx(radar)

    received = MonostaticChannel().propagate(
        tx, waveform, radar, [stationary_target()], NoiseConfig(0.0)
    )

    assert received.dimensions == ("frame", "slow_time", "rx", "fast_time")
    assert received.data.shape == tx.data.shape
    expected_delay_samples = int(np.ceil((2 * 300 / 299_792_458) * 2e6))
    assert np.flatnonzero(np.abs(received.data[0, 0, 0]) > 0)[0] == expected_delay_samples


def test_noise_is_reproducible_for_the_same_seed() -> None:
    """A fixed seed must produce repeatable circular complex noise."""
    radar = RadarConfig()
    waveform = LFMWaveform(LFMConfig(1e6, 4e-6, 20e-6, 2e6, 2, 1))
    tx = waveform.build_tx(radar)
    channel = MonostaticChannel()

    first = channel.propagate(tx, waveform, radar, [], NoiseConfig(1e-6, seed=9))
    second = channel.propagate(tx, waveform, radar, [], NoiseConfig(1e-6, seed=9))

    assert first.data == pytest.approx(second.data)
    assert np.mean(np.abs(first.data) ** 2) == pytest.approx(1e-6, rel=0.35)


def test_channel_rejects_simulation_times_outside_trajectory() -> None:
    """Coverage must be checked before simulating a target trajectory."""
    radar = RadarConfig()
    waveform = LFMWaveform(LFMConfig(1e6, 4e-6, 20e-6, 2e6, 4, 1))
    trajectory = Trajectory.from_cartesian([
        CartesianWaypoint(0.0, (100.0, 0.0, 0.0)),
        CartesianWaypoint(1e-6, (100.0, 0.0, 0.0)),
    ])

    with pytest.raises(ValueError, match="outside trajectory"):
        MonostaticChannel().propagate(
            waveform.build_tx(radar), waveform, radar,
            [Target("short", "short", 0.0, trajectory)], NoiseConfig(0.0),
        )


@pytest.mark.parametrize(
    "waveform",
    [
        LFMWaveform(LFMConfig(1e6, 4e-6, 20e-6, 2e6, 2, 1)),
        FMCWWaveform(FMCWConfig(1e6, 10e-6, 20e-6, 2e6, 2, 1)),
    ],
    ids=("lfm", "fmcw"),
)
def test_channel_has_no_echo_when_supplied_tx_samples_are_zero(
    waveform: LFMWaveform | FMCWWaveform,
) -> None:
    """The supplied Tx cube, not just the waveform object, drives echoes."""
    radar = RadarConfig(carrier_hz=1e9)
    tx = waveform.build_tx(radar)
    zero_tx = SignalCube(
        np.zeros_like(tx.data), tx.dimensions, tx.fast_time_s, tx.slow_time_s,
        tx.sample_times_s, tx.metadata,
    )

    received = MonostaticChannel().propagate(
        zero_tx, waveform, radar, [stationary_target()], NoiseConfig(0.0)
    )

    assert np.all(received.data == 0)


@pytest.mark.parametrize(
    "waveform",
    [
        LFMWaveform(LFMConfig(1e6, 4e-6, 20e-6, 2e6, 2, 1)),
        FMCWWaveform(FMCWConfig(1e6, 10e-6, 20e-6, 2e6, 2, 1)),
    ],
    ids=("lfm", "fmcw"),
)
def test_channel_preserves_supplied_complex_tx_weighting(
    waveform: LFMWaveform | FMCWWaveform,
) -> None:
    """Complex Tx weights must scale the delayed waveform echo coherently."""
    radar = RadarConfig(carrier_hz=1e9)
    tx = waveform.build_tx(radar)
    weight = 2.0 - 0.5j
    weighted_tx = SignalCube(
        weight * tx.data, tx.dimensions, tx.fast_time_s, tx.slow_time_s,
        tx.sample_times_s, tx.metadata,
    )
    channel = MonostaticChannel()
    baseline = channel.propagate(tx, waveform, radar, [stationary_target()], NoiseConfig(0.0))
    weighted = channel.propagate(
        weighted_tx, waveform, radar, [stationary_target()], NoiseConfig(0.0)
    )

    nonzero = np.abs(baseline.data) > 0
    assert weighted.data[nonzero] == pytest.approx(weight * baseline.data[nonzero])


def test_channel_rejects_unrepresentable_thermal_noise_figure() -> None:
    """Thermal-noise overflow must report the parameter that caused it."""
    radar = RadarConfig()
    waveform = LFMWaveform(LFMConfig(1e6, 4e-6, 20e-6, 2e6, 2, 1))
    noise = NoiseConfig(
        noise_power_w=None, temperature_k=290.0, noise_figure_db=4_000.0,
        bandwidth_hz=1e6,
    )

    with pytest.raises(ValueError, match="noise_figure_db"):
        MonostaticChannel().propagate(
            waveform.build_tx(radar), waveform, radar, [], noise
        )


def _with_repetition_weights(tx: SignalCube, weights: list[complex]) -> SignalCube:
    """Apply one complex Tx weight per frame/slow-time waveform repetition."""
    assert len(weights) == tx.data.shape[0] * tx.data.shape[1]
    data = tx.data.copy()
    for repetition, weight in enumerate(weights):
        frame, slow_time = divmod(repetition, tx.data.shape[1])
        data[frame, slow_time, 0, :] *= weight
    return SignalCube(
        data, tx.dimensions, tx.fast_time_s, tx.slow_time_s,
        tx.sample_times_s, tx.metadata,
    )


def test_fmcw_tail_does_not_interpolate_the_next_chirp_or_frame_weight() -> None:
    """Continuous FMCW tails must hold their own repetition's Tx weight."""
    radar = RadarConfig(carrier_hz=1e9)
    waveform = FMCWWaveform(FMCWConfig(1e6, 10e-6, 10e-6, 2e6, 2, 2))
    tx = waveform.build_tx(radar)
    weighted_tx = _with_repetition_weights(tx, [1.0, 2.0, 3.0, 4.0])
    half_sample_range_m = 299_792_458.0 / (4.0 * waveform.sample_rate_hz)
    target = stationary_target(half_sample_range_m)
    channel = MonostaticChannel()
    baseline = channel.propagate(tx, waveform, radar, [target], NoiseConfig(0.0))
    weighted = channel.propagate(weighted_tx, waveform, radar, [target], NoiseConfig(0.0))

    # Each next-repetition receive start samples the preceding chirp tail.
    boundary_samples = ((0, 1), (1, 0), (1, 1))
    expected_weights = (1.0, 2.0, 3.0)
    observed = [
        weighted.data[frame, slow, 0, 0] / baseline.data[frame, slow, 0, 0]
        for frame, slow in boundary_samples
    ]
    assert observed == pytest.approx(expected_weights)
    assert weighted.data[0, 0, 0, 0] == 0


def test_lfm_pulse_weights_do_not_leak_across_frames_or_idle_time() -> None:
    """Pulsed LFM preserves each pulse weight, including zero and complex values."""
    radar = RadarConfig(carrier_hz=1e9)
    waveform = LFMWaveform(LFMConfig(1e6, 4e-6, 20e-6, 2e6, 1, 3))
    tx = waveform.build_tx(radar)
    weights = [1.0 + 0.5j, 0.0, -2.0 + 1.0j]
    weighted_tx = _with_repetition_weights(tx, weights)
    half_sample_range_m = 299_792_458.0 / (4.0 * waveform.sample_rate_hz)
    target = stationary_target(half_sample_range_m)
    channel = MonostaticChannel()
    baseline = channel.propagate(tx, waveform, radar, [target], NoiseConfig(0.0))
    weighted = channel.propagate(weighted_tx, waveform, radar, [target], NoiseConfig(0.0))

    # Fast-time index 8 receives the final half-sample pulse tail.
    assert weighted.data[0, 0, 0, 8] == pytest.approx(weights[0] * baseline.data[0, 0, 0, 8])
    assert weighted.data[1, 0, 0, 8] == 0
    assert weighted.data[2, 0, 0, 8] == pytest.approx(weights[2] * baseline.data[2, 0, 0, 8])
    # At a receive-frame boundary the delayed query is in the previous PRI's idle time.
    assert weighted.data[0, 0, 0, 0] == 0
    assert weighted.data[1, 0, 0, 0] == 0


@pytest.mark.parametrize(
    "waveform",
    [
        FMCWWaveform(FMCWConfig(1e6, 10e-6, 10e-6, 1e6, 9, 4)),
        LFMWaveform(LFMConfig(1e6, 4e-6, 10e-6, 1e6, 9, 4)),
    ],
    ids=("fmcw", "lfm"),
)
def test_channel_uses_waveform_boundary_repetition_for_exact_delayed_start(
    waveform: LFMWaveform | FMCWWaveform,
) -> None:
    """An exact frame-start query must select the same Tx repetition as the waveform."""
    radar = RadarConfig(carrier_hz=1e9)
    tx = waveform.build_tx(radar)
    weights = [complex(index + 1, -(index + 1) / 10.0) for index in range(36)]
    weighted_tx = _with_repetition_weights(tx, weights)
    exact_boundary_delay_s = tx.sample_times_s[3, 0, 1] - tx.sample_times_s[3, 0, 0]
    one_sample_range_m = 299_792_458.0 * exact_boundary_delay_s / 2.0
    target = stationary_target(one_sample_range_m)
    channel = MonostaticChannel()
    baseline = channel.propagate(tx, waveform, radar, [target], NoiseConfig(0.0))
    weighted = channel.propagate(weighted_tx, waveform, radar, [target], NoiseConfig(0.0))

    # Frame 3 starts at repetition 27 / 270 us.  Its first receive sample is
    # delayed by exactly one sample, so the query is the exact frame boundary.
    boundary_echo = (3, 0, 0, 1)
    assert abs(baseline.data[boundary_echo]) > 0
    assert weighted.data[boundary_echo] == pytest.approx(
        weights[27] * baseline.data[boundary_echo]
    )
