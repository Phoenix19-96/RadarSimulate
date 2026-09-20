import math

import pytest

from radarsim.config import NoiseConfig, RadarConfig


def test_initial_release_rejects_more_than_one_tx_or_rx():
    with pytest.raises(ValueError, match="num_tx"):
        RadarConfig(num_tx=2)
    with pytest.raises(ValueError, match="num_rx"):
        RadarConfig(num_rx=2)


def test_noise_sources_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        NoiseConfig(noise_power_w=1e-12, temperature_k=290.0, noise_figure_db=3.0, bandwidth_hz=1e6)


def test_thermal_noise_power():
    cfg = NoiseConfig(None, 290.0, 3.0, 1e6, seed=7)
    assert cfg.resolved_power_w() == pytest.approx(1.380_649e-23 * 290.0 * 10 ** 0.3 * 1e6)


def test_noise_power_and_thermal_parameters_must_be_physical():
    with pytest.raises(ValueError, match="noise_power_w"):
        NoiseConfig(noise_power_w=-1)
    with pytest.raises(ValueError, match="temperature_k"):
        NoiseConfig(None, 0, 3, 1e6)
    with pytest.raises(ValueError, match="bandwidth_hz"):
        NoiseConfig(None, 290, 3, 0)


@pytest.mark.parametrize("field", ["carrier_hz", "transmit_power_w", "tx_gain_db", "rx_gain_db"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_radar_rejects_non_finite_physical_scalars(field, value):
    with pytest.raises(ValueError, match=field):
        RadarConfig(**{field: value})


@pytest.mark.parametrize("value", [0, -1, 1.0, True, False])
def test_radar_rejects_invalid_channel_counts(value):
    with pytest.raises(ValueError, match="num_tx"):
        RadarConfig(num_tx=value)


@pytest.mark.parametrize("value", [0, -1, 1.0, True, False])
def test_radar_rejects_invalid_rx_counts(value):
    with pytest.raises(ValueError, match="num_rx"):
        RadarConfig(num_rx=value)


@pytest.mark.parametrize("value", [-1, 1.0, True, 1.5])
def test_noise_rejects_invalid_seed(value):
    with pytest.raises(ValueError, match="seed"):
        NoiseConfig(seed=value)


@pytest.mark.parametrize("field", ["noise_power_w", "temperature_k", "noise_figure_db", "bandwidth_hz"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_noise_rejects_non_finite_scalars(field, value):
    kwargs = {field: value}
    if field != "noise_power_w":
        kwargs = {"noise_power_w": None, "temperature_k": 290.0, "noise_figure_db": 3.0, "bandwidth_hz": 1e6}
        kwargs[field] = value
    with pytest.raises(ValueError, match=field):
        NoiseConfig(**kwargs)


@pytest.mark.parametrize("field", ["training_cells", "guard_cells"])
@pytest.mark.parametrize("value", [(1,), (1, 2, 3), (1.0, 2), (True, 2), (-1, 2)])
def test_cfar_rejects_invalid_cell_tuples(field, value):
    with pytest.raises(ValueError, match=field):
        from radarsim.config import CFARConfig
        CFARConfig(**{field: value})


@pytest.mark.parametrize("field,value", [("position_m", None)])
def test_radar_rejects_invalid_structured_fields(field, value):
    with pytest.raises(ValueError, match=field):
        RadarConfig(**{field: value})


@pytest.mark.parametrize("field,value", [("training_cells", None), ("training_cells", 3), ("guard_cells", None), ("guard_cells", 3)])
def test_cfar_rejects_non_sequence_structured_fields(field, value):
    with pytest.raises(ValueError, match=field):
        from radarsim.config import CFARConfig
        CFARConfig(**{field: value})


@pytest.mark.parametrize("field,value", [("position_m", [0.0, 0.0, 0.0])])
def test_radar_rejects_list_for_tuple_field(field, value):
    with pytest.raises(ValueError, match=field):
        RadarConfig(**{field: value})


@pytest.mark.parametrize("field,value", [("training_cells", [1, 2]), ("guard_cells", [1, 2]), ("training_cells", [0, 0])])
def test_cfar_rejects_lists_and_zero_training(field, value):
    with pytest.raises(ValueError, match=field if value != [0, 0] else "training_cells"):
        from radarsim.config import CFARConfig
        CFARConfig(**{field: value})


def test_tuple_config_fields_are_immutable_and_hashable():
    radar = RadarConfig()
    cfar = __import__("radarsim.config", fromlist=["CFARConfig"]).CFARConfig()
    with pytest.raises(AttributeError):
        radar.position_m = (1.0, 0.0, 0.0)
    with pytest.raises(AttributeError):
        cfar.training_cells = (1, 1)
    assert isinstance(hash(radar), int)
    assert isinstance(hash(cfar), int)
