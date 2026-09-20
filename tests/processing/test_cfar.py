import numpy as np
import pytest

from radarsim.config import CFARConfig
from radarsim.models import CFARResult, RangeDopplerResult
from radarsim.processing.cfar import ca_cfar_2d, extract_detections


def test_cfar_threshold_uses_training_count_and_pfa():
    power = np.ones((9, 11))
    power[4, 5] = 100
    cfg = CFARConfig(training_cells=(2, 2), guard_cells=(1, 1), false_alarm_rate=1e-3)
    result = ca_cfar_2d(power, cfg)
    training_count = 7 * 7 - 3 * 3
    alpha = training_count * (cfg.false_alarm_rate ** (-1 / training_count) - 1)
    assert result.noise_w[4, 5] == pytest.approx(1)
    assert result.threshold_w[4, 5] == pytest.approx(alpha)
    assert result.detections[4, 5]


def test_cfar_marks_incomplete_edges_invalid():
    result = ca_cfar_2d(np.ones((9, 11)), CFARConfig(training_cells=(2, 2), guard_cells=(1, 1)))
    assert np.isnan(result.threshold_w[0, 0])
    assert not result.detections[0, 0]
    assert np.isfinite(result.threshold_w[4, 5])


def test_extraction_keeps_local_peak_and_reports_physical_coordinates():
    power = np.zeros((1, 1, 5, 6))
    power[0, 0, 2, 3] = 20
    power[0, 0, 2, 4] = 10
    mask = power[0, 0] > 0
    cfar = CFARResult(mask, np.ones((5, 6)) * 2, np.ones((5, 6)))
    rd = RangeDopplerResult(
        spectrum=np.sqrt(power).astype(complex), power_w=power,
        range_m=np.arange(6) * 10, velocity_mps=np.arange(5) - 2,
        range_resolution_m=10, velocity_resolution_mps=1,
        max_unambiguous_range_m=50, max_unambiguous_velocity_mps=2)
    detections = extract_detections(rd, [[cfar]])
    assert len(detections) == 1
    assert detections[0].range_m == 30
    assert detections[0].velocity_mps == 0
    assert detections[0].snr_db == pytest.approx(10 * np.log10(20))


def test_extraction_supports_multiple_frames_and_channels():
    power = np.ones((2, 2, 5, 6))
    power[1, 0, 2, 3] = 20
    cfar = [[ca_cfar_2d(power[f, c], CFARConfig((1, 1), (0, 0), 0.1)) for c in range(2)] for f in range(2)]
    rd = RangeDopplerResult(
        spectrum=np.sqrt(power).astype(complex), power_w=power,
        range_m=np.arange(6, dtype=float), velocity_mps=np.arange(5, dtype=float),
        range_resolution_m=1, velocity_resolution_mps=1,
        max_unambiguous_range_m=5, max_unambiguous_velocity_mps=2)
    detections = extract_detections(rd, cfar)
    assert any(d.frame_index == 1 and d.channel_index == 0 for d in detections)


def test_cfar_rejects_invalid_power_and_window():
    with pytest.raises(ValueError, match="finite"):
        ca_cfar_2d(np.array([[1.0, np.nan], [1.0, 1.0]]), CFARConfig((0, 1), (0, 0)))
    with pytest.raises(ValueError, match="fit"):
        ca_cfar_2d(np.ones((3, 3)), CFARConfig((2, 2), (1, 1)))

