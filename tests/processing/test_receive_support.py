import numpy as np
import pytest

from radarsim.config import ProcessingConfig, RadarConfig
from radarsim.frontend import prepare_lfm
from radarsim.models import SignalCube
from radarsim.processing.range_doppler import process_lfm
from radarsim.waveforms.lfm import LFMConfig, LFMWaveform


def lfm_record(frames=2, slow=2):
    waveform = LFMWaveform(LFMConfig(1e6, 2e-6, 10e-6, 2e6, slow, frames))
    tx = waveform.build_tx(RadarConfig())
    cube = SignalCube(np.zeros_like(tx.data), ("frame", "slow_time", "rx", "fast_time"),
                      tx.fast_time_s, tx.slow_time_s, tx.sample_times_s, tx.metadata)
    return waveform, cube


def test_lfm_full_reference_support_can_cross_adjacent_frame_boundary():
    """A pulse's complete receive support belongs to its transmit slow-time index."""
    waveform, rx = lfm_record()
    rx.data.reshape(-1)[38:42] = (1 + 2j) * waveform.reference()
    rd = process_lfm(prepare_lfm(rx, waveform), RadarConfig(), waveform.config,
                     ProcessingConfig(doppler_window="boxcar"))
    np.testing.assert_allclose(np.abs(rd.spectrum[0, 0, :, 18]), np.sqrt(5) * 4)
    np.testing.assert_allclose(rd.spectrum[1, 0, :, 18], 0, atol=1e-14)


def test_lfm_incomplete_final_echo_is_not_treated_as_full_matched_support():
    """Zero padding cannot invent the unobserved tail of the final delayed pulse."""
    waveform, rx = lfm_record(frames=1)
    rx.data.reshape(-1)[38:40] = waveform.reference()[:2]
    rd = process_lfm(prepare_lfm(rx, waveform), RadarConfig(), waveform.config,
                     ProcessingConfig(doppler_window="boxcar"))
    np.testing.assert_allclose(rd.spectrum[..., 18], 0, atol=1e-14)


def test_lfm_single_pulse_range_limit_excludes_unobserved_reference_tails():
    """One PRI cannot advertise delays that require an unrecorded next PRI."""
    waveform, rx = lfm_record(frames=1, slow=1)
    rd = process_lfm(prepare_lfm(rx, waveform), RadarConfig(), waveform.config,
                     ProcessingConfig(doppler_window="boxcar"))
    assert rd.max_unambiguous_range_m == pytest.approx(1274.1179465)
    assert rd.range_m.size == 17


def test_lfm_rejects_unrelated_absolute_frame_times_before_joining_receive_support():
    waveform, rx = lfm_record()
    rx.sample_times_s[1] += 1.0
    with pytest.raises(ValueError, match="sample_times_s"):
        process_lfm(prepare_lfm(rx, waveform), RadarConfig(), waveform.config,
                    ProcessingConfig())
