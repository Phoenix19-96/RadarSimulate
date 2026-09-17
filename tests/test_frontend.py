import numpy as np
import pytest

from radarsim.frontend import dechirp_fmcw, prepare_lfm
from radarsim.models import LFMFrontendResult, SignalCube
from radarsim.waveforms.lfm import LFMConfig, LFMWaveform


def cube(data, channel, metadata=None):
    data = np.asarray(data, complex)
    frames, slow, _, fast = data.shape
    metadata = dict(metadata or {"waveform_kind": "fmcw"})
    if metadata.get("waveform_kind") == "lfm":
        for key, value in {"sample_rate_hz": 1e6, "slope_hz_per_s": 5e11,
                           "repetition_interval_s": 10e-6}.items():
            metadata.setdefault(key, value)
    return SignalCube(data, ("frame", "slow_time", channel, "fast_time"),
                      np.arange(fast, dtype=float), np.arange(slow, dtype=float),
                      np.zeros((frames, slow, fast)),
                      metadata)


def test_dechirp_uses_received_times_conjugate_of_transmit():
    tx = cube(np.array([[[[1, 1j, -1, -1j]]]]), "tx")
    rx = cube(np.array([[[[1j, -1, -1j, 1]]]]), "rx")
    beat = dechirp_fmcw(rx, tx)
    assert beat.data == pytest.approx(rx.data * np.conj(tx.data))
    assert beat.dimensions == ("frame", "slow_time", "rx", "fast_time")
    assert beat.metadata["frontend"] == "fmcw_dechirp"


def test_dechirp_rejects_mismatched_time_shapes():
    tx = cube(np.ones((1, 1, 1, 4)), "tx")
    rx = cube(np.ones((1, 1, 1, 5)), "rx")
    with pytest.raises(ValueError, match="matching"):
        dechirp_fmcw(rx, tx)


def test_dechirp_rejects_shifted_absolute_sample_times():
    tx = cube(np.ones((1, 1, 1, 4)), "tx")
    rx = cube(np.ones((1, 1, 1, 4)), "rx")
    rx.sample_times_s[...] = 1e-6
    with pytest.raises(ValueError, match="sample_times_s"):
        dechirp_fmcw(rx, tx)


def test_dechirp_known_phase_is_received_times_conjugate_reference():
    tx_values = np.exp(1j * np.array([0.2, 0.8, 1.4]))
    rx_values = tx_values * np.exp(1j * 0.6)
    tx = cube(tx_values.reshape(1, 1, 1, -1), "tx")
    rx = cube(rx_values.reshape(1, 1, 1, -1), "rx")
    np.testing.assert_allclose(dechirp_fmcw(rx, tx).data, np.exp(1j * 0.6))


def test_prepare_lfm_returns_time_reversed_conjugate_reference():
    waveform = LFMWaveform(LFMConfig(2e6, 4e-6, 10e-6, 1e6, 2, 1))
    rx = cube(np.ones((1, 2, 1, 10)), "rx", {"waveform_kind": "lfm"})
    result = prepare_lfm(rx, waveform)
    assert result.received is rx
    assert result.matched_filter_reference == pytest.approx(
        np.conj(waveform.reference()[::-1]))


def test_prepare_lfm_rejects_incompatible_waveform_and_nonfinite_data():
    waveform = LFMWaveform(LFMConfig(2e6, 4e-6, 10e-6, 1e6, 2, 1))
    rx = cube(np.ones((1, 2, 1, 10)), "rx", {"waveform_kind": "fmcw"})
    with pytest.raises(ValueError, match="waveform"):
        prepare_lfm(rx, waveform)
    bad = cube(np.ones((1, 2, 1, 10)), "rx", {"waveform_kind": "lfm"})
    bad.data[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        prepare_lfm(bad, waveform)


def test_prepare_lfm_rejects_reference_defining_metadata_mismatch():
    waveform = LFMWaveform(LFMConfig(2e6, 4e-6, 10e-6, 1e6, 2, 1))
    metadata = {"waveform_kind": "lfm", "sample_rate_hz": 1e6,
                "slope_hz_per_s": 1.5e12, "repetition_interval_s": 10e-6}
    rx = cube(np.ones((1, 2, 1, 10)), "rx", metadata)
    with pytest.raises(ValueError, match="slope"):
        prepare_lfm(rx, waveform)


def test_lfm_result_is_identity_equal_and_defensively_copies_reference():
    waveform = LFMWaveform(LFMConfig(2e6, 4e-6, 10e-6, 1e6, 1, 1))
    rx = cube(np.ones((1, 1, 1, 10)), "rx", {"waveform_kind": "lfm"})
    reference = np.array([1 + 1j, 2 + 0j])
    result = LFMFrontendResult(rx, reference)
    reference[0] = 99 + 0j
    assert result.matched_filter_reference[0] == 1 + 1j
    assert result != LFMFrontendResult(rx, result.matched_filter_reference)


@pytest.mark.parametrize("reference", [np.array([]), np.ones((1, 2), complex),
                                        np.array([np.nan + 0j]), np.array([1.0])])
def test_lfm_result_rejects_invalid_direct_construction(reference):
    rx = cube(np.ones((1, 1, 1, 10)), "rx", {"waveform_kind": "lfm"})
    with pytest.raises(ValueError):
        LFMFrontendResult(rx, reference)


def test_lfm_result_rejects_non_rx_received_cube():
    tx = cube(np.ones((1, 1, 1, 10)), "tx", {"waveform_kind": "lfm"})
    with pytest.raises(ValueError, match="received"):
        LFMFrontendResult(tx, np.ones(2, complex))


def test_lfm_reference_gives_peak_at_known_delayed_pulse():
    waveform = LFMWaveform(LFMConfig(2e6, 4e-6, 10e-6, 1e6, 1, 1))
    reference = waveform.reference()
    delay = 2
    samples = np.zeros(10, complex)
    samples[delay:delay + reference.size] = reference
    rx = cube(samples.reshape(1, 1, 1, -1), "rx", {"waveform_kind": "lfm"})
    result = prepare_lfm(rx, waveform)
    matched = np.convolve(samples, result.matched_filter_reference, mode="full")
    assert np.argmax(np.abs(matched)) == delay + reference.size - 1
    assert np.abs(matched[delay + reference.size - 1]) == pytest.approx(reference.size)
