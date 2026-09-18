"""Failure-safety tests for simulation artifact publication."""

import matplotlib.pyplot as plt
import pytest

from radarsim.config import OutputConfig
from radarsim.output import save_simulation
from test_output import small_result


def test_failed_artifact_write_leaves_no_partial_output_and_can_retry(tmp_path, monkeypatch):
    """Break caught: a failed write leaves a final or staging directory behind."""
    config, result = small_result()
    output = OutputConfig("atomic", tmp_path)

    def fail_plots(*_args):
        raise OSError("forced plot failure")

    monkeypatch.setattr("radarsim.output._save_plots", fail_plots)
    with pytest.raises(OSError, match="forced plot failure"):
        save_simulation(result, config, output)

    assert list(tmp_path.iterdir()) == []
    monkeypatch.undo()
    directory = save_simulation(result, config, output)
    assert directory == tmp_path / "atomic"


def test_savefig_failure_closes_the_created_figure(tmp_path, monkeypatch):
    """Break caught: a save failure leaks a Matplotlib figure."""
    config, result = small_result()
    output = OutputConfig("figure-failure", tmp_path)

    def fail_savefig(*_args, **_kwargs):
        raise OSError("forced savefig failure")

    monkeypatch.setattr("matplotlib.figure.Figure.savefig", fail_savefig)
    with pytest.raises(OSError, match="forced savefig failure"):
        save_simulation(result, config, output)

    assert plt.get_fignums() == []
    assert list(tmp_path.iterdir()) == []
