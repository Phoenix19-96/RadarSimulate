"""Race-safety test for final artifact publication."""

import pytest

from radarsim.config import OutputConfig
from radarsim.output import save_simulation
from test_output import small_result


def test_publish_refuses_a_final_directory_created_during_staging(tmp_path, monkeypatch):
    """Break caught: publication overwrites a final directory created by another writer."""
    config, result = small_result()
    output = OutputConfig("race", tmp_path)
    final_directory = tmp_path / "race"

    def create_competing_final(*_args):
        final_directory.mkdir()

    monkeypatch.setattr("radarsim.output._save_plots", create_competing_final)
    with pytest.raises(FileExistsError, match="race"):
        save_simulation(result, config, output)

    assert list(final_directory.iterdir()) == []
    assert [path.name for path in tmp_path.iterdir()] == ["race"]
