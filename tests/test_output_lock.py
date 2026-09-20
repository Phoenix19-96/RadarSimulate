"""Cooperative reservation tests for artifact publication."""

import threading

import pytest

from radarsim.config import OutputConfig
from radarsim.output import save_simulation
from test_output import small_result


def test_second_writer_fails_on_reservation_while_first_writer_is_staged(
    tmp_path, monkeypatch,
):
    """Break caught: supported concurrent writers race after the initial final check."""
    config, result = small_result()
    output = OutputConfig("reserved", tmp_path)
    staged = threading.Event()
    resume = threading.Event()
    first_errors = []
    original_write_rows = __import__("radarsim.output", fromlist=["_write_rows"])._write_rows

    def pause_after_staging(*args, **kwargs):
        staged.set()
        assert resume.wait(timeout=5)
        return original_write_rows(*args, **kwargs)

    monkeypatch.setattr("radarsim.output._write_rows", pause_after_staging)

    def first_writer():
        try:
            save_simulation(result, config, output)
        except Exception as error:  # pragma: no cover - asserted below
            first_errors.append(error)

    thread = threading.Thread(target=first_writer)
    thread.start()
    assert staged.wait(timeout=5)

    with pytest.raises(FileExistsError, match="reservation"):
        save_simulation(result, config, output)

    resume.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert first_errors == []
    assert [path.name for path in tmp_path.iterdir()] == ["reserved"]


def test_external_final_before_publish_is_preserved_and_reservation_is_cleaned(
    tmp_path, monkeypatch,
):
    """Break caught: an external final appearing while locked is overwritten or leaks a lock."""
    config, result = small_result()
    output = OutputConfig("external", tmp_path)
    final_directory = tmp_path / "external"

    def create_external_final(*_args):
        final_directory.mkdir()

    monkeypatch.setattr("radarsim.output._save_plots", create_external_final)
    with pytest.raises(FileExistsError, match="external"):
        save_simulation(result, config, output)

    assert list(final_directory.iterdir()) == []
    assert [path.name for path in tmp_path.iterdir()] == ["external"]
