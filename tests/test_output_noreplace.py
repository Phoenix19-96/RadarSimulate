"""Native no-replace publication tests."""

import errno
import os

import pytest

from radarsim import output
from radarsim.config import OutputConfig
from radarsim.output import save_simulation
from test_output import small_result


def test_windows_dispatch_normalizes_existing_destination(monkeypatch, tmp_path):
    """Break caught: the Windows native move leaks platform-specific exists errors."""
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()

    def exists_error(*_args):
        raise FileExistsError(errno.EEXIST, "already exists")

    monkeypatch.setattr(output.sys, "platform", "win32")
    monkeypatch.setattr(output.os, "rename", exists_error)
    with pytest.raises(FileExistsError, match="destination already exists"):
        output._rename_directory_noreplace(source, destination)


def test_unsupported_platform_refuses_without_path_rename(monkeypatch, tmp_path):
    """Break caught: unsupported systems fall back to overwrite-capable Path.rename."""
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    monkeypatch.setattr(output.sys, "platform", "plan9")
    monkeypatch.setattr(output.Path, "rename", lambda *_args: pytest.fail("rename fallback"))

    with pytest.raises(NotImplementedError, match="no-replace"):
        output._rename_directory_noreplace(source, destination)


def test_native_publish_race_preserves_external_final_and_cleans_owned_paths(
    monkeypatch, tmp_path,
):
    """Break caught: a final appearing at native publish is overwritten or leaves debris."""
    config, result = small_result()
    output_config = OutputConfig("native-race", tmp_path)
    final_directory = tmp_path / "native-race"

    def native_race(_staging, destination):
        destination.mkdir()
        raise FileExistsError(f"destination already exists: {destination}")

    monkeypatch.setattr("radarsim.output._rename_directory_noreplace", native_race)
    with pytest.raises(FileExistsError, match="native-race"):
        save_simulation(result, config, output_config)

    assert list(final_directory.iterdir()) == []
    assert [path.name for path in tmp_path.iterdir()] == ["native-race"]
