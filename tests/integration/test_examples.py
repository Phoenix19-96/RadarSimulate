"""Acceptance coverage for the documented, editable example scripts."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


ARTIFACTS = {
    "config.json",
    "truth.csv",
    "detections.csv",
    "raw_data.npz",
    "range_profile.png",
    "range_doppler.png",
    "cfar_map.png",
}


@pytest.mark.parametrize(
    ("script", "experiment"),
    [("run_fmcw.py", "fmcw_example"), ("run_lfm.py", "lfm_example")],
)
def test_example_script_runs_and_writes_artifacts(script, experiment, tmp_path):
    """Break caught: a documented example cannot run from a clean checkout."""
    root = Path(__file__).parents[2]
    env = os.environ.copy()
    env["RADARSIM_OUTPUT_ROOT"] = str(tmp_path)

    completed = subprocess.run(
        [sys.executable, str(root / "examples" / script)],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Saved" in completed.stdout
    assert "detections" in completed.stdout
    directory = tmp_path / experiment
    assert {path.name for path in directory.iterdir()} == ARTIFACTS
