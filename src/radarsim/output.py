"""Persist explicit simulation artifacts without mutating simulation results.

Publication uses a same-root, cooperative ``.<experiment>.lock`` directory. Writers
using :func:`save_simulation` respect this reservation; external writers do not.

``raw_data.npz`` contains these arrays and shapes:

* ``tx`` and ``rx``: ``[frame, slow_time, channel, fast_time]``;
* ``range_doppler``: ``[frame, rx_channel, doppler_bin, range_bin]``;
* ``range_m``: ``[range_bin]``; and
* ``velocity_mps``: ``[doppler_bin]``.
"""

import csv
import ctypes
import errno
import json
import os
import shutil
import tempfile
import sys
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import OutputConfig
from .models import Detection, SimulationResult, TruthRecord
from .simulation import ExperimentConfig


def _jsonable(value):
    """Convert configuration values to portable JSON primitives."""
    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "__dict__"):
        return {
            name: _jsonable(item) for name, item in vars(value).items()
            if not name.startswith("_")
        }
    return value


def _write_rows(path: Path, rows: Iterable[TruthRecord | Detection], row_type) -> None:
    records = [asdict(row) for row in rows]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[field.name for field in fields(row_type)])
        writer.writeheader()
        writer.writerows(records)


def _artifact_directory(output: OutputConfig) -> Path:
    if not isinstance(output, OutputConfig):
        raise TypeError("output must be an OutputConfig")
    name = output.experiment_name
    if not isinstance(name, str) or not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("experiment_name must be a non-empty directory name")
    root = Path(output.output_root)
    if root.exists() and not root.is_dir():
        raise ValueError("output_root must be a directory")
    directory = root / name
    if directory.exists():
        raise FileExistsError(f"experiment output already exists: {directory}")
    return directory


def _db(power_w: np.ndarray) -> np.ndarray:
    return 10 * np.log10(np.maximum(power_w, np.finfo(float).tiny))


def _save_plots(directory: Path, result: SimulationResult) -> None:
    range_doppler = result.range_doppler
    power_w = range_doppler.power_w[0, 0]

    fig, axis = plt.subplots()
    try:
        axis.plot(range_doppler.range_m, _db(power_w.max(axis=0)))
        axis.set(xlabel="Range (m)", ylabel="Processing power (dBW, uncalibrated)", title="Range profile")
        fig.tight_layout()
        fig.savefig(directory / "range_profile.png", dpi=150)
    finally:
        plt.close(fig)

    fig, axis = plt.subplots()
    try:
        image = axis.pcolormesh(
            range_doppler.range_m, range_doppler.velocity_mps, _db(power_w),
            shading="auto",
        )
        axis.set(
            xlabel="Range (m)", ylabel="Radial velocity (m/s)",
            title="Range-Doppler map",
        )
        fig.colorbar(image, ax=axis, label="Processing power (dBW, uncalibrated)")
        fig.tight_layout()
        fig.savefig(directory / "range_doppler.png", dpi=150)
    finally:
        plt.close(fig)

    fig, axis = plt.subplots()
    try:
        image = axis.pcolormesh(
            range_doppler.range_m, range_doppler.velocity_mps,
            result.cfar[0][0].detections.astype(int), shading="auto",
        )
        axis.set(
            xlabel="Range (m)", ylabel="Radial velocity (m/s)",
            title="CFAR detections",
        )
        fig.colorbar(image, ax=axis, label="Detection")
        fig.tight_layout()
        fig.savefig(directory / "cfar_map.png", dpi=150)
    finally:
        plt.close(fig)


def _acquire_reservation(directory: Path) -> Path:
    """Atomically reserve a final name for cooperative writers on this filesystem."""
    reservation = directory.parent / f".{directory.name}.lock"
    try:
        reservation.mkdir()
    except FileExistsError as error:
        raise FileExistsError(
            f"experiment reservation already exists: {reservation}; "
            "another writer may be active or the reservation may be stale"
        ) from error
    return reservation


def _raise_destination_exists(destination: Path, error: OSError | None = None) -> None:
    message = f"destination already exists: {destination}"
    if error is None:
        raise FileExistsError(message)
    raise FileExistsError(error.errno, message, str(destination)) from error


def _rename_windows_noreplace(source: Path, destination: Path) -> None:
    """Use Windows rename semantics, which fail when the destination exists."""
    try:
        os.rename(source, destination)
    except OSError as error:
        if error.errno in {errno.EEXIST, errno.ENOTEMPTY} or error.winerror == 183:
            _raise_destination_exists(destination, error)
        raise


def _rename_linux_noreplace(source: Path, destination: Path) -> None:
    """Use Linux renameat2(RENAME_NOREPLACE), never an overwrite fallback."""
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as error:
        raise NotImplementedError("Linux renameat2 no-replace support is unavailable") from error
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        error = OSError(ctypes.get_errno(), os.strerror(ctypes.get_errno()))
        if error.errno in {errno.EEXIST, errno.ENOTEMPTY}:
            _raise_destination_exists(destination, error)
        raise error


def _rename_macos_noreplace(source: Path, destination: Path) -> None:
    """Use macOS renamex_np(RENAME_EXCL), never an overwrite fallback."""
    try:
        renamex_np = ctypes.CDLL(None, use_errno=True).renamex_np
    except AttributeError as error:
        raise NotImplementedError("macOS renamex_np no-replace support is unavailable") from error
    renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
    renamex_np.restype = ctypes.c_int
    if renamex_np(os.fsencode(source), os.fsencode(destination), 4) != 0:
        error = OSError(ctypes.get_errno(), os.strerror(ctypes.get_errno()))
        if error.errno in {errno.EEXIST, errno.ENOTEMPTY}:
            _raise_destination_exists(destination, error)
        raise error


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a same-filesystem directory without replacing destination."""
    if sys.platform == "win32":
        _rename_windows_noreplace(source, destination)
    elif sys.platform.startswith("linux"):
        _rename_linux_noreplace(source, destination)
    elif sys.platform == "darwin":
        _rename_macos_noreplace(source, destination)
    else:
        raise NotImplementedError(
            f"atomic no-replace directory rename is unsupported on {sys.platform}"
        )

def save_simulation(
    result: SimulationResult, config: ExperimentConfig, output: OutputConfig,
) -> Path:
    """Save one simulation result through an atomic, non-overwriting publish."""
    if not isinstance(result, SimulationResult):
        raise TypeError("result must be a SimulationResult")
    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig")
    directory = _artifact_directory(output)
    directory.parent.mkdir(parents=True, exist_ok=True)
    reservation = _acquire_reservation(directory)
    staging: Path | None = None
    try:
        if directory.exists():
            raise FileExistsError(f"experiment output already exists: {directory}")
        staging = Path(
            tempfile.mkdtemp(prefix=f".{directory.name}.", dir=directory.parent)
        )
        snapshot = _jsonable(config)
        snapshot["output"] = _jsonable(output)
        snapshot["signal_metadata"] = {
            "tx": _jsonable(result.tx.metadata),
            "rx": _jsonable(result.rx.metadata),
        }
        (staging / "config.json").write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        _write_rows(staging / "truth.csv", result.truth, TruthRecord)
        _write_rows(staging / "detections.csv", result.detections, Detection)
        if output.save_raw_data:
            np.savez_compressed(
                staging / "raw_data.npz", tx=result.tx.data, rx=result.rx.data,
                range_doppler=result.range_doppler.spectrum,
                range_m=result.range_doppler.range_m,
                velocity_mps=result.range_doppler.velocity_mps,
            )
        _save_plots(staging, result)
        _rename_directory_noreplace(staging, directory)
        staging = None
        return directory
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        reservation.rmdir()
