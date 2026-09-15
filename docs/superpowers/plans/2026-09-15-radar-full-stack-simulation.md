# Radar Full-Stack Simulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a research-oriented Python package that simulates explicit FMCW and pulsed-LFM transmit/propagation/receive chains, then performs range-Doppler processing and 2-D CA-CFAR for configurable 1T1R multi-target scenes.

**Architecture:** Use a typed, layered complex-baseband pipeline. Configuration and scene truth feed explicit waveform objects; a monostatic channel samples delayed waveforms and adds propagation phase, radar-equation amplitude, and noise; waveform-specific frontends feed processing, detection, and artifact writers. Arrays retain frame, slow-time, channel, and fast-time axes so the 1T1R restriction stays at validation boundaries.

**Tech Stack:** Python 3.11+, NumPy, SciPy, Matplotlib, pytest, standard-library dataclasses/csv/json/pathlib.

**Spec:** `docs/superpowers/specs/2026-09-15-radar-full-stack-simulation-design.md`

## Global Constraints

- Python 3.11 or higher.
- Radar default position is `(0, 0, 0)`; azimuth `0°` is `+X`, azimuth `-90°` is `+Y`, and elevation `+90°` is `+Z`.
- Radial velocity is positive when a target recedes; physical monostatic Doppler is `fd = -2 * vr / wavelength`.
- Simulate complex baseband, retaining carrier-dependent wavelength, phase, and Doppler.
- Tx arrays use `[frame, slow_time, tx, fast_time]`; Rx arrays use `[frame, slow_time, rx, fast_time]`.
- Validate `num_tx == 1` and `num_rx == 1` without deleting channel axes.
- Processing consumes signal data and metadata only, never target truth.
- Every random calculation uses an explicitly seeded `numpy.random.Generator`.
- Follow red-green-refactor for every behavior.
- Exclude GUI, MIMO coding, beamforming, fluctuating RCS, antenna patterns, clutter, multipath, atmospheric attenuation, range migration correction, and real-time execution.

## File Map

```text
pyproject.toml                          package and dependencies
.gitignore                              generated Python and simulation artifacts
README.md                               setup and research usage
src/radarsim/constants.py               physical constants
src/radarsim/models.py                  signal/result dataclasses
src/radarsim/config.py                  base validated configs
src/radarsim/geometry.py                coordinate/radial formulas
src/radarsim/scene.py                   trajectories and targets
src/radarsim/waveforms/{base,fmcw,lfm}.py
src/radarsim/channel.py                 propagation and noise
src/radarsim/frontend.py                FMCW/LFM receive frontends
src/radarsim/processing/range_doppler.py
src/radarsim/processing/cfar.py
src/radarsim/simulation.py              orchestration
src/radarsim/output.py                  JSON/CSV/NPZ/plots
configs/{fmcw_example,lfm_example}.py
examples/{run_fmcw,run_lfm}.py
tests/                                  unit, physics, integration tests
```

---

### Task 1: Package Skeleton, Core Models, and Base Configuration

**Files:**
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `src/radarsim/__init__.py`
- Create: `src/radarsim/constants.py`
- Create: `src/radarsim/models.py`
- Create: `src/radarsim/config.py`
- Create: `tests/test_models.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Consumes: no project interfaces.
- Produces: `C_MPS`, `BOLTZMANN_J_PER_K`; `SignalCube`; `RadarConfig`, `NoiseConfig`, `CFARConfig`, `ProcessingConfig`, `OutputConfig`.

- [ ] **Step 1: Write failing model-shape tests**

```python
# tests/test_models.py
import numpy as np
import pytest
from radarsim.models import SignalCube

def test_signal_cube_accepts_named_four_dimensional_data():
    cube = SignalCube(
        np.zeros((2, 4, 1, 8), complex),
        ("frame", "slow_time", "rx", "fast_time"),
        np.arange(8) / 1e6, np.arange(4) / 1e3,
        np.zeros((2, 4, 8)), {"waveform_kind": "fmcw"},
    )
    assert cube.data.shape == (2, 4, 1, 8)

def test_signal_cube_rejects_non_four_dimensional_data():
    with pytest.raises(ValueError, match="four-dimensional"):
        SignalCube(np.zeros((4, 8), complex), ("slow_time", "fast_time"),
                   np.arange(8), np.arange(4), np.zeros((1, 4, 8)), {})
```

- [ ] **Step 2: Run the tests and confirm the package is missing**

Run: `python -m pytest tests/test_models.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'radarsim'`.

- [ ] **Step 3: Add packaging and the minimal model**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "radarsim"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["numpy>=1.26", "scipy>=1.11", "matplotlib>=3.8"]

[project.optional-dependencies]
test = ["pytest>=8.0"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

Create `.gitignore`:

```gitignore
__pycache__/
*.py[cod]
.pytest_cache/
*.egg-info/
build/
dist/
outputs/
```

```python
# src/radarsim/constants.py
C_MPS = 299_792_458.0
BOLTZMANN_J_PER_K = 1.380_649e-23
```

```python
# src/radarsim/models.py
from dataclasses import dataclass
from typing import Mapping
import numpy as np
from numpy.typing import NDArray

@dataclass(frozen=True)
class SignalCube:
    data: NDArray[np.complex128]
    dimensions: tuple[str, str, str, str]
    fast_time_s: NDArray[np.float64]
    slow_time_s: NDArray[np.float64]
    sample_times_s: NDArray[np.float64]
    metadata: Mapping[str, float | int | str]

    def __post_init__(self) -> None:
        if self.data.ndim != 4:
            raise ValueError("data must be four-dimensional")
        if len(self.dimensions) != 4:
            raise ValueError("dimensions must contain four names")
        frames, slow, _, fast = self.data.shape
        if self.fast_time_s.shape != (fast,):
            raise ValueError("fast_time_s length must match data")
        if self.slow_time_s.shape != (slow,):
            raise ValueError("slow_time_s length must match data")
        if self.sample_times_s.shape != (frames, slow, fast):
            raise ValueError("sample_times_s shape must be [frame, slow_time, fast_time]")
```

Export `SignalCube` from `src/radarsim/__init__.py`.

Run: `python -m pytest tests/test_models.py -v`

Expected: 2 tests pass.

- [ ] **Step 4: Write failing configuration tests**

```python
# tests/test_config.py
import pytest
from radarsim.config import NoiseConfig, RadarConfig

def test_initial_release_rejects_more_than_one_tx_or_rx():
    with pytest.raises(ValueError, match="num_tx"):
        RadarConfig(num_tx=2)
    with pytest.raises(ValueError, match="num_rx"):
        RadarConfig(num_rx=2)

def test_noise_sources_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        NoiseConfig(noise_power_w=1e-12, temperature_k=290.0,
                    noise_figure_db=3.0, bandwidth_hz=1e6)

def test_thermal_noise_power():
    cfg = NoiseConfig(None, 290.0, 3.0, 1e6, seed=7)
    assert cfg.resolved_power_w() == pytest.approx(
        1.380_649e-23 * 290.0 * 10 ** 0.3 * 1e6)

def test_noise_power_and_thermal_parameters_must_be_physical():
    with pytest.raises(ValueError, match="noise_power_w"):
        NoiseConfig(noise_power_w=-1)
    with pytest.raises(ValueError, match="temperature_k"):
        NoiseConfig(None, 0, 3, 1e6)
    with pytest.raises(ValueError, match="bandwidth_hz"):
        NoiseConfig(None, 290, 3, 0)
```

- [ ] **Step 5: Run configuration tests and confirm the module is missing**

Run: `python -m pytest tests/test_config.py -v`

Expected: collection fails because `radarsim.config` does not exist.

- [ ] **Step 6: Implement validated configuration dataclasses**

```python
# src/radarsim/config.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from .constants import BOLTZMANN_J_PER_K, C_MPS

@dataclass(frozen=True)
class RadarConfig:
    position_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    boresight_az_deg: float = 0.0
    boresight_el_deg: float = 0.0
    carrier_hz: float = 10e9
    transmit_power_w: float = 1.0
    tx_gain_db: float = 0.0
    rx_gain_db: float = 0.0
    num_tx: int = 1
    num_rx: int = 1

    def __post_init__(self):
        if self.carrier_hz <= 0: raise ValueError("carrier_hz must be positive")
        if self.transmit_power_w < 0: raise ValueError("transmit_power_w must be non-negative")
        if self.num_tx != 1: raise ValueError("num_tx must equal 1 in the initial release")
        if self.num_rx != 1: raise ValueError("num_rx must equal 1 in the initial release")

    @property
    def wavelength_m(self): return C_MPS / self.carrier_hz

@dataclass(frozen=True)
class NoiseConfig:
    noise_power_w: float | None = 0.0
    temperature_k: float | None = None
    noise_figure_db: float | None = None
    bandwidth_hz: float | None = None
    seed: int = 0

    def __post_init__(self):
        thermal = (self.temperature_k, self.noise_figure_db, self.bandwidth_hz)
        if self.noise_power_w is not None and any(v is not None for v in thermal):
            raise ValueError("direct and thermal noise settings are mutually exclusive")
        if self.noise_power_w is None and any(v is None for v in thermal):
            raise ValueError("all thermal noise settings are required")
        if self.noise_power_w is not None and self.noise_power_w < 0:
            raise ValueError("noise_power_w must be non-negative")
        if self.temperature_k is not None and self.temperature_k <= 0:
            raise ValueError("temperature_k must be positive")
        if self.bandwidth_hz is not None and self.bandwidth_hz <= 0:
            raise ValueError("bandwidth_hz must be positive")

    def resolved_power_w(self):
        if self.noise_power_w is not None: return self.noise_power_w
        return (BOLTZMANN_J_PER_K * self.temperature_k
                * 10 ** (self.noise_figure_db / 10) * self.bandwidth_hz)

@dataclass(frozen=True)
class CFARConfig:
    # Tuple order is (doppler_cells, range_cells).
    training_cells: tuple[int, int] = (8, 4)
    guard_cells: tuple[int, int] = (2, 1)
    false_alarm_rate: float = 1e-5

    def __post_init__(self):
        if any(v < 0 for v in (*self.training_cells, *self.guard_cells)):
            raise ValueError("CFAR cell counts must be non-negative")
        if self.training_cells == (0, 0):
            raise ValueError("CFAR requires training cells")
        if not 0 < self.false_alarm_rate < 1:
            raise ValueError("false_alarm_rate must be between 0 and 1")

@dataclass(frozen=True)
class ProcessingConfig:
    range_window: Literal["hann", "hamming", "blackman", "boxcar"] = "hann"
    doppler_window: Literal["hann", "hamming", "blackman", "boxcar"] = "hann"
    range_fft_size: int | None = None
    doppler_fft_size: int | None = None
    cfar: CFARConfig = field(default_factory=CFARConfig)

@dataclass(frozen=True)
class OutputConfig:
    experiment_name: str
    output_root: Path = Path("outputs")
    save_raw_data: bool = True
```

Run: `python -m pytest tests/test_models.py tests/test_config.py -v`

Expected: 6 tests pass.

- [ ] **Step 7: Commit the package foundation**

```powershell
git add .gitignore pyproject.toml src/radarsim tests/test_models.py tests/test_config.py
git commit -m "feat: add core radar simulation models and config"
```

---

### Task 2: Coordinate Geometry and Piecewise-Linear Trajectories

**Files:**
- Create: `src/radarsim/geometry.py`
- Create: `src/radarsim/scene.py`
- Modify: `src/radarsim/__init__.py`
- Create: `tests/test_geometry.py`
- Create: `tests/test_scene.py`

**Interfaces:**
- Consumes: NumPy.
- Produces: `cartesian_to_polar`, `polar_to_cartesian`, `radial_velocity`; `CartesianWaypoint`, `PolarWaypoint`, `Trajectory`, `Target`, `TargetState`.

- [ ] **Step 1: Write failing coordinate-convention tests**

```python
# tests/test_geometry.py
import numpy as np
import pytest
from radarsim.geometry import cartesian_to_polar, polar_to_cartesian, radial_velocity

@pytest.mark.parametrize(("xyz", "expected"), [
    ((1, 0, 0), (1, 0, 0)),
    ((0, 1, 0), (1, -90, 0)),
    ((0, -1, 0), (1, 90, 0)),
    ((0, 0, 1), (1, 0, 90)),
])
def test_axis_conventions(xyz, expected):
    assert cartesian_to_polar(np.asarray(xyz)) == pytest.approx(expected)

def test_polar_cartesian_round_trip():
    xyz = polar_to_cartesian(1250, -32, 14)
    assert cartesian_to_polar(xyz) == pytest.approx((1250, -32, 14))

def test_zero_range_has_no_defined_angles():
    with pytest.raises(ValueError, match="zero range"):
        cartesian_to_polar(np.zeros(3))

def test_radial_velocity_is_positive_when_receding():
    assert radial_velocity(np.array([100, 0, 0]), np.array([12, 3, 0])) == 12
```

- [ ] **Step 2: Run coordinate tests and observe the missing module**

Run: `python -m pytest tests/test_geometry.py -v`

Expected: collection fails because `radarsim.geometry` does not exist.

- [ ] **Step 3: Implement the centralized geometry formulas**

```python
# src/radarsim/geometry.py
import numpy as np
from numpy.typing import ArrayLike, NDArray

def normalize_azimuth_deg(angle_deg: float) -> float:
    return (angle_deg + 180.0) % 360.0 - 180.0

def cartesian_to_polar(xyz_m: ArrayLike) -> tuple[float, float, float]:
    xyz = np.asarray(xyz_m, dtype=float)
    if xyz.shape != (3,): raise ValueError("xyz_m must contain three coordinates")
    x, y, z = xyz
    range_m = float(np.linalg.norm(xyz))
    if range_m == 0: raise ValueError("angles are undefined at zero range")
    az = normalize_azimuth_deg(float(np.degrees(np.arctan2(-y, x))))
    el = float(np.degrees(np.arctan2(z, np.hypot(x, y))))
    return range_m, az, el

def polar_to_cartesian(range_m: float, azimuth_deg: float,
                       elevation_deg: float) -> NDArray[np.float64]:
    if range_m <= 0: raise ValueError("range_m must be positive")
    az, el = np.radians([azimuth_deg, elevation_deg])
    return np.array([range_m*np.cos(el)*np.cos(az),
                     -range_m*np.cos(el)*np.sin(az),
                     range_m*np.sin(el)])

def radial_velocity(position_m: ArrayLike, velocity_mps: ArrayLike) -> float:
    position = np.asarray(position_m, float)
    range_m = np.linalg.norm(position)
    if range_m == 0: raise ValueError("radial velocity is undefined at zero range")
    return float(np.dot(position, np.asarray(velocity_mps, float)) / range_m)
```

Run: `python -m pytest tests/test_geometry.py -v`

Expected: all coordinate tests pass.

- [ ] **Step 4: Write failing scene tests**

```python
# tests/test_scene.py
import pytest
from radarsim.scene import CartesianWaypoint, PolarWaypoint, Target, Trajectory

def test_trajectory_interpolates_position_and_velocity():
    tr = Trajectory.from_cartesian([
        CartesianWaypoint(0, (100, 0, 0)),
        CartesianWaypoint(2, (120, 10, 0)),
        CartesianWaypoint(5, (150, 10, 0)),
    ])
    state = tr.state_at(1)
    assert state.position_m == pytest.approx((110, 5, 0))
    assert state.velocity_mps == pytest.approx((10, 5, 0))

def test_internal_waypoint_uses_right_segment_velocity():
    tr = Trajectory.from_cartesian([
        CartesianWaypoint(0, (0, 0, 1)),
        CartesianWaypoint(1, (1, 0, 1)),
        CartesianWaypoint(3, (1, 4, 1)),
    ])
    assert tr.state_at(1).velocity_mps == pytest.approx((0, 2, 0))

def test_polar_points_are_converted_before_interpolation():
    tr = Trajectory.from_polar([
        PolarWaypoint(0, 100, 0, 0),
        PolarWaypoint(1, 100, -90, 0),
    ])
    assert tr.state_at(.5).position_m == pytest.approx((50, 50, 0))

def test_invalid_times_and_extrapolation_fail():
    with pytest.raises(ValueError, match="strictly increasing"):
        Trajectory.from_cartesian([
            CartesianWaypoint(1, (1, 0, 0)),
            CartesianWaypoint(1, (2, 0, 0)),
        ])
    tr = Trajectory.from_cartesian([
        CartesianWaypoint(0, (1, 0, 0)),
        CartesianWaypoint(1, (2, 0, 0)),
    ])
    with pytest.raises(ValueError, match="outside trajectory"):
        tr.state_at(2)

def test_target_rcs_dbsm_conversion():
    tr = Trajectory.from_cartesian([
        CartesianWaypoint(0, (1, 0, 0)),
        CartesianWaypoint(1, (2, 0, 0)),
    ])
    assert Target("t1", "test", 10, tr).rcs_m2 == pytest.approx(10)
```

- [ ] **Step 5: Run scene tests and observe the missing module**

Run: `python -m pytest tests/test_scene.py -v`

Expected: collection fails because `radarsim.scene` does not exist.

- [ ] **Step 6: Implement scene types and interpolation**

```python
# src/radarsim/scene.py
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from numpy.typing import NDArray
from .geometry import polar_to_cartesian

@dataclass(frozen=True)
class CartesianWaypoint:
    time_s: float
    position_m: tuple[float, float, float]

@dataclass(frozen=True)
class PolarWaypoint:
    time_s: float
    range_m: float
    azimuth_deg: float
    elevation_deg: float

@dataclass(frozen=True)
class TargetState:
    position_m: NDArray[np.float64]
    velocity_mps: NDArray[np.float64]

class Trajectory:
    def __init__(self, times_s, positions_m):
        self.times_s = np.asarray(times_s, float)
        self.positions_m = np.asarray(positions_m, float)
        if self.times_s.size < 2:
            raise ValueError("trajectory requires two waypoints")
        if self.positions_m.shape != (self.times_s.size, 3):
            raise ValueError("positions must have shape [waypoint, xyz]")
        if np.any(np.diff(self.times_s) <= 0):
            raise ValueError("waypoint times must be strictly increasing")

    @classmethod
    def from_cartesian(cls, points: Sequence[CartesianWaypoint]):
        return cls([p.time_s for p in points], [p.position_m for p in points])

    @classmethod
    def from_polar(cls, points: Sequence[PolarWaypoint]):
        return cls([p.time_s for p in points],
                   [polar_to_cartesian(p.range_m, p.azimuth_deg, p.elevation_deg)
                    for p in points])

    def state_at(self, time_s: float) -> TargetState:
        if not self.times_s[0] <= time_s <= self.times_s[-1]:
            raise ValueError("time_s is outside trajectory coverage")
        i = min(int(np.searchsorted(self.times_s, time_s, side="right") - 1),
                self.times_s.size - 2)
        t0, t1 = self.times_s[i:i+2]
        p0, p1 = self.positions_m[i:i+2]
        velocity = (p1 - p0) / (t1 - t0)
        return TargetState(p0 + (time_s - t0) * velocity, velocity)

    def states_at(self, times_s):
        times = np.asarray(times_s, float)
        states = [self.state_at(float(t)) for t in times.ravel()]
        shape = times.shape + (3,)
        return (np.stack([s.position_m for s in states]).reshape(shape),
                np.stack([s.velocity_mps for s in states]).reshape(shape))

@dataclass(frozen=True)
class Target:
    target_id: str
    name: str
    rcs_dbsm: float
    trajectory: Trajectory

    @property
    def rcs_m2(self): return 10 ** (self.rcs_dbsm / 10)
```

Export geometry and scene names from `src/radarsim/__init__.py`.

Run: `python -m pytest tests/test_geometry.py tests/test_scene.py -v`

Expected: all tests pass.

- [ ] **Step 7: Commit geometry and scene support**

```powershell
git add src/radarsim/geometry.py src/radarsim/scene.py src/radarsim/__init__.py tests/test_geometry.py tests/test_scene.py
git commit -m "feat: add coordinates and target trajectories"
```

---

### Task 3: Explicit FMCW and Pulsed-LFM Waveforms

**Files:**
- Create: `src/radarsim/waveforms/__init__.py`
- Create: `src/radarsim/waveforms/base.py`
- Create: `src/radarsim/waveforms/fmcw.py`
- Create: `src/radarsim/waveforms/lfm.py`
- Create: `tests/waveforms/test_fmcw.py`
- Create: `tests/waveforms/test_lfm.py`

**Interfaces:**
- Consumes: `RadarConfig`, `SignalCube`.
- Produces: `Waveform` protocol; `FMCWConfig`, `FMCWWaveform`; `LFMConfig`, `LFMWaveform`. Each waveform exposes `build_tx(radar)`, `sample_at(times_s)`, `reference()`, `sample_rate_hz`, `slow_time_count`, `frame_count`, and `repetition_interval_s`.

- [ ] **Step 1: Write failing FMCW tests**

```python
# tests/waveforms/test_fmcw.py
import numpy as np
import pytest
from radarsim.config import RadarConfig
from radarsim.waveforms.fmcw import FMCWConfig, FMCWWaveform

def test_fmcw_cube_shape_and_absolute_times():
    cfg = FMCWConfig(2e6, 8e-6, 10e-6, 1e6, 4, 2)
    cube = FMCWWaveform(cfg).build_tx(RadarConfig())
    assert cube.data.shape == (2, 4, 1, 8)
    assert cube.dimensions == ("frame", "slow_time", "tx", "fast_time")
    assert cube.sample_times_s[1, 0, 0] == pytest.approx(40e-6)

def test_fmcw_phase_matches_linear_chirp():
    cfg = FMCWConfig(2e6, 8e-6, 10e-6, 1e6, 2, 1)
    times = np.array([1e-6, 2e-6])
    expected = np.exp(1j * np.pi * cfg.slope_hz_per_s * times**2)
    assert FMCWWaveform(cfg).sample_at(times) == pytest.approx(expected)

def test_fmcw_is_zero_during_idle_time():
    cfg = FMCWConfig(2e6, 8e-6, 10e-6, 1e6, 2, 1)
    assert FMCWWaveform(cfg).sample_at(np.array([9e-6]))[0] == 0j
```

- [ ] **Step 2: Run FMCW tests and observe the missing module**

Run: `python -m pytest tests/waveforms/test_fmcw.py -v`

Expected: collection fails because `radarsim.waveforms.fmcw` does not exist.

- [ ] **Step 3: Implement the waveform protocol and FMCW**

```python
# src/radarsim/waveforms/base.py
from typing import Protocol
import numpy as np
from numpy.typing import NDArray
from radarsim.config import RadarConfig
from radarsim.models import SignalCube

class Waveform(Protocol):
    sample_rate_hz: float
    slow_time_count: int
    frame_count: int
    repetition_interval_s: float
    def build_tx(self, radar: RadarConfig) -> SignalCube: ...
    def sample_at(self, times_s: NDArray[np.float64]) -> NDArray[np.complex128]: ...
    def reference(self) -> NDArray[np.complex128]: ...
```

```python
# src/radarsim/waveforms/fmcw.py
from dataclasses import dataclass
import numpy as np
from radarsim.models import SignalCube

@dataclass(frozen=True)
class FMCWConfig:
    bandwidth_hz: float
    chirp_duration_s: float
    chirp_repetition_interval_s: float
    sample_rate_hz: float
    chirps_per_frame: int
    frame_count: int = 1

    def __post_init__(self):
        if min(self.bandwidth_hz, self.chirp_duration_s, self.sample_rate_hz) <= 0:
            raise ValueError("FMCW bandwidth, duration, and sample_rate_hz must be positive")
        if self.chirp_repetition_interval_s < self.chirp_duration_s:
            raise ValueError("chirp repetition interval must cover chirp duration")
        if min(self.chirps_per_frame, self.frame_count) < 1:
            raise ValueError("FMCW counts must be positive")

    @property
    def slope_hz_per_s(self):
        return self.bandwidth_hz / self.chirp_duration_s

class FMCWWaveform:
    def __init__(self, config):
        self.config = config
        self.sample_rate_hz = config.sample_rate_hz
        self.slow_time_count = config.chirps_per_frame
        self.frame_count = config.frame_count
        self.repetition_interval_s = config.chirp_repetition_interval_s

    def sample_at(self, times_s):
        local = np.mod(times_s, self.repetition_interval_s)
        active = local < self.config.chirp_duration_s
        out = np.zeros(times_s.shape, complex)
        out[active] = np.exp(1j*np.pi*self.config.slope_hz_per_s*local[active]**2)
        return out

    def reference(self):
        t = np.arange(round(self.config.chirp_duration_s*self.sample_rate_hz))
        return self.sample_at(t/self.sample_rate_hz)

    def build_tx(self, radar):
        fast = np.arange(round(self.config.chirp_duration_s*self.sample_rate_hz))/self.sample_rate_hz
        slow = np.arange(self.slow_time_count)*self.repetition_interval_s
        frame_span = self.slow_time_count*self.repetition_interval_s
        times = (np.arange(self.frame_count)[:, None, None]*frame_span
                 + slow[None, :, None] + fast[None, None, :])
        return SignalCube(self.sample_at(times)[:, :, None, :],
            ("frame", "slow_time", "tx", "fast_time"), fast, slow, times,
            {"waveform_kind": "fmcw", "sample_rate_hz": self.sample_rate_hz,
             "slope_hz_per_s": self.config.slope_hz_per_s,
             "repetition_interval_s": self.repetition_interval_s})
```

Run: `python -m pytest tests/waveforms/test_fmcw.py -v`

Expected: all FMCW tests pass.

- [ ] **Step 4: Write failing LFM tests**

```python
# tests/waveforms/test_lfm.py
import numpy as np
import pytest
from radarsim.config import RadarConfig
from radarsim.waveforms.lfm import LFMConfig, LFMWaveform

def test_lfm_cube_covers_pri_and_retains_tx_axis():
    cfg = LFMConfig(2e6, 4e-6, 10e-6, 1e6, 3, 1)
    cube = LFMWaveform(cfg).build_tx(RadarConfig())
    assert cube.data.shape == (1, 3, 1, 10)
    assert np.count_nonzero(cube.data[0, 0, 0]) == 4

def test_lfm_reference_contains_only_transmit_pulse():
    waveform = LFMWaveform(LFMConfig(2e6, 4e-6, 10e-6, 1e6, 3, 1))
    assert waveform.reference().shape == (4,)
    assert np.abs(waveform.reference()) == pytest.approx(np.ones(4))

def test_lfm_is_zero_outside_pulse():
    waveform = LFMWaveform(LFMConfig(2e6, 4e-6, 10e-6, 1e6, 3, 1))
    assert waveform.sample_at(np.array([5e-6]))[0] == 0j
```

- [ ] **Step 5: Run LFM tests and observe the missing module**

Run: `python -m pytest tests/waveforms/test_lfm.py -v`

Expected: collection fails because `radarsim.waveforms.lfm` does not exist.

- [ ] **Step 6: Implement pulsed LFM**

```python
# src/radarsim/waveforms/lfm.py
from dataclasses import dataclass
import numpy as np
from radarsim.models import SignalCube

@dataclass(frozen=True)
class LFMConfig:
    bandwidth_hz: float
    pulse_width_s: float
    pulse_repetition_interval_s: float
    sample_rate_hz: float
    pulses_per_frame: int
    frame_count: int = 1

    def __post_init__(self):
        if min(self.bandwidth_hz, self.pulse_width_s, self.sample_rate_hz) <= 0:
            raise ValueError("LFM bandwidth, width, and sample_rate_hz must be positive")
        if self.pulse_repetition_interval_s <= self.pulse_width_s:
            raise ValueError("pulse repetition interval must exceed pulse width")
        if min(self.pulses_per_frame, self.frame_count) < 1:
            raise ValueError("LFM counts must be positive")

    @property
    def slope_hz_per_s(self):
        return self.bandwidth_hz / self.pulse_width_s

class LFMWaveform:
    def __init__(self, config):
        self.config = config
        self.sample_rate_hz = config.sample_rate_hz
        self.slow_time_count = config.pulses_per_frame
        self.frame_count = config.frame_count
        self.repetition_interval_s = config.pulse_repetition_interval_s

    def sample_at(self, times_s):
        local = np.mod(times_s, self.repetition_interval_s)
        active = local < self.config.pulse_width_s
        out = np.zeros(times_s.shape, complex)
        out[active] = np.exp(1j*np.pi*self.config.slope_hz_per_s*local[active]**2)
        return out

    def reference(self):
        t = np.arange(round(self.config.pulse_width_s*self.sample_rate_hz))/self.sample_rate_hz
        return np.exp(1j*np.pi*self.config.slope_hz_per_s*t**2)

    def build_tx(self, radar):
        fast = np.arange(round(self.repetition_interval_s*self.sample_rate_hz))/self.sample_rate_hz
        slow = np.arange(self.slow_time_count)*self.repetition_interval_s
        frame_span = self.slow_time_count*self.repetition_interval_s
        times = (np.arange(self.frame_count)[:, None, None]*frame_span
                 + slow[None, :, None] + fast[None, None, :])
        return SignalCube(self.sample_at(times)[:, :, None, :],
            ("frame", "slow_time", "tx", "fast_time"), fast, slow, times,
            {"waveform_kind": "lfm", "sample_rate_hz": self.sample_rate_hz,
             "repetition_interval_s": self.repetition_interval_s})
```

Export both waveform classes/configs from `waveforms/__init__.py`.

Run: `python -m pytest tests/waveforms -v`

Expected: all waveform tests pass.

- [ ] **Step 7: Run all tests and commit**

Run: `python -m pytest -q`

Expected: all tests pass.

```powershell
git add src/radarsim/waveforms tests/waveforms
git commit -m "feat: add explicit FMCW and LFM waveforms"
```

---

### Task 4: Monostatic Baseband Channel and Reproducible Noise

**Files:**
- Create: `src/radarsim/channel.py`
- Create: `tests/test_channel.py`
- Create: `tests/physics/test_channel_physics.py`

**Interfaces:**
- Consumes: `RadarConfig`, `NoiseConfig`, `Target`, `Waveform`, and Tx `SignalCube`.
- Produces: `MonostaticChannel.propagate(tx, waveform, radar, targets, noise) -> SignalCube`, with Rx dimensions; `radar_received_power_w`.

- [ ] **Step 1: Write failing channel shape, delay, and seed tests**

```python
# tests/test_channel.py
import numpy as np
import pytest
from radarsim.channel import MonostaticChannel
from radarsim.config import NoiseConfig, RadarConfig
from radarsim.scene import CartesianWaypoint, Target, Trajectory
from radarsim.waveforms.lfm import LFMConfig, LFMWaveform

def stationary_target(range_m=300.0, rcs_dbsm=0.0):
    tr = Trajectory.from_cartesian([
        CartesianWaypoint(0.0, (range_m, 0, 0)),
        CartesianWaypoint(1.0, (range_m, 0, 0)),
    ])
    return Target("target", "target", rcs_dbsm, tr)

def test_channel_returns_rx_cube_and_delays_lfm_echo():
    radar = RadarConfig(carrier_hz=1e9)
    waveform = LFMWaveform(LFMConfig(1e6, 4e-6, 20e-6, 2e6, 2, 1))
    tx = waveform.build_tx(radar)
    rx = MonostaticChannel().propagate(
        tx, waveform, radar, [stationary_target()], NoiseConfig(0.0))
    assert rx.dimensions == ("frame", "slow_time", "rx", "fast_time")
    assert rx.data.shape == tx.data.shape
    expected_delay_samples = int(np.ceil((2 * 300 / 299_792_458) * 2e6))
    assert np.flatnonzero(np.abs(rx.data[0, 0, 0]) > 0)[0] == expected_delay_samples

def test_noise_is_reproducible_for_the_same_seed():
    radar = RadarConfig()
    waveform = LFMWaveform(LFMConfig(1e6, 4e-6, 20e-6, 2e6, 2, 1))
    tx = waveform.build_tx(radar)
    channel = MonostaticChannel()
    a = channel.propagate(tx, waveform, radar, [], NoiseConfig(1e-6, seed=9))
    b = channel.propagate(tx, waveform, radar, [], NoiseConfig(1e-6, seed=9))
    assert a.data == pytest.approx(b.data)
    assert np.mean(np.abs(a.data) ** 2) == pytest.approx(1e-6, rel=0.35)
```

- [ ] **Step 2: Run channel tests and observe the missing module**

Run: `python -m pytest tests/test_channel.py -v`

Expected: collection fails because `radarsim.channel` does not exist.

- [ ] **Step 3: Implement delayed waveform sampling, propagation phase, and noise**

```python
# src/radarsim/channel.py
from collections.abc import Sequence
import numpy as np
from .constants import C_MPS
from .models import SignalCube

def radar_received_power_w(radar, range_m, rcs_m2):
    gt = 10 ** (radar.tx_gain_db / 10)
    gr = 10 ** (radar.rx_gain_db / 10)
    return (radar.transmit_power_w * gt * gr * radar.wavelength_m**2 * rcs_m2
            / ((4*np.pi)**3 * range_m**4))

class MonostaticChannel:
    def propagate(self, tx, waveform, radar, targets: Sequence, noise):
        times = tx.sample_times_s
        received = np.zeros((tx.data.shape[0], tx.data.shape[1],
                             radar.num_rx, tx.data.shape[3]), complex)
        radar_position = np.asarray(radar.position_m)
        for target in targets:
            positions, _ = target.trajectory.states_at(times)
            offsets = positions - radar_position
            ranges = np.linalg.norm(offsets, axis=-1)
            if np.any(ranges == 0):
                raise ValueError("target range must be non-zero")
            delay_s = 2 * ranges / C_MPS
            delayed_tx = waveform.sample_at(times - delay_s)
            power = radar_received_power_w(radar, ranges, target.rcs_m2)
            carrier_phase = np.exp(-1j * 2*np.pi*radar.carrier_hz*delay_s)
            path = np.sqrt(power) * delayed_tx * carrier_phase
            received += path[:, :, None, :]
        generator = np.random.default_rng(noise.seed)
        noise_power = noise.resolved_power_w()
        if noise_power:
            scale = np.sqrt(noise_power / 2)
            received += scale * (
                generator.standard_normal(received.shape)
                + 1j * generator.standard_normal(received.shape))
        return SignalCube(received, ("frame", "slow_time", "rx", "fast_time"),
                          tx.fast_time_s, tx.slow_time_s, times,
                          dict(tx.metadata))
```

Run: `python -m pytest tests/test_channel.py -v`

Expected: both tests pass.

- [ ] **Step 4: Write failing radar-equation and Doppler physics tests**

```python
# tests/physics/test_channel_physics.py
import numpy as np
import pytest
from radarsim.channel import MonostaticChannel, radar_received_power_w
from radarsim.config import NoiseConfig, RadarConfig
from radarsim.scene import CartesianWaypoint, Target, Trajectory
from radarsim.waveforms.lfm import LFMConfig, LFMWaveform

def moving_target(speed_mps):
    tr = Trajectory.from_cartesian([
        CartesianWaypoint(0, (100, 0, 0)),
        CartesianWaypoint(1, (100 + speed_mps, 0, 0)),
    ])
    return Target("t", "moving", 0, tr)

def test_received_power_scales_with_rcs_and_inverse_fourth_range():
    radar = RadarConfig()
    p100 = radar_received_power_w(radar, 100, 1)
    assert radar_received_power_w(radar, 100, 10) / p100 == pytest.approx(10)
    assert p100 / radar_received_power_w(radar, 200, 1) == pytest.approx(16)

def test_slow_time_phase_has_negative_doppler_for_receding_target():
    radar = RadarConfig(carrier_hz=1e9)
    cfg = LFMConfig(1e6, 2e-6, 1e-3, 2e6, 8, 1)
    waveform = LFMWaveform(cfg)
    rx = MonostaticChannel().propagate(
        waveform.build_tx(radar), waveform, radar, [moving_target(10)],
        NoiseConfig(0))
    delay_bin = int(np.ceil((2 * 100 / 299_792_458) * cfg.sample_rate_hz))
    phase_step = np.angle(rx.data[0, 1, 0, delay_bin]
                          * np.conj(rx.data[0, 0, 0, delay_bin]))
    expected = -2*np.pi*(2*10/radar.wavelength_m)*cfg.pulse_repetition_interval_s
    assert phase_step == pytest.approx(np.angle(np.exp(1j*expected)), abs=0.03)
```

- [ ] **Step 5: Run physics tests and verify the first-order model**

Run: `python -m pytest tests/physics/test_channel_physics.py -v`

Expected: both tests pass. The test uses the first non-zero delayed sample; do not add a separate Doppler multiplier because the time-varying carrier-delay phase already creates the Doppler progression.

- [ ] **Step 6: Validate trajectory coverage before allocating Rx data**

Add this failing test to `tests/test_channel.py`:

```python
def test_channel_rejects_simulation_times_outside_trajectory():
    radar = RadarConfig()
    waveform = LFMWaveform(LFMConfig(1e6, 4e-6, 20e-6, 2e6, 4, 1))
    tr = Trajectory.from_cartesian([
        CartesianWaypoint(0, (100, 0, 0)),
        CartesianWaypoint(1e-6, (100, 0, 0)),
    ])
    with pytest.raises(ValueError, match="outside trajectory"):
        MonostaticChannel().propagate(
            waveform.build_tx(radar), waveform, radar,
            [Target("short", "short", 0, tr)], NoiseConfig(0))
```

Run: `python -m pytest tests/test_channel.py::test_channel_rejects_simulation_times_outside_trajectory -v`

Expected: fails because `states_at` is reached only after the receive array has been allocated.

Insert before the `received = np.zeros(...)` line in `MonostaticChannel.propagate`:

```python
        start_s = float(times.min())
        end_s = float(times.max())
        for target in targets:
            if (target.trajectory.times_s[0] > start_s
                    or target.trajectory.times_s[-1] < end_s):
                raise ValueError("simulation time is outside trajectory coverage")
```

Run: `python -m pytest tests/test_channel.py::test_channel_rejects_simulation_times_outside_trajectory -v`

Expected: pass before any receive array allocation.

- [ ] **Step 7: Run all tests and commit the channel**

Run: `python -m pytest -q`

Expected: all tests pass.

```powershell
git add src/radarsim/channel.py tests/test_channel.py tests/physics/test_channel_physics.py
git commit -m "feat: add monostatic propagation and noise"
```

---

### Task 5: Waveform-Specific Receive Frontends

**Files:**
- Modify: `src/radarsim/models.py`
- Create: `src/radarsim/frontend.py`
- Create: `tests/test_frontend.py`

**Interfaces:**
- Consumes: Tx/Rx `SignalCube`, `FMCWWaveform`, `LFMWaveform`.
- Produces: `dechirp_fmcw(rx, tx) -> SignalCube`; `prepare_lfm(rx, waveform) -> LFMFrontendResult`; `LFMFrontendResult(received, matched_filter_reference)` in `models.py`.

- [ ] **Step 1: Write failing FMCW dechirp tests**

```python
# tests/test_frontend.py
import numpy as np
import pytest
from radarsim.frontend import dechirp_fmcw
from radarsim.models import SignalCube
from radarsim.waveforms.lfm import LFMConfig, LFMWaveform

def cube(data, channel):
    frames, slow, _, fast = data.shape
    return SignalCube(
        np.asarray(data, complex),
        ("frame", "slow_time", channel, "fast_time"),
        np.arange(fast, dtype=float),
        np.arange(slow, dtype=float),
        np.zeros((frames, slow, fast)),
        {"waveform_kind": "fmcw"},
    )

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
```

- [ ] **Step 2: Run frontend tests and observe the missing module**

Run: `python -m pytest tests/test_frontend.py -v`

Expected: collection fails because `radarsim.frontend` does not exist.

- [ ] **Step 3: Implement FMCW dechirp without removing axes**

```python
# src/radarsim/frontend.py
import numpy as np
from .models import SignalCube

def dechirp_fmcw(rx: SignalCube, tx: SignalCube) -> SignalCube:
    if rx.data.shape[:2] + rx.data.shape[3:] != tx.data.shape[:2] + tx.data.shape[3:]:
        raise ValueError("Tx and Rx require matching frame, slow-time, and fast-time shapes")
    if tx.data.shape[2] != 1:
        raise ValueError("initial FMCW frontend requires one Tx reference")
    data = rx.data * np.conj(tx.data[:, :, :1, :])
    metadata = dict(rx.metadata)
    metadata["frontend"] = "fmcw_dechirp"
    return SignalCube(data, ("frame", "slow_time", "rx", "fast_time"),
                      rx.fast_time_s, rx.slow_time_s, rx.sample_times_s, metadata)
```

Run: `python -m pytest tests/test_frontend.py -v`

Expected: the two FMCW tests pass.

- [ ] **Step 4: Write and run the failing LFM reference test**

Append to `tests/test_frontend.py`:

```python
from radarsim.frontend import prepare_lfm
from radarsim.waveforms.lfm import LFMConfig, LFMWaveform

def test_prepare_lfm_returns_time_reversed_conjugate_reference():
    waveform = LFMWaveform(LFMConfig(2e6, 4e-6, 10e-6, 1e6, 2, 1))
    rx = cube(np.ones((1, 2, 1, 10)), "rx")
    result = prepare_lfm(rx, waveform)
    assert result.received is rx
    assert result.matched_filter_reference == pytest.approx(
        np.conj(waveform.reference()[::-1]))
```

Run: `python -m pytest tests/test_frontend.py::test_prepare_lfm_returns_time_reversed_conjugate_reference -v`

Expected: collection fails because `prepare_lfm` is not defined.

- [ ] **Step 5: Implement the LFM frontend result**

Append to `src/radarsim/models.py`:

```python
@dataclass(frozen=True)
class LFMFrontendResult:
    received: SignalCube
    matched_filter_reference: NDArray[np.complex128]
```

Append to `src/radarsim/frontend.py`:

```python
from .models import LFMFrontendResult

def prepare_lfm(rx: SignalCube, waveform) -> LFMFrontendResult:
    reference = waveform.reference()
    if reference.size > rx.data.shape[-1]:
        raise ValueError("LFM reference cannot exceed receive fast-time length")
    return LFMFrontendResult(rx, np.conj(reference[::-1]))
```

Run: `python -m pytest tests/test_frontend.py -v`

Expected: all three frontend tests pass.

- [ ] **Step 6: Run all tests and commit frontends**

Run: `python -m pytest -q`

Expected: all tests pass.

```powershell
git add src/radarsim/models.py src/radarsim/frontend.py tests/test_frontend.py
git commit -m "feat: add FMCW and LFM receive frontends"
```

---

### Task 6: FMCW and LFM Range-Doppler Processing

**Files:**
- Modify: `src/radarsim/models.py`
- Create: `src/radarsim/processing/__init__.py`
- Create: `src/radarsim/processing/range_doppler.py`
- Create: `tests/processing/test_range_doppler.py`

**Interfaces:**
- Consumes: frontend `SignalCube`/`LFMFrontendResult`, `RadarConfig`, `ProcessingConfig`, `FMCWConfig`, `LFMConfig`.
- Produces: `RangeDopplerResult`; `process_fmcw(...) -> RangeDopplerResult`; `process_lfm(...) -> RangeDopplerResult`.

- [ ] **Step 1: Write failing result-model and FMCW axis tests**

```python
# tests/processing/test_range_doppler.py
import numpy as np
import pytest
from radarsim.config import ProcessingConfig, RadarConfig
from radarsim.models import SignalCube
from radarsim.processing.range_doppler import process_fmcw
from radarsim.waveforms.fmcw import FMCWConfig

def synthetic_beat(radar, waveform, range_m, velocity_mps):
    nfast = round(waveform.chirp_duration_s * waveform.sample_rate_hz)
    fast = np.arange(nfast) / waveform.sample_rate_hz
    slow = np.arange(waveform.chirps_per_frame) * waveform.chirp_repetition_interval_s
    beat_hz = 2 * waveform.slope_hz_per_s * range_m / 299_792_458
    doppler_hz = -2 * velocity_mps / radar.wavelength_m
    data = np.exp(-1j*2*np.pi*beat_hz*fast)[None, None, None, :]
    data = data * np.exp(1j*2*np.pi*doppler_hz*slow)[None, :, None, None]
    times = slow[None, :, None] + fast[None, None, :]
    return SignalCube(data, ("frame", "slow_time", "rx", "fast_time"),
                      fast, slow, times,
                      {"waveform_kind": "fmcw", "frontend": "fmcw_dechirp"})

def test_fmcw_axes_and_peak_use_receding_positive_velocity():
    radar = RadarConfig(carrier_hz=10e9)
    waveform = FMCWConfig(20e6, 40e-6, 50e-6, 10e6, 64, 1)
    cube = synthetic_beat(radar, waveform, 30, 5)
    result = process_fmcw(
        cube, radar, waveform,
        ProcessingConfig(range_window="boxcar", doppler_window="boxcar",
                         range_fft_size=512, doppler_fft_size=64))
    d, r = np.unravel_index(np.argmax(result.power_w[0, 0]),
                            result.power_w[0, 0].shape)
    assert result.spectrum.shape == (1, 1, 64, result.range_m.size)
    assert result.range_m[r] == pytest.approx(30, abs=result.range_resolution_m)
    assert result.velocity_mps[d] == pytest.approx(5, abs=result.velocity_resolution_mps)
```

- [ ] **Step 2: Run the focused test and observe the missing processing module**

Run: `python -m pytest tests/processing/test_range_doppler.py::test_fmcw_axes_and_peak_use_receding_positive_velocity -v`

Expected: collection fails because `radarsim.processing.range_doppler` does not exist.

- [ ] **Step 3: Add the result type and FMCW processor**

Append to `src/radarsim/models.py`:

```python
@dataclass(frozen=True)
class RangeDopplerResult:
    spectrum: NDArray[np.complex128]
    power_w: NDArray[np.float64]
    range_m: NDArray[np.float64]
    velocity_mps: NDArray[np.float64]
    range_resolution_m: float
    velocity_resolution_mps: float
    max_unambiguous_range_m: float
    max_unambiguous_velocity_mps: float
```

Create `src/radarsim/processing/range_doppler.py`:

```python
import numpy as np
from scipy.signal import fftconvolve, get_window
from radarsim.constants import C_MPS
from radarsim.models import RangeDopplerResult

def _fft_size(configured, required, name):
    size = required if configured is None else configured
    if size < required:
        raise ValueError(f"{name} must be at least {required}")
    return size

def _slow_time_fft(range_data, radar, repetition_s, window_name, fft_size):
    slow_count = range_data.shape[1]
    size = _fft_size(fft_size, slow_count, "doppler_fft_size")
    window = get_window(window_name, slow_count)
    spectrum = np.fft.fftshift(
        np.fft.fft(range_data * window[None, :, None, None],
                   n=size, axis=1), axes=1)
    doppler_hz = np.fft.fftshift(np.fft.fftfreq(size, repetition_s))
    velocity = -doppler_hz * radar.wavelength_m / 2
    order = np.argsort(velocity)
    return np.transpose(spectrum[:, order], (0, 2, 1, 3)), velocity[order]

def process_fmcw(beat, radar, waveform, processing):
    if beat.metadata.get("frontend") != "fmcw_dechirp":
        raise ValueError("process_fmcw requires dechirped FMCW data")
    fast_count = beat.data.shape[-1]
    range_size = _fft_size(processing.range_fft_size, fast_count, "range_fft_size")
    window = get_window(processing.range_window, fast_count)
    fast_spectrum = np.fft.fft(
        beat.data * window[None, None, None, :], n=range_size, axis=-1)
    frequencies = np.fft.fftfreq(range_size, 1 / waveform.sample_rate_hz)
    keep = frequencies <= 0
    ranges = -frequencies[keep] * C_MPS / (2 * waveform.slope_hz_per_s)
    order = np.argsort(ranges)
    range_data = fast_spectrum[..., keep][..., order]
    ranges = ranges[order]
    spectrum, velocity = _slow_time_fft(
        range_data, radar, waveform.chirp_repetition_interval_s,
        processing.doppler_window, processing.doppler_fft_size)
    range_resolution = C_MPS / (2 * waveform.bandwidth_hz)
    velocity_resolution = radar.wavelength_m / (
        2 * waveform.chirps_per_frame * waveform.chirp_repetition_interval_s)
    return RangeDopplerResult(
        spectrum, np.abs(spectrum)**2, ranges, velocity,
        range_resolution, velocity_resolution, float(ranges[-1]),
        radar.wavelength_m / (4 * waveform.chirp_repetition_interval_s))
```

Run: `python -m pytest tests/processing/test_range_doppler.py::test_fmcw_axes_and_peak_use_receding_positive_velocity -v`

Expected: pass.

- [ ] **Step 4: Add failing LFM matched-filter and axis test**

Append to `tests/processing/test_range_doppler.py`:

```python
from radarsim.models import LFMFrontendResult
from radarsim.processing.range_doppler import process_lfm
from radarsim.waveforms.lfm import LFMConfig, LFMWaveform

def test_lfm_match_filter_peak_and_velocity_axis():
    radar = RadarConfig(carrier_hz=10e9)
    cfg = LFMConfig(2e6, 8e-6, 80e-6, 4e6, 32, 1)
    waveform = LFMWaveform(cfg)
    reference = waveform.reference()
    delay_samples = 40
    velocity = 3.0
    doppler = -2 * velocity / radar.wavelength_m
    slow = np.arange(cfg.pulses_per_frame) * cfg.pulse_repetition_interval_s
    raw = np.zeros((1, cfg.pulses_per_frame, 1,
                    round(cfg.pulse_repetition_interval_s*cfg.sample_rate_hz)), complex)
    raw[0, :, 0, delay_samples:delay_samples+reference.size] = (
        reference[None, :] * np.exp(1j*2*np.pi*doppler*slow)[:, None])
    fast = np.arange(raw.shape[-1]) / cfg.sample_rate_hz
    cube = SignalCube(raw, ("frame", "slow_time", "rx", "fast_time"),
                      fast, slow, slow[None, :, None] + fast[None, None, :],
                      {"waveform_kind": "lfm"})
    result = process_lfm(
        LFMFrontendResult(cube, np.conj(reference[::-1])),
        radar, cfg,
        ProcessingConfig(doppler_window="boxcar", doppler_fft_size=32))
    d, r = np.unravel_index(np.argmax(result.power_w[0, 0]),
                            result.power_w[0, 0].shape)
    expected_range = delay_samples / cfg.sample_rate_hz * 299_792_458 / 2
    assert result.range_m[r] == pytest.approx(expected_range,
                                              abs=299_792_458/(2*cfg.sample_rate_hz))
    assert result.velocity_mps[d] == pytest.approx(velocity,
                                                   abs=result.velocity_resolution_mps)
```

- [ ] **Step 5: Run the LFM test and observe the missing function**

Run: `python -m pytest tests/processing/test_range_doppler.py::test_lfm_match_filter_peak_and_velocity_axis -v`

Expected: fails because `process_lfm` is not defined.

- [ ] **Step 6: Implement LFM matched filtering and Doppler processing**

Append to `src/radarsim/processing/range_doppler.py`:

```python
def process_lfm(frontend, radar, waveform, processing):
    rx = frontend.received
    frames, slow, channels, fast = rx.data.shape
    reference_size = frontend.matched_filter_reference.size
    compressed = np.empty((frames, slow, channels,
                           fast + reference_size - 1), complex)
    for frame in range(frames):
        for pulse in range(slow):
            for channel in range(channels):
                compressed[frame, pulse, channel] = fftconvolve(
                    rx.data[frame, pulse, channel],
                    frontend.matched_filter_reference, mode="full")
    delay_samples = np.arange(compressed.shape[-1]) - (reference_size - 1)
    valid = (delay_samples >= 0) & (
        delay_samples / waveform.sample_rate_hz
        < waveform.pulse_repetition_interval_s)
    ranges = delay_samples[valid] / waveform.sample_rate_hz * C_MPS / 2
    spectrum, velocity = _slow_time_fft(
        compressed[..., valid], radar, waveform.pulse_repetition_interval_s,
        processing.doppler_window, processing.doppler_fft_size)
    return RangeDopplerResult(
        spectrum, np.abs(spectrum)**2, ranges, velocity,
        C_MPS/(2*waveform.bandwidth_hz),
        radar.wavelength_m/(2*waveform.pulses_per_frame
                            * waveform.pulse_repetition_interval_s),
        float(ranges[-1]),
        radar.wavelength_m/(4*waveform.pulse_repetition_interval_s))
```

Export both processors from `src/radarsim/processing/__init__.py`.

Run: `python -m pytest tests/processing/test_range_doppler.py -v`

Expected: both processing tests pass.

- [ ] **Step 7: Add invalid FFT-size tests**

Append to `tests/processing/test_range_doppler.py`:

```python
def test_fmcw_rejects_range_fft_smaller_than_fast_time():
    radar = RadarConfig()
    waveform = FMCWConfig(20e6, 40e-6, 50e-6, 10e6, 8, 1)
    cube = synthetic_beat(radar, waveform, 30, 0)
    with pytest.raises(ValueError, match="range_fft_size"):
        process_fmcw(
            cube, radar, waveform,
            ProcessingConfig(range_fft_size=cube.data.shape[-1] - 1))

def test_fmcw_rejects_doppler_fft_smaller_than_slow_time():
    radar = RadarConfig()
    waveform = FMCWConfig(20e6, 40e-6, 50e-6, 10e6, 8, 1)
    cube = synthetic_beat(radar, waveform, 30, 0)
    with pytest.raises(ValueError, match="doppler_fft_size"):
        process_fmcw(
            cube, radar, waveform,
            ProcessingConfig(doppler_fft_size=cube.data.shape[1] - 1))
```

Run: `python -m pytest tests/processing/test_range_doppler.py -v`

Expected: all processing tests pass.

- [ ] **Step 8: Run all tests and commit**

Run: `python -m pytest -q`

Expected: all tests pass.

```powershell
git add src/radarsim/models.py src/radarsim/processing tests/processing
git commit -m "feat: add range Doppler processing"
```

---

### Task 7: Two-Dimensional CA-CFAR and Detection Extraction

**Files:**
- Modify: `src/radarsim/models.py`
- Create: `src/radarsim/processing/cfar.py`
- Modify: `src/radarsim/processing/__init__.py`
- Create: `tests/processing/test_cfar.py`

**Interfaces:**
- Consumes: `RangeDopplerResult`, `CFARConfig`.
- Produces: `CFARResult`, `Detection`; `ca_cfar_2d(power_w, config) -> CFARResult`; `extract_detections(rd, cfar) -> tuple[Detection, ...]`.

- [ ] **Step 1: Write failing CA-CFAR threshold and edge tests**

```python
# tests/processing/test_cfar.py
import numpy as np
import pytest
from radarsim.config import CFARConfig
from radarsim.processing.cfar import ca_cfar_2d

def test_cfar_threshold_uses_training_count_and_pfa():
    power = np.ones((9, 11))
    power[4, 5] = 100
    cfg = CFARConfig(training_cells=(2, 2), guard_cells=(1, 1),
                     false_alarm_rate=1e-3)
    result = ca_cfar_2d(power, cfg)
    training_count = 7*7 - 3*3
    alpha = training_count * (cfg.false_alarm_rate**(-1/training_count) - 1)
    assert result.noise_w[4, 5] == pytest.approx(1)
    assert result.threshold_w[4, 5] == pytest.approx(alpha)
    assert result.detections[4, 5]

def test_cfar_marks_incomplete_edges_invalid():
    result = ca_cfar_2d(
        np.ones((9, 11)),
        CFARConfig(training_cells=(2, 2), guard_cells=(1, 1)))
    assert np.isnan(result.threshold_w[0, 0])
    assert not result.detections[0, 0]
    assert np.isfinite(result.threshold_w[4, 5])
```

- [ ] **Step 2: Run CFAR tests and observe the missing module**

Run: `python -m pytest tests/processing/test_cfar.py -v`

Expected: collection fails because `radarsim.processing.cfar` does not exist.

- [ ] **Step 3: Add `CFARResult` and implement the training mask**

Append to `src/radarsim/models.py`:

```python
@dataclass(frozen=True)
class CFARResult:
    detections: NDArray[np.bool_]
    threshold_w: NDArray[np.float64]
    noise_w: NDArray[np.float64]
```

Create `src/radarsim/processing/cfar.py`:

```python
from dataclasses import dataclass
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import maximum_filter
from radarsim.models import CFARResult

def ca_cfar_2d(power_w, config):
    power = np.asarray(power_w, float)
    if power.ndim != 2:
        raise ValueError("power_w must be a two-dimensional Doppler-range map")
    train_doppler, train_range = config.training_cells
    guard_doppler, guard_range = config.guard_cells
    outer_doppler = train_doppler + guard_doppler
    outer_range = train_range + guard_range
    kernel = np.ones((2*outer_doppler+1, 2*outer_range+1), float)
    kernel[outer_doppler-guard_doppler:outer_doppler+guard_doppler+1,
           outer_range-guard_range:outer_range+guard_range+1] = 0
    count = int(kernel.sum())
    if (power.shape[0] <= 2*outer_doppler
            or power.shape[1] <= 2*outer_range):
        raise ValueError("CFAR window does not fit the power map")
    noise = convolve2d(power, kernel, mode="same", boundary="fill") / count
    alpha = count * (config.false_alarm_rate**(-1/count) - 1)
    threshold = alpha * noise
    valid = np.zeros(power.shape, bool)
    valid[outer_doppler:power.shape[0]-outer_doppler,
          outer_range:power.shape[1]-outer_range] = True
    threshold[~valid] = np.nan
    noise[~valid] = np.nan
    detections = valid & (power > threshold)
    return CFARResult(detections, threshold, noise)
```

Run: `python -m pytest tests/processing/test_cfar.py -v`

Expected: both tests pass.

- [ ] **Step 4: Write failing local-maximum detection extraction test**

Append to `tests/processing/test_cfar.py`:

```python
from radarsim.models import CFARResult, RangeDopplerResult
from radarsim.processing.cfar import extract_detections

def test_extraction_keeps_local_peak_and_reports_physical_coordinates():
    power = np.zeros((1, 1, 5, 6))
    power[0, 0, 2, 3] = 20
    power[0, 0, 2, 4] = 10
    mask = power[0, 0] > 0
    cfar = CFARResult(mask, np.ones((5, 6))*2, np.ones((5, 6)))
    rd = RangeDopplerResult(
        spectrum=np.sqrt(power).astype(complex), power_w=power,
        range_m=np.arange(6)*10, velocity_mps=np.arange(5)-2,
        range_resolution_m=10, velocity_resolution_mps=1,
        max_unambiguous_range_m=50, max_unambiguous_velocity_mps=2)
    detections = extract_detections(rd, [[cfar]])
    assert len(detections) == 1
    assert detections[0].range_m == 30
    assert detections[0].velocity_mps == 0
    assert detections[0].snr_db == pytest.approx(10*np.log10(20))
```

- [ ] **Step 5: Run extraction test and observe the missing interface**

Run: `python -m pytest tests/processing/test_cfar.py::test_extraction_keeps_local_peak_and_reports_physical_coordinates -v`

Expected: fails because `Detection` or `extract_detections` is missing.

- [ ] **Step 6: Add `Detection` and local-maximum extraction**

Append to `src/radarsim/models.py`:

```python
@dataclass(frozen=True)
class Detection:
    frame_index: int
    channel_index: int
    range_m: float
    velocity_mps: float
    power_w: float
    noise_w: float
    snr_db: float
```

Append to `src/radarsim/processing/cfar.py`:

```python
from radarsim.models import Detection

def extract_detections(rd, cfar_by_frame_channel):
    output = []
    for frame in range(rd.power_w.shape[0]):
        for channel in range(rd.power_w.shape[1]):
            power = rd.power_w[frame, channel]
            cfar = cfar_by_frame_channel[frame][channel]
            peaks = cfar.detections & (power == maximum_filter(power, size=3))
            for doppler_index, range_index in np.argwhere(peaks):
                noise = cfar.noise_w[doppler_index, range_index]
                value = power[doppler_index, range_index]
                output.append(Detection(
                    frame, channel, float(rd.range_m[range_index]),
                    float(rd.velocity_mps[doppler_index]), float(value),
                    float(noise), float(10*np.log10(value/noise))))
    return tuple(output)
```

Run: `python -m pytest tests/processing/test_cfar.py -v`

Expected: all CFAR and extraction tests pass.

- [ ] **Step 7: Run all tests and commit detection**

Run: `python -m pytest -q`

Expected: all tests pass.

```powershell
git add src/radarsim/models.py src/radarsim/processing tests/processing/test_cfar.py
git commit -m "feat: add CA-CFAR detection"
```

---

### Task 8: End-to-End Simulation Orchestration

**Files:**
- Modify: `src/radarsim/models.py`
- Create: `src/radarsim/simulation.py`
- Modify: `src/radarsim/__init__.py`
- Create: `tests/test_simulation.py`
- Create: `tests/physics/test_peak_localization.py`

**Interfaces:**
- Consumes: every completed model, waveform, channel, frontend, range-Doppler, and CFAR interface.
- Produces: `ExperimentConfig`, `TruthRecord`, `SimulationResult`; `run_processing(rx, tx, waveform, radar, processing)`; `run_simulation(config)`.

- [ ] **Step 1: Write a failing FMCW end-to-end shape test**

```python
# tests/test_simulation.py
import inspect
import numpy as np
import pytest
from radarsim.config import CFARConfig, NoiseConfig, ProcessingConfig, RadarConfig
from radarsim.scene import CartesianWaypoint, Target, Trajectory
from radarsim.simulation import ExperimentConfig, run_processing, run_simulation
from radarsim.waveforms.fmcw import FMCWConfig
from radarsim.waveforms.lfm import LFMConfig

def target_at(range_m, speed_mps=0):
    tr = Trajectory.from_cartesian([
        CartesianWaypoint(0, (range_m, 0, 0)),
        CartesianWaypoint(1, (range_m + speed_mps, 0, 0)),
    ])
    return Target(f"t{range_m}", f"target-{range_m}", 20, tr)

def test_fmcw_simulation_preserves_axes_and_returns_finite_results():
    config = ExperimentConfig(
        radar=RadarConfig(carrier_hz=10e9, transmit_power_w=1e6),
        waveform=FMCWConfig(20e6, 40e-6, 50e-6, 10e6, 16, 1),
        targets=(target_at(30, 2),),
        noise=NoiseConfig(0),
        processing=ProcessingConfig(
            range_fft_size=512, doppler_fft_size=16,
            cfar=CFARConfig((2, 2), (1, 1), 1e-3)),
    )
    result = run_simulation(config)
    assert result.tx.data.shape[:3] == (1, 16, 1)
    assert result.rx.data.shape[:3] == (1, 16, 1)
    assert result.range_doppler.power_w.shape[:3] == (1, 1, 16)
    assert np.isfinite(result.range_doppler.power_w).all()
    assert len(result.truth) == 1

def test_lfm_simulation_runs_the_same_result_contract():
    config = ExperimentConfig(
        RadarConfig(carrier_hz=10e9, transmit_power_w=1e6),
        LFMConfig(2e6, 8e-6, 80e-6, 4e6, 16, 1),
        (target_at(300, -1),), NoiseConfig(0),
        ProcessingConfig(
            doppler_fft_size=16,
            cfar=CFARConfig((2, 2), (1, 1), 1e-3)))
    result = run_simulation(config)
    assert result.rx.dimensions == ("frame", "slow_time", "rx", "fast_time")
    assert result.range_doppler.spectrum.ndim == 4

def test_truth_outside_unambiguous_range_emits_warning():
    config = ExperimentConfig(
        RadarConfig(transmit_power_w=1e6),
        FMCWConfig(20e6, 40e-6, 50e-6, 1e6, 8, 1),
        (target_at(10_000),), NoiseConfig(0),
        ProcessingConfig(
            doppler_fft_size=8,
            cfar=CFARConfig((2, 2), (1, 1), 1e-3)))
    with pytest.warns(RuntimeWarning, match="range"):
        run_simulation(config)

def test_processing_boundary_does_not_accept_target_truth():
    assert tuple(inspect.signature(run_processing).parameters) == (
        "rx", "tx", "waveform", "radar", "processing")
```

Create `tests/physics/test_peak_localization.py` in the same red step:

```python
import numpy as np
from radarsim.config import NoiseConfig, ProcessingConfig, RadarConfig
from radarsim.scene import CartesianWaypoint, Target, Trajectory
from radarsim.simulation import ExperimentConfig, run_simulation
from radarsim.waveforms.fmcw import FMCWConfig
from radarsim.waveforms.lfm import LFMConfig

def moving_target(initial_range, speed):
    return Target("single", "single", 20, Trajectory.from_cartesian([
        CartesianWaypoint(0, (initial_range, 0, 0)),
        CartesianWaypoint(1, (initial_range + speed, 0, 0)),
    ]))

def assert_within_one_resolution_cell(result):
    d, r = np.unravel_index(np.argmax(result.range_doppler.power_w[0, 0]),
                            result.range_doppler.power_w[0, 0].shape)
    truth = result.truth[0]
    measured_range = result.range_doppler.range_m[r]
    measured_velocity = result.range_doppler.velocity_mps[d]
    assert abs(measured_range - truth.range_m) <= result.range_doppler.range_resolution_m
    assert abs(measured_velocity - truth.radial_velocity_mps) <= (
        result.range_doppler.velocity_resolution_mps)

def test_fmcw_no_noise_single_target_peak_is_within_one_cell():
    result = run_simulation(ExperimentConfig(
        RadarConfig(carrier_hz=10e9, transmit_power_w=1e9),
        FMCWConfig(50e6, 40e-6, 50e-6, 10e6, 64, 1),
        (moving_target(150, 4),), NoiseConfig(0),
        ProcessingConfig(range_fft_size=512, doppler_fft_size=64)))
    assert_within_one_resolution_cell(result)

def test_lfm_no_noise_single_target_peak_is_within_one_cell():
    result = run_simulation(ExperimentConfig(
        RadarConfig(carrier_hz=10e9, transmit_power_w=1e9),
        LFMConfig(5e6, 10e-6, 100e-6, 10e6, 64, 1),
        (moving_target(1200, 3),), NoiseConfig(0),
        ProcessingConfig(doppler_fft_size=64)))
    assert_within_one_resolution_cell(result)
```

- [ ] **Step 2: Run the test and observe the missing simulation module**

Run: `python -m pytest tests/test_simulation.py tests/physics/test_peak_localization.py -v`

Expected: collection fails because `radarsim.simulation` does not exist.

- [ ] **Step 3: Add orchestration result models**

Append to `src/radarsim/models.py`:

```python
@dataclass(frozen=True)
class TruthRecord:
    frame_index: int
    target_id: str
    time_s: float
    x_m: float
    y_m: float
    z_m: float
    range_m: float
    radial_velocity_mps: float
    azimuth_deg: float
    elevation_deg: float
    rcs_dbsm: float

@dataclass(frozen=True)
class SimulationResult:
    tx: SignalCube
    rx: SignalCube
    frontend: SignalCube | LFMFrontendResult
    range_doppler: RangeDopplerResult
    cfar: tuple[tuple[CFARResult, ...], ...]
    detections: tuple[Detection, ...]
    truth: tuple[TruthRecord, ...]
```

- [ ] **Step 4: Implement processing-only and full simulation functions**

Create `src/radarsim/simulation.py`:

```python
from dataclasses import dataclass
import warnings
import numpy as np
from .channel import MonostaticChannel
from .config import NoiseConfig, ProcessingConfig, RadarConfig
from .frontend import dechirp_fmcw, prepare_lfm
from .geometry import cartesian_to_polar, radial_velocity
from .models import SimulationResult, TruthRecord
from .processing.cfar import ca_cfar_2d, extract_detections
from .processing.range_doppler import process_fmcw, process_lfm
from .scene import Target
from .waveforms.fmcw import FMCWConfig, FMCWWaveform
from .waveforms.lfm import LFMConfig, LFMWaveform

@dataclass(frozen=True)
class ExperimentConfig:
    radar: RadarConfig
    waveform: FMCWConfig | LFMConfig
    targets: tuple[Target, ...]
    noise: NoiseConfig
    processing: ProcessingConfig

def _waveform(config):
    if isinstance(config, FMCWConfig): return FMCWWaveform(config)
    if isinstance(config, LFMConfig): return LFMWaveform(config)
    raise TypeError("waveform must be FMCWConfig or LFMConfig")

def run_processing(rx, tx, waveform, radar, processing):
    if isinstance(waveform, FMCWWaveform):
        frontend = dechirp_fmcw(rx, tx)
        rd = process_fmcw(frontend, radar, waveform.config, processing)
    else:
        frontend = prepare_lfm(rx, waveform)
        rd = process_lfm(frontend, radar, waveform.config, processing)
    cfar = tuple(tuple(
        ca_cfar_2d(rd.power_w[frame, channel], processing.cfar)
        for channel in range(rd.power_w.shape[1]))
        for frame in range(rd.power_w.shape[0]))
    detections = extract_detections(rd, cfar)
    return frontend, rd, cfar, detections

def _truth_records(config, tx):
    records = []
    frame_span = tx.sample_times_s.shape[1] * float(
        tx.metadata["repetition_interval_s"])
    for frame in range(tx.data.shape[0]):
        time_s = frame * frame_span + frame_span / 2
        for target in config.targets:
            state = target.trajectory.state_at(time_s)
            relative = state.position_m - np.asarray(config.radar.position_m)
            range_m, azimuth, elevation = cartesian_to_polar(relative)
            records.append(TruthRecord(
                frame, target.target_id, time_s, *state.position_m,
                range_m, radial_velocity(relative, state.velocity_mps),
                azimuth, elevation, target.rcs_dbsm))
    return tuple(records)

def _warn_for_ambiguity(truth, rd):
    if any(item.range_m > rd.max_unambiguous_range_m for item in truth):
        warnings.warn("target range exceeds maximum unambiguous range", RuntimeWarning)
    if any(abs(item.radial_velocity_mps) > rd.max_unambiguous_velocity_mps
           for item in truth):
        warnings.warn("target speed exceeds maximum unambiguous velocity", RuntimeWarning)

def run_simulation(config):
    waveform = _waveform(config.waveform)
    tx = waveform.build_tx(config.radar)
    rx = MonostaticChannel().propagate(
        tx, waveform, config.radar, config.targets, config.noise)
    frontend, rd, cfar, detections = run_processing(
        rx, tx, waveform, config.radar, config.processing)
    truth = _truth_records(config, tx)
    _warn_for_ambiguity(truth, rd)
    return SimulationResult(tx, rx, frontend, rd, cfar, detections, truth)
```

Export `ExperimentConfig` and `run_simulation` from `src/radarsim/__init__.py`.

Run: `python -m pytest tests/test_simulation.py tests/physics/test_peak_localization.py -v`

Expected: all orchestration, boundary, warning, and one-resolution-cell tests pass.

- [ ] **Step 5: Run all tests and commit orchestration**

Run: `python -m pytest -q`

Expected: all tests pass.

```powershell
git add src/radarsim/models.py src/radarsim/simulation.py src/radarsim/__init__.py tests/test_simulation.py tests/physics/test_peak_localization.py
git commit -m "feat: orchestrate full radar simulation"
```

---

### Task 9: Reproducible Artifact Output and Diagnostic Plots

**Files:**
- Create: `src/radarsim/output.py`
- Modify: `src/radarsim/simulation.py`
- Create: `tests/test_output.py`

**Interfaces:**
- Consumes: `SimulationResult`, `ExperimentConfig`, `OutputConfig`.
- Produces: `save_simulation(result, config, output) -> pathlib.Path`; `run_and_save(config) -> tuple[SimulationResult, pathlib.Path]`.

- [ ] **Step 1: Write a failing artifact-manifest test**

```python
# tests/test_output.py
import json
import numpy as np
from radarsim.config import CFARConfig, NoiseConfig, OutputConfig, ProcessingConfig, RadarConfig
from radarsim.output import save_simulation
from radarsim.scene import CartesianWaypoint, Target, Trajectory
from radarsim.simulation import ExperimentConfig, run_simulation
from radarsim.waveforms.fmcw import FMCWConfig

def small_result():
    trajectory = Trajectory.from_cartesian([
        CartesianWaypoint(0, (30, 0, 0)),
        CartesianWaypoint(1, (31, 0, 0)),
    ])
    config = ExperimentConfig(
        RadarConfig(transmit_power_w=1e6),
        FMCWConfig(10e6, 20e-6, 25e-6, 5e6, 8, 1),
        (Target("t1", "target", 10, trajectory),),
        NoiseConfig(0), ProcessingConfig(
            range_fft_size=128, doppler_fft_size=8,
            cfar=CFARConfig((2, 2), (1, 1), 1e-3)))
    return config, run_simulation(config)

def test_save_simulation_writes_complete_manifest(tmp_path):
    config, result = small_result()
    directory = save_simulation(
        result, config, OutputConfig("smoke", tmp_path, save_raw_data=True))
    expected = {
        "config.json", "truth.csv", "detections.csv", "raw_data.npz",
        "range_profile.png", "range_doppler.png", "cfar_map.png",
    }
    assert {path.name for path in directory.iterdir()} == expected
    snapshot = json.loads((directory / "config.json").read_text("utf-8"))
    assert snapshot["waveform"]["bandwidth_hz"] == 10e6
    assert snapshot["output"]["experiment_name"] == "smoke"
    arrays = np.load(directory / "raw_data.npz")
    assert arrays["rx"].shape == result.rx.data.shape
```

- [ ] **Step 2: Run the output test and observe the missing module**

Run: `python -m pytest tests/test_output.py::test_save_simulation_writes_complete_manifest -v`

Expected: collection fails because `radarsim.output` does not exist.

- [ ] **Step 3: Implement lossless tabular/array output**

Create `src/radarsim/output.py` with serialization helpers:

```python
import csv
import json
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def _jsonable(value):
    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name))
                for field in fields(value)}
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, (tuple, list)): return [_jsonable(item) for item in value]
    if isinstance(value, dict): return {str(k): _jsonable(v) for k, v in value.items()}
    if hasattr(value, "__dict__"):
        return {k: _jsonable(v) for k, v in vars(value).items()
                if not k.startswith("_")}
    if isinstance(value, np.generic): return value.item()
    return value

def _write_rows(path, rows):
    records = [asdict(row) for row in rows]
    if not records:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

def save_simulation(result, config, output):
    directory = Path(output.output_root) / output.experiment_name
    directory.mkdir(parents=True, exist_ok=True)
    config_snapshot = _jsonable(config)
    config_snapshot["output"] = _jsonable(output)
    (directory / "config.json").write_text(
        json.dumps(config_snapshot, indent=2, ensure_ascii=False),
        encoding="utf-8")
    _write_rows(directory / "truth.csv", result.truth)
    _write_rows(directory / "detections.csv", result.detections)
    np.savez_compressed(
        directory / "raw_data.npz",
        tx=result.tx.data, rx=result.rx.data,
        range_doppler=result.range_doppler.spectrum,
        range_m=result.range_doppler.range_m,
        velocity_mps=result.range_doppler.velocity_mps)
    _save_plots(directory, result)
    return directory
```

- [ ] **Step 4: Implement plots with physical axes**

Append to `src/radarsim/output.py`:

```python
def _db(power):
    floor = np.finfo(float).tiny
    return 10*np.log10(np.maximum(power, floor))

def _save_plots(directory, result):
    rd = result.range_doppler
    power = rd.power_w[0, 0]

    fig, ax = plt.subplots()
    ax.plot(rd.range_m, _db(power.max(axis=0)))
    ax.set(xlabel="Range (m)", ylabel="Power (dB)", title="Range profile")
    fig.tight_layout()
    fig.savefig(directory / "range_profile.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    image = ax.pcolormesh(rd.range_m, rd.velocity_mps, _db(power), shading="auto")
    ax.set(xlabel="Range (m)", ylabel="Radial velocity (m/s)",
           title="Range-Doppler map")
    fig.colorbar(image, ax=ax, label="Power (dB)")
    fig.tight_layout()
    fig.savefig(directory / "range_doppler.png", dpi=150)
    plt.close(fig)

    cfar = result.cfar[0][0]
    fig, ax = plt.subplots()
    image = ax.pcolormesh(rd.range_m, rd.velocity_mps,
                          cfar.detections.astype(int), shading="auto")
    ax.set(xlabel="Range (m)", ylabel="Radial velocity (m/s)",
           title="CFAR detections")
    fig.colorbar(image, ax=ax, label="Detection")
    fig.tight_layout()
    fig.savefig(directory / "cfar_map.png", dpi=150)
    plt.close(fig)
```

Run: `python -m pytest tests/test_output.py::test_save_simulation_writes_complete_manifest -v`

Expected: pass and all PNG files have non-zero size.

- [ ] **Step 5: Add explicit behavior for disabled raw-data output**

Append to `tests/test_output.py`:

```python
def test_save_simulation_can_skip_raw_data(tmp_path):
    config, result = small_result()
    directory = save_simulation(
        result, config, OutputConfig("no-raw", tmp_path, save_raw_data=False))
    assert not (directory / "raw_data.npz").exists()
    assert (directory / "range_doppler.png").stat().st_size > 0
```

Run: `python -m pytest tests/test_output.py::test_save_simulation_can_skip_raw_data -v`

Expected: fail because Step 3 creates `raw_data.npz` unconditionally.

- [ ] **Step 6: Implement the tested raw-data switch**

Replace the unconditional `np.savez_compressed` call in `save_simulation` with:

```python
    if output.save_raw_data:
        np.savez_compressed(
            directory / "raw_data.npz",
            tx=result.tx.data, rx=result.rx.data,
            range_doppler=result.range_doppler.spectrum,
            range_m=result.range_doppler.range_m,
            velocity_mps=result.range_doppler.velocity_mps)
```

Run: `python -m pytest tests/test_output.py::test_save_simulation_can_skip_raw_data -v`

Expected: pass.

- [ ] **Step 7: Add `run_and_save` without coupling processing to output**

Append to `src/radarsim/simulation.py`:

```python
def run_and_save(config, output):
    from .output import save_simulation
    result = run_simulation(config)
    return result, save_simulation(result, config, output)
```

Run: `python -m pytest tests/test_output.py -v`

Expected: all output tests pass.

- [ ] **Step 8: Run all tests and commit artifact output**

Run: `python -m pytest -q`

Expected: all tests pass.

```powershell
git add src/radarsim/output.py src/radarsim/simulation.py tests/test_output.py
git commit -m "feat: save simulation artifacts"
```

---

### Task 10: Editable Configs, Executable Examples, and Documentation

**Files:**
- Create: `configs/__init__.py`
- Create: `configs/fmcw_example.py`
- Create: `configs/lfm_example.py`
- Create: `examples/run_fmcw.py`
- Create: `examples/run_lfm.py`
- Create: `tests/integration/test_examples.py`
- Create: `README.md`

**Interfaces:**
- Consumes: `ExperimentConfig`, `OutputConfig`, `run_simulation`, `run_and_save`.
- Produces: editable `CONFIG` and `OUTPUT` objects for both waveform families; executable example scripts; researcher documentation.

- [ ] **Step 1: Write failing subprocess tests for both missing examples**

```python
# tests/integration/test_examples.py
import os
import subprocess
import sys
from pathlib import Path

import pytest

@pytest.mark.parametrize(
    ("script", "experiment"),
    [("run_fmcw.py", "fmcw_example"), ("run_lfm.py", "lfm_example")],
)
def test_example_script_runs_and_writes_artifacts(script, experiment, tmp_path):
    root = Path(__file__).parents[2]
    env = os.environ.copy()
    env["RADARSIM_OUTPUT_ROOT"] = str(tmp_path)
    completed = subprocess.run(
        [sys.executable, str(root / "examples" / script)],
        cwd=root, env=env, text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert "Saved" in completed.stdout
    directory = tmp_path / experiment
    assert {path.name for path in directory.iterdir()} == {
        "config.json", "truth.csv", "detections.csv", "raw_data.npz",
        "range_profile.png", "range_doppler.png", "cfar_map.png",
    }
```

- [ ] **Step 2: Run the integration tests and confirm examples are absent**

Run: `python -m pytest tests/integration/test_examples.py -v`

Expected: both cases fail because `examples/run_fmcw.py` and `examples/run_lfm.py` do not exist.

- [ ] **Step 3: Add editable FMCW and LFM configs**

Create `configs/fmcw_example.py`:

```python
import os
from pathlib import Path
from radarsim.config import CFARConfig, NoiseConfig, OutputConfig, ProcessingConfig, RadarConfig
from radarsim.scene import CartesianWaypoint, PolarWaypoint, Target, Trajectory
from radarsim.simulation import ExperimentConfig
from radarsim.waveforms.fmcw import FMCWConfig

CONFIG = ExperimentConfig(
    radar=RadarConfig(carrier_hz=10e9, transmit_power_w=1e9,
                      tx_gain_db=20, rx_gain_db=20),
    waveform=FMCWConfig(20e6, 40e-6, 50e-6, 5e6, 64, 1),
    targets=(
        Target("fmcw-1", "receding", 10, Trajectory.from_cartesian([
            CartesianWaypoint(0, (120, 20, 5)),
            CartesianWaypoint(1, (128, 20, 5)),
        ])),
        Target("fmcw-2", "approaching", 5, Trajectory.from_polar([
            PolarWaypoint(0, 250, -20, 3),
            PolarWaypoint(1, 240, -20, 3),
        ])),
    ),
    noise=NoiseConfig(noise_power_w=1e-14, seed=20260915),
    processing=ProcessingConfig(
        range_fft_size=512, doppler_fft_size=64,
        cfar=CFARConfig((6, 4), (2, 1), 1e-4)),
)
OUTPUT = OutputConfig(
    "fmcw_example",
    Path(os.environ.get("RADARSIM_OUTPUT_ROOT", "outputs")),
)
```

Create `configs/lfm_example.py`:

```python
import os
from pathlib import Path
from radarsim.config import CFARConfig, NoiseConfig, OutputConfig, ProcessingConfig, RadarConfig
from radarsim.scene import CartesianWaypoint, PolarWaypoint, Target, Trajectory
from radarsim.simulation import ExperimentConfig
from radarsim.waveforms.lfm import LFMConfig

CONFIG = ExperimentConfig(
    radar=RadarConfig(carrier_hz=10e9, transmit_power_w=1e9,
                      tx_gain_db=25, rx_gain_db=25),
    waveform=LFMConfig(5e6, 10e-6, 100e-6, 10e6, 64, 1),
    targets=(
        Target("lfm-1", "receding", 15, Trajectory.from_cartesian([
            CartesianWaypoint(0, (1200, 100, 20)),
            CartesianWaypoint(1, (1210, 100, 20)),
        ])),
        Target("lfm-2", "approaching", 10, Trajectory.from_polar([
            PolarWaypoint(0, 2500, 15, 5),
            PolarWaypoint(1, 2485, 15, 5),
        ])),
    ),
    noise=NoiseConfig(noise_power_w=1e-15, seed=20260915),
    processing=ProcessingConfig(
        doppler_fft_size=64,
        cfar=CFARConfig((6, 4), (2, 1), 1e-4)),
)
OUTPUT = OutputConfig(
    "lfm_example",
    Path(os.environ.get("RADARSIM_OUTPUT_ROOT", "outputs")),
)
```

- [ ] **Step 4: Add executable scripts**

```python
# examples/run_fmcw.py
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for import_root in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from configs.fmcw_example import CONFIG, OUTPUT
from radarsim.simulation import run_and_save

if __name__ == "__main__":
    result, directory = run_and_save(CONFIG, OUTPUT)
    print(f"Saved {len(result.detections)} detections to {directory}")
```

```python
# examples/run_lfm.py
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for import_root in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from configs.lfm_example import CONFIG, OUTPUT
from radarsim.simulation import run_and_save

if __name__ == "__main__":
    result, directory = run_and_save(CONFIG, OUTPUT)
    print(f"Saved {len(result.detections)} detections to {directory}")
```

- [ ] **Step 5: Run example integration tests**

Run: `python -m pytest tests/integration/test_examples.py -v`

Expected: both subprocesses exit 0 and write their experiment directories under pytest's temporary directory.

- [ ] **Step 6: Document setup, conventions, and configuration**

Create `README.md` with these sections and commands:

````markdown
# RadarSimulate

Python 3.11+ complex-baseband radar simulation for algorithm research.

## Install

```powershell
python -m pip install -e ".[test]"
```

## Run

```powershell
python examples/run_fmcw.py
python examples/run_lfm.py
```

Edit `configs/fmcw_example.py` or `configs/lfm_example.py` to change radar,
waveform, target trajectory, RCS, noise, FFT, and CFAR parameters.

## Coordinates

Azimuth 0° is +X, azimuth -90° is +Y, elevation +90° is +Z. Positive
radial velocity means receding from the radar.

## Data shapes

Tx: `[frame, slow_time, tx, fast_time]`

Rx: `[frame, slow_time, rx, fast_time]`

The initial release validates 1T1R while retaining both channel axes.
````

Also list the seven generated artifact names and link to the approved design document.

- [ ] **Step 7: Run the complete acceptance suite**

Run: `python -m pytest -q`

Expected: all unit, physics, output, and integration tests pass with no unexpected warnings.

Run: `python examples/run_fmcw.py`

Expected: exits 0 and creates all seven files under `outputs/fmcw_example/`.

Run: `python examples/run_lfm.py`

Expected: exits 0 and creates all seven files under `outputs/lfm_example/`.

- [ ] **Step 8: Commit examples and acceptance coverage**

```powershell
git add README.md configs examples tests/integration/test_examples.py
git commit -m "feat: add runnable radar simulation examples"
```

---

## Final Verification

- [ ] Run: `python -m pytest -q`

Expected: zero failures and no unexpected warnings.

- [ ] Run: `git status --short`

Expected: no output.

- [ ] Confirm both example output directories contain `config.json`, `truth.csv`, `detections.csv`, `raw_data.npz`, `range_profile.png`, `range_doppler.png`, and `cfar_map.png`.

- [ ] Compare the implementation against every acceptance criterion in the design spec before claiming completion.
