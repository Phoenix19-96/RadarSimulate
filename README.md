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

The scripts run small, deterministic FMCW and pulsed-LFM two-target scenes. They
print the detection count and artifact directory. Each creates its output only when
run: `outputs/fmcw_example/` or `outputs/lfm_example/` by default. Set
`RADARSIM_OUTPUT_ROOT` to write a one-off run elsewhere.

Output directories are immutable: running an example again with the same
destination raises `FileExistsError`. Choose a new `OUTPUT.experiment_name` or a
new `RADARSIM_OUTPUT_ROOT` for each saved run.

## Configuration

Edit `CONFIG` and `OUTPUT` directly in `configs/fmcw_example.py` or
`configs/lfm_example.py`. The Python dataclasses expose radar, waveform, target
trajectory, constant RCS, noise, FFT, and CFAR settings. This initial release has
no GUI, command-line configuration system, or external configuration format.

Each target trajectory is piecewise-linear in Cartesian space. Waypoints may be
entered in Cartesian or polar coordinates; RCS is constant for the full run.

`range_window` applies only to FMCW; LFM always uses its waveform-matched pulse
reference and ignores this setting. `range_fft_size` controls the FMCW delay-grid
density and must be `None` for LFM (explicit LFM values are rejected).
`doppler_window` and `doppler_fft_size` apply to both waveforms.

FMCW matches the dechirped current and previous chirp branches coherently, with
fast-time Doppler compensation and equal-energy gated references. Its supported
range is limited by both IF sampling (`c * Fs / (4 * slope)`) and acquired delay
support. When previous-chirp support bridges the acquisition gap, the latter
limit is `c * repetition_interval / 2`; otherwise it is the range of the last
acquired fast-time sample. The first chirp has no artificial transmit prehistory.
Acquisition-limited grids use range-cell centers, so the zero and repetition
endpoints cannot represent the same cell. Increasing `range_fft_size` makes the
grid denser without improving the nominal `c / (2 * bandwidth)` resolution.

LFM matches the continuous receive record across adjacent PRI/frame boundaries
and assigns each nonnegative delay to its original transmitted pulse. A final
echo without a complete recorded reference span contributes zero; a single-pulse
frame exposes only range gates with complete support. The usual multi-pulse
range limit remains `c * PRI / 2`.

Scenes at or beyond the supported range produce a warning before Tx allocation;
their output may alias or contain no acquired echo. Small supported range maps
may require smaller CFAR windows. Single-sample inputs are degenerate processing
cases and cannot provide a resolved range or velocity estimate.

## Coordinates

The radar is at the origin and points toward +X. Azimuth 0° is +X, azimuth -90°
is +Y, and elevation +90° is +Z. Positive radial velocity means receding from the
radar. Range is three-dimensional slant range.

## Signal chain and data shapes

The explicit complex-baseband chain is configuration → waveform → transmitter →
scene/propagation → receiver frontend → range/Doppler processing → 2-D CA-CFAR →
detections and saved artifacts. Processing consumes signal products and metadata,
not scene truth.

Tx data: `[frame, slow_time, tx, fast_time]`

Rx data: `[frame, slow_time, rx, fast_time]`

The initial release validates 1T1R while retaining both channel axes. This keeps
the data contracts ready for future Tx scheduling, Rx arrays, MIMO coding/decoding,
and virtual-channel processing; those capabilities are not implemented yet.

## Outputs

Every example writes exactly these seven diagnostic artifacts:

- `config.json` — configuration and signal-metadata snapshot
- `truth.csv` — target truth at frame reference times
- `detections.csv` — CFAR-selected local peaks
- `raw_data.npz` — Tx, Rx, range-Doppler, and axes arrays
- `range_profile.png` — maximum-over-Doppler range profile
- `range_doppler.png` — range-Doppler power map
- `cfar_map.png` — CFAR detection map

Plots and `power_w` report uncalibrated processing-domain power, including
matched-filter/window and coherent Doppler gains. The dBW scale is relative to
one watt in these processing arrays; it is not a calibrated receiver-input power
estimate and should not be compared across processing settings without gain
correction.

## Testing

```powershell
python -m pytest -q
```

The tests cover configuration and scene validation, waveform and propagation
physics, frontends, range-Doppler/CFAR processing, artifact persistence, and the
two executable examples.

## Limitations

This is a monostatic 1T1R research simulator. It does not include array angle
estimation or beamforming, MIMO scheduling/coding/decoding, fluctuating RCS,
antenna patterns, multipath, occlusion, clutter, atmospheric loss, range-migration
correction, advanced parameter estimation, real-time processing, or a GUI/Web
interface. It uses a complex-baseband equivalent model rather than direct GHz
carrier sampling.

See the [approved design](docs/superpowers/specs/2026-09-15-radar-full-stack-simulation-design.md)
for the complete conventions, interfaces, and scope.
