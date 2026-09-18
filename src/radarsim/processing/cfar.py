import numpy as np
from scipy.ndimage import maximum_filter
from scipy.signal import convolve2d

from radarsim.config import CFARConfig
from radarsim.models import CFARResult, Detection, RangeDopplerResult


def ca_cfar_2d(power_w, config: CFARConfig) -> CFARResult:
    power = np.asarray(power_w, dtype=float)
    if power.ndim != 2:
        raise ValueError("power_w must be a two-dimensional Doppler-range map")
    if not np.all(np.isfinite(power)):
        raise ValueError("power_w must be finite")
    if np.any(power < 0):
        raise ValueError("power_w must be non-negative")
    train_doppler, train_range = config.training_cells
    guard_doppler, guard_range = config.guard_cells
    outer_doppler = train_doppler + guard_doppler
    outer_range = train_range + guard_range
    if power.shape[0] <= 2 * outer_doppler or power.shape[1] <= 2 * outer_range:
        raise ValueError("CFAR window does not fit the power map")
    kernel = np.ones((2 * outer_doppler + 1, 2 * outer_range + 1), dtype=float)
    kernel[outer_doppler - guard_doppler:outer_doppler + guard_doppler + 1,
           outer_range - guard_range:outer_range + guard_range + 1] = 0
    count = int(kernel.sum())
    if count <= 0:
        raise ValueError("CFAR window must contain training cells")
    noise = convolve2d(power, kernel, mode="same", boundary="fill") / count
    alpha = count * (config.false_alarm_rate ** (-1 / count) - 1)
    threshold = alpha * noise
    valid = np.zeros(power.shape, dtype=bool)
    valid[outer_doppler:power.shape[0] - outer_doppler,
          outer_range:power.shape[1] - outer_range] = True
    threshold[~valid] = np.nan
    noise[~valid] = np.nan
    detections = valid & (power > threshold)
    return CFARResult(detections, threshold, noise)


def extract_detections(rd: RangeDopplerResult, cfar_by_frame_channel) -> tuple[Detection, ...]:
    if not isinstance(rd, RangeDopplerResult):
        raise TypeError("rd must be a RangeDopplerResult")
    frames, channels, _, _ = rd.power_w.shape
    try:
        if len(cfar_by_frame_channel) != frames or any(len(row) != channels for row in cfar_by_frame_channel):
            raise ValueError("CFAR results must match frame and channel dimensions")
    except TypeError as exc:
        raise ValueError("CFAR results must be nested by frame and channel") from exc
    output: list[Detection] = []
    for frame in range(frames):
        for channel in range(channels):
            power = rd.power_w[frame, channel]
            cfar = cfar_by_frame_channel[frame][channel]
            if not isinstance(cfar, CFARResult) or cfar.detections.shape != power.shape:
                raise ValueError("CFAR result shape must match range-Doppler map")
            peaks = cfar.detections & (power == maximum_filter(power, size=3, mode="nearest"))
            for doppler_index, range_index in np.argwhere(peaks):
                value = float(power[doppler_index, range_index])
                noise = float(cfar.noise_w[doppler_index, range_index])
                snr_db = float(10 * np.log10(value / noise)) if noise > 0 else float("inf")
                output.append(Detection(
                    frame, channel, float(rd.range_m[range_index]),
                    float(rd.velocity_mps[doppler_index]), value, noise, snr_db,
                    int(doppler_index), int(range_index)))
    return tuple(output)
