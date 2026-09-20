import numpy as np
from numpy.typing import ArrayLike, NDArray

def normalize_azimuth_deg(angle_deg: float) -> float:
    return (float(angle_deg) + 180.0) % 360.0 - 180.0

def cartesian_to_polar(xyz_m: ArrayLike) -> tuple[float, float, float]:
    try:
        raw = np.asarray(xyz_m)
        if raw.dtype.kind in "OUS": raise ValueError("xyz_m must contain numeric coordinates")
    except (TypeError, ValueError) as exc:
        raise ValueError("xyz_m must contain numeric coordinates") from exc
    try:
        xyz = np.asarray(xyz_m, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("xyz_m must contain numeric coordinates") from exc
    if xyz.shape != (3,): raise ValueError("xyz_m must contain three coordinates")
    if not np.all(np.isfinite(xyz)): raise ValueError("xyz_m must contain finite coordinates")
    x, y, z = xyz
    scale = float(np.max(np.abs(xyz)))
    range_m = scale * float(np.sqrt(np.sum((xyz / scale) ** 2))) if scale else 0.0
    if not np.isfinite(range_m): raise ValueError("range is not representable")
    if range_m == 0.0: raise ValueError("angles are undefined at zero range")
    return range_m, normalize_azimuth_deg(np.degrees(np.arctan2(-y, x))), float(np.degrees(np.arctan2(z, np.hypot(x, y))))

def polar_to_cartesian(range_m: float, azimuth_deg: float, elevation_deg: float) -> NDArray[np.float64]:
    try:
        finite = all(np.isscalar(v) and np.isfinite(v) for v in (range_m, azimuth_deg, elevation_deg))
    except TypeError as exc:
        raise ValueError("range_m, azimuth_deg, and elevation_deg must be finite scalars") from exc
    if not finite: raise ValueError("range_m, azimuth_deg, and elevation_deg must be finite scalars")
    if range_m <= 0.0: raise ValueError("range_m must be positive")
    az, el = np.radians([azimuth_deg, elevation_deg])
    return np.array([range_m*np.cos(el)*np.cos(az), -range_m*np.cos(el)*np.sin(az), range_m*np.sin(el)], dtype=float)

def radial_velocity(position_m: ArrayLike, velocity_mps: ArrayLike) -> float:
    try:
        p = np.asarray(position_m, dtype=float); v = np.asarray(velocity_mps, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("position and velocity must be numeric") from exc
    if p.shape != (3,): raise ValueError("position_m must contain three coordinates")
    if v.shape != (3,): raise ValueError("velocity_mps must contain three components")
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(v)): raise ValueError("position and velocity must be finite")
    scale = float(np.max(np.abs(p)))
    r = scale * float(np.sqrt(np.sum((p / scale) ** 2))) if scale else 0.0
    if not np.isfinite(r): raise ValueError("range is not representable")
    if r == 0.0: raise ValueError("radial velocity is undefined at zero range")
    unit = p / scale / (r / scale)
    velocity_scale = float(np.max(np.abs(v)))
    projection = velocity_scale * float(np.dot(unit, v / velocity_scale)) if velocity_scale else 0.0
    if not np.isfinite(projection): raise ValueError("radial velocity is not representable")
    return projection
