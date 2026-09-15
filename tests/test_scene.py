import numpy as np
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

def test_final_waypoint_uses_left_segment_velocity_and_position():
    tr = Trajectory.from_cartesian([CartesianWaypoint(0, (0, 0, 0)), CartesianWaypoint(2, (4, 2, 0))])
    state = tr.state_at(2)
    assert state.position_m == pytest.approx((4, 2, 0))
    assert state.velocity_mps == pytest.approx((2, 1, 0))


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

def test_trajectory_copies_input_arrays_and_keeps_callers_writeable():
    times = np.array([0.0, 1.0]); positions = np.array([[1., 0., 0.], [2., 0., 0.]])
    tr = Trajectory(times, positions)
    assert times.flags.writeable and positions.flags.writeable
    times[0] = 99.; positions[0, 0] = 99.
    assert tr.state_at(0).position_m == pytest.approx((1, 0, 0))

def test_trajectory_rejects_nonfinite_velocity_slope():
    with pytest.raises(ValueError, match="velocity"):
        Trajectory([0., 1e-320], [[0, 0, 0], [1, 0, 0]])

def test_target_state_array_equality_is_safe():
    state = Trajectory.from_cartesian([CartesianWaypoint(0, (1, 0, 0)), CartesianWaypoint(1, (2, 0, 0))]).state_at(0)
    assert state == state

def test_numeric_string_scene_inputs_are_rejected():
    with pytest.raises(ValueError, match="time_s"):
        Trajectory.from_cartesian([CartesianWaypoint("0", (0, 0, 0)), CartesianWaypoint(1, (1, 0, 0))])

def test_numeric_string_polar_fields_and_positions_are_rejected():
    with pytest.raises(ValueError, match="time_s"):
        Trajectory.from_polar([PolarWaypoint("0", 1, 0, 0), PolarWaypoint(1, 2, 0, 0)])
    with pytest.raises(ValueError, match="position"):
        Trajectory.from_cartesian([CartesianWaypoint(0, ("1", 0, 0)), CartesianWaypoint(1, (2, 0, 0))])

def test_target_rejects_string_and_unrepresentable_rcs():
    tr = Trajectory.from_cartesian([CartesianWaypoint(0, (1, 0, 0)), CartesianWaypoint(1, (2, 0, 0))])
    with pytest.raises(ValueError, match="rcs_dbsm"):
        Target("id", "name", "10", tr)
    with pytest.raises(ValueError, match="rcs_dbsm"):
        Target("id", "name", 3090, tr)
    with pytest.raises(ValueError, match="rcs_dbsm"):
        Target("id", "name", -4000, tr)

def test_slope_uses_stable_endpoint_difference():
    tr = Trajectory([0.0, 1e308], [[-1e308, 0, 0], [1e308, 0, 0]])
    assert tr.state_at(1e308).velocity_mps == pytest.approx((2, 0, 0))

def test_slope_accepts_finite_nextafter_delta_and_tiny_dt():
    p0 = 1e308
    p1 = np.nextafter(p0, np.inf)
    tr = Trajectory([0.0, 2e-16], [[p0, 0, 0], [p1, 0, 0]])
    assert tr.state_at(2e-16).velocity_mps[0] == pytest.approx(9.9792015476736e307)

def test_slope_handles_time_endpoint_difference_overflow():
    tr = Trajectory([-1e308, 1e308], [[1, 0, 0], [2, 0, 0]])
    assert tr.state_at(1e308).velocity_mps[0] == pytest.approx(5e-309, abs=1e-310)
