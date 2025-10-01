import pytest
import numpy as np
from src.data_structures import Waypoint, Trajectory
from src.deconfliction_service import DeconflictionServer, MissionApprovalResult

# Unit Tests

def test_min_distance_between_segments():
    """Unit test for min_distance_between_segments function"""
    p1 = np.array([0, 0, 0])
    q1 = np.array([10, 0, 0])
    p2 = np.array([5, 5, 0])
    q2 = np.array([5, 5, 10])

    # Expected: shortest distance from segment1 to point (5,5,0) is 5
    from src.deconfliction_service import min_distance_between_segments
    dist = min_distance_between_segments(p1, q1, p2, q2)
    assert abs(dist - 5) < 1e-3, f"Distance mismatch: got {dist}"


def test_point_segment_distance_degenerate():
    """Test point-segment distance for degenerate zero-length segment"""
    from src.deconfliction_service import _point_segment_distance
    p = np.array([1, 1, 1])
    seg_p1 = np.array([1, 1, 1])
    seg_p2 = np.array([1, 1, 1])
    dist = _point_segment_distance(p, seg_p1, seg_p2)
    assert abs(dist) < 1e-6


# Integration Tests

def make_trajectory(drone_id, start_pos, end_pos, start_time=0, end_time=10):
    return Trajectory(
        drone_id,
        [
            Waypoint(np.array(start_pos), start_time),
            Waypoint(np.array(end_pos), end_time)
        ]
    )


def test_no_conflict():
    server = DeconflictionServer()
    traj1 = make_trajectory("drone1", [0,0,10], [10,0,10])
    traj2 = make_trajectory("drone2", [0,20,10], [10,20,10])

    result1 = server.submit_mission_for_approval(traj1, 5.0)
    assert result1.approved

    result2 = server.submit_mission_for_approval(traj2, 5.0)
    assert result2.approved


def test_with_collision():
    server = DeconflictionServer()
    traj1 = make_trajectory("droneA", [0,0,10], [10,0,10])
    traj2 = make_trajectory("droneB", [5,0,10], [15,0,10])  # Overlapping horizontally

    result1 = server.submit_mission_for_approval(traj1, 5.0)
    assert result1.approved

    result2 = server.submit_mission_for_approval(traj2, 5.0)
    assert not result2.approved
    assert len(result2.conflicts) > 0


def test_spawn_collision():
    server = DeconflictionServer()
    traj1 = make_trajectory("droneA", [0,0,10], [50,0,10])
    traj2 = make_trajectory("droneB", [2,0,10], [60,0,10])  # Spawn conflict, too close at start

    result1 = server.submit_mission_for_approval(traj1, 5.0)
    assert result1.approved

    result2 = server.submit_mission_for_approval(traj2, 5.0)
    assert not result2.approved
    assert result2.spawn_conflict


def test_multiple_conflicts():
    server = DeconflictionServer()
    traj1 = make_trajectory("drone1", [0,0,10], [10,0,10])
    traj2 = make_trajectory("drone2", [0,5,10], [10,5,10])
    traj3 = make_trajectory("drone3", [0,2,10], [10,2,10])  # Conflicts with drone1 and drone2

    result1 = server.submit_mission_for_approval(traj1, 5.0)
    assert result1.approved
    result2 = server.submit_mission_for_approval(traj2, 5.0)
    assert result2.approved
    result3 = server.submit_mission_for_approval(traj3, 5.0)
    assert not result3.approved
    assert len(result3.conflicts) >= 2

