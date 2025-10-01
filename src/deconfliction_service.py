# src/deconfliction_service.py

from typing import List, Dict, Tuple
import numpy as np
from .data_structures import Trajectory, Conflict


def _point_segment_distance(point: np.ndarray, seg_p1: np.ndarray, seg_p2: np.ndarray) -> float:
    """Calculates the shortest distance between a point and a line segment."""
    line_vec = seg_p2 - seg_p1
    point_vec = point - seg_p1

    line_len_sq = np.dot(line_vec, line_vec)
    if line_len_sq == 0.0:
        return np.linalg.norm(point_vec)

    t = np.dot(point_vec, line_vec) / line_len_sq
    t = np.clip(t, 0, 1)

    closest_point_on_segment = seg_p1 + t * line_vec
    return np.linalg.norm(point - closest_point_on_segment)


def min_distance_between_segments(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> float:
    """
    Calculates the minimum distance between two 3D line segments.
    This version is robust against degenerate cases (e.g., zero-length segments).
    """
    u = q1 - p1
    v = q2 - p2
    w = p1 - p2

    a = np.dot(u, u)  # Squared length of segment 1
    c = np.dot(v, v)  # Squared length of segment 2

    # Handle degenerate cases: one or both segments are points
    if a < 1e-7 and c < 1e-7: # Both are points
        return np.linalg.norm(w)
    if a < 1e-7: # Segment 1 is a point
        return _point_segment_distance(p1, p2, q2)
    if c < 1e-7: # Segment 2 is a point
        return _point_segment_distance(p2, p1, q1)

    # Standard algorithm for non-degenerate (skew) lines
    b = np.dot(u, v)
    d = np.dot(u, w)
    e = np.dot(v, w)
    D = a * c - b * b

    sc, sN, sD = D, D, D
    tc, tN, tD = D, D, D

    if D < 1e-7:
        sN = 0.0
        sD = 1.0
        tN = e
        tD = c
    else:
        sN = (b * e - c * d)
        tN = (a * e - b * d)
        if sN < 0.0:
            sN = 0.0
            tN = e
            tD = c
        elif sN > sD:
            sN = sD
            tN = e + b
            tD = c

    if tN < 0.0:
        tN = 0.0
        if -d < 0.0:
            sN = 0.0
        elif -d > a:
            sN = sD
        else:
            sN = -d
            sD = a
    elif tN > tD:
        tN = tD
        if (-d + b) < 0.0:
            sN = 0
        elif (-d + b) > a:
            sN = sD
        else:
            sN = (-d + b)
            sD = a

    sc = 0.0 if abs(sN) < 1e-7 else sN / sD
    tc = 0.0 if abs(tN) < 1e-7 else tN / tD

    dP = w + (sc * u) - (tc * v)
    return np.linalg.norm(dP)


class MissionApprovalResult:
    """Result of a mission approval request"""
    def __init__(self, drone_id: str, approved: bool, conflicts: List[Conflict]):
        self.drone_id = drone_id
        self.approved = approved
        self.conflicts = conflicts
        self.conflicting_drones = list(set([
            other_drone for conflict in conflicts 
            for other_drone in conflict.conflicting_drone_ids 
            if other_drone != drone_id
        ]))


class DeconflictionServer:
    """A centralized service to manage and validate drone trajectories sequentially."""

    def __init__(self):
        self.approved_trajectories: Dict[str, Trajectory] = {}
        self.rejected_trajectories: Dict[str, Trajectory] = {}
        self.approval_history: List[MissionApprovalResult] = []

    def clear_all_trajectories(self):
        """Clear all trajectories for a new scenario"""
        self.approved_trajectories.clear()
        self.rejected_trajectories.clear()
        self.approval_history.clear()

    def submit_mission_for_approval(self, trajectory: Trajectory, safety_buffer: float) -> MissionApprovalResult:
        """
        Submit a mission for sequential approval against all previously approved trajectories.
        Returns MissionApprovalResult with approval status and any conflicts found.
        """
        conflicts = []

        # Check against all previously approved trajectories
        for approved_traj in self.approved_trajectories.values():
            trajectory_conflicts = self._check_trajectory_conflicts(
                trajectory, approved_traj, safety_buffer
            )
            conflicts.extend(trajectory_conflicts)

        # Determine approval status
        approved = len(conflicts) == 0

        # Store the trajectory in appropriate list
        if approved:
            self.approved_trajectories[trajectory.drone_id] = trajectory
        else:
            self.rejected_trajectories[trajectory.drone_id] = trajectory

        # Create and store result
        result = MissionApprovalResult(trajectory.drone_id, approved, conflicts)
        self.approval_history.append(result)

        return result

    def _check_trajectory_conflicts(self, traj1: Trajectory, traj2: Trajectory, safety_buffer: float) -> List[Conflict]:
        """
        Check for conflicts between two trajectories.
        Returns list of Conflict objects found.
        """
        conflicts = []

        for i in range(len(traj1.waypoints) - 1):
            wp1_1 = traj1.waypoints[i]
            wp1_2 = traj1.waypoints[i+1]

            for j in range(len(traj2.waypoints) - 1):
                wp2_1 = traj2.waypoints[j]
                wp2_2 = traj2.waypoints[j+1]

                # Check if time intervals overlap
                time_overlap = (wp1_1.timestamp <= wp2_2.timestamp) and \
                              (wp2_1.timestamp <= wp1_2.timestamp)

                if not time_overlap:
                    continue

                # Calculate minimum distance between trajectory segments
                min_dist = min_distance_between_segments(
                    wp1_1.position, wp1_2.position,
                    wp2_1.position, wp2_2.position
                )

                if min_dist < safety_buffer:
                    # Calculate conflict time and location
                    overlap_start = max(wp1_1.timestamp, wp2_1.timestamp)
                    overlap_end = min(wp1_2.timestamp, wp2_2.timestamp)
                    conflict_time = (overlap_start + overlap_end) / 2.0

                    loc1 = traj1.get_state_at_time(conflict_time)
                    loc2 = traj2.get_state_at_time(conflict_time)

                    if loc1 is not None and loc2 is not None:
                        conflict_location = (loc1 + loc2) / 2.0

                        conflict = Conflict(
                            time_of_conflict=conflict_time,
                            location_of_conflict=conflict_location,
                            conflicting_drone_ids=(traj1.drone_id, traj2.drone_id),
                            minimum_separation=min_dist
                        )
                        conflicts.append(conflict)

        return conflicts

    def get_all_trajectories(self) -> List[Trajectory]:
        """Get all trajectories (both approved and rejected) for visualization"""
        all_trajectories = list(self.approved_trajectories.values()) + \
                          list(self.rejected_trajectories.values())
        return all_trajectories

    def get_all_conflicts(self) -> List[Conflict]:
        """Get all conflicts from all approval attempts"""
        all_conflicts = []
        for result in self.approval_history:
            all_conflicts.extend(result.conflicts)
        return all_conflicts

    def get_approval_summary(self) -> Dict:
        """Get summary of all approval attempts"""
        return {
            'total_submitted': len(self.approval_history),
            'approved': len(self.approved_trajectories),
            'rejected': len(self.rejected_trajectories),
            'total_conflicts': sum(len(result.conflicts) for result in self.approval_history)
        }

    # Legacy method for backward compatibility
    def add_simulated_trajectory(self, trajectory: Trajectory):
        """Legacy method - adds trajectory directly to approved list (for backward compatibility)"""
        self.approved_trajectories[trajectory.drone_id] = trajectory

    def validate_mission(self, primary_trajectory: Trajectory, safety_buffer: float) -> List[Conflict]:
        """Legacy method - validates against all approved trajectories"""
        result = self.submit_mission_for_approval(primary_trajectory, safety_buffer)
        return result.conflicts
