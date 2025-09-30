# src/deconfliction_service.py

from typing import List, Dict
import numpy as np
from.data_structures import Trajectory, Conflict

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


class DeconflictionServer:
    """A centralized service to manage and validate drone trajectories."""

    def __init__(self):
        self.simulated_trajectories: Dict = {}

    def add_simulated_trajectory(self, trajectory: Trajectory):
        """Adds a known trajectory to the server's airspace database."""
        self.simulated_trajectories[trajectory.drone_id] = trajectory

    def validate_mission(self, primary_trajectory: Trajectory, safety_buffer: float) -> List[Conflict]:
        """
        Validates a primary mission against all simulated trajectories.
        Returns a list of Conflict objects if any are found, otherwise an empty list.
        """
        conflicts = []

        for i in range(len(primary_trajectory.waypoints) - 1):
            p_wp1 = primary_trajectory.waypoints[i]
            p_wp2 = primary_trajectory.waypoints[i+1]
            
            for sim_traj in self.simulated_trajectories.values():
                for j in range(len(sim_traj.waypoints) - 1):
                    s_wp1 = sim_traj.waypoints[j]
                    s_wp2 = sim_traj.waypoints[j+1]

                    time_overlap = (p_wp1.timestamp <= s_wp2.timestamp) and \
                                   (s_wp1.timestamp <= p_wp2.timestamp)

                    if not time_overlap:
                        continue

                    min_dist = min_distance_between_segments(
                        p_wp1.position, p_wp2.position,
                        s_wp1.position, s_wp2.position
                    )

                    if min_dist < safety_buffer:
                        overlap_start = max(p_wp1.timestamp, s_wp1.timestamp)
                        overlap_end = min(p_wp2.timestamp, s_wp2.timestamp)
                        conflict_time = (overlap_start + overlap_end) / 2.0
                        
                        loc1 = primary_trajectory.get_state_at_time(conflict_time)
                        loc2 = sim_traj.get_state_at_time(conflict_time)
                        
                        if loc1 is not None and loc2 is not None:
                            conflict_location = (loc1 + loc2) / 2.0
                            
                            conflict = Conflict(
                                time_of_conflict=conflict_time,
                                location_of_conflict=conflict_location,
                                conflicting_drone_ids=(primary_trajectory.drone_id, sim_traj.drone_id),
                                minimum_separation=min_dist
                            )
                            conflicts.append(conflict)
        return conflicts