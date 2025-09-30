# src/data_structures.py

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np

@dataclass
class Waypoint:
    """Represents a single point in a drone's trajectory, with position and time."""
    position: np.ndarray  # 3D position as a NumPy array [x, y, z]
    timestamp: float      # Time in seconds from the start of the mission

@dataclass
class Trajectory:
    """Represents the full flight path of a single drone."""
    drone_id: str
    waypoints: List[Waypoint]

    def get_state_at_time(self, t: float) -> np.ndarray:
        """
        Calculates the drone's position at a specific time 't' using linear interpolation.
        Returns a NumPy array for the position. Returns None if t is outside the trajectory's time range.
        """
        # Ensure waypoints are sorted by timestamp
        self.waypoints.sort(key=lambda wp: wp.timestamp)

        # ← FIXED: Added  to access first waypoint's timestamp
        if not self.waypoints or t < self.waypoints[0].timestamp or t > self.waypoints[-1].timestamp:
            return None

        for i in range(len(self.waypoints) - 1):
            wp1 = self.waypoints[i]
            wp2 = self.waypoints[i+1]

            if wp1.timestamp <= t <= wp2.timestamp:
                # Linearly interpolate between wp1 and wp2
                time_interval = wp2.timestamp - wp1.timestamp
                if time_interval == 0:
                    return wp1.position
                ratio = (t - wp1.timestamp) / time_interval
                position = wp1.position + ratio * (wp2.position - wp1.position)
                return position

        # If t is exactly the last waypoint's timestamp
        if t == self.waypoints[-1].timestamp:
            return self.waypoints[-1].position

        return None

@dataclass
class Conflict:
    """Stores detailed information about a detected conflict."""
    time_of_conflict: float
    location_of_conflict: np.ndarray
    conflicting_drone_ids: Tuple[str, str]
    minimum_separation: float
